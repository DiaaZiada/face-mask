from utils.aux_functions import *
import pickle
import dlib


def mask_face(image, face_location, six_points, angle, type="surgical"):
    debug = False

    # Find the face angle
    threshold = 13
    if angle < -threshold:
        type += "_right"
    elif angle > threshold:
        type += "_left"

    face_height = face_location[2] - face_location[0]
    face_width = face_location[1] - face_location[3]
    
    w = image.shape[0]
    h = image.shape[1]
    if not "empty" in type and not "inpaint" in type:
        cfg = read_cfg(config_filename="masks/masks.cfg", mask_type=type, verbose=False)
    else:
        if "left" in type:
            str = "surgical_blue_left"
        elif "right" in type:
            str = "surgical_blue_right"
        else:
            str = "surgical_blue"
        cfg = read_cfg(config_filename="masks/masks.cfg", mask_type=str, verbose=False)
    img = cv2.imread(cfg.template, cv2.IMREAD_UNCHANGED)
    img = color_the_mask(img, "#0473e2", 0.5)

    mask_line = np.float32(
        [cfg.mask_a, cfg.mask_b, cfg.mask_c, cfg.mask_f, cfg.mask_e, cfg.mask_d]
    )
    # Warp the mask
    M, mask = cv2.findHomography(mask_line, six_points)
    dst_mask = cv2.warpPerspective(img, M, (h, w))
    dst_mask_points = cv2.perspectiveTransform(mask_line.reshape(-1, 1, 2), M)
    mask = dst_mask[:, :, 3]
    face_height = face_location[2] - face_location[0]
    face_width = face_location[1] - face_location[3]
    image_face = image[
        face_location[0] + int(face_height / 2) : face_location[2],
        face_location[3] : face_location[1],
        :,
    ]

    image_face = image

    # Adjust Brightness
    mask_brightness = get_avg_brightness(img)
    img_brightness = get_avg_brightness(image_face)
    delta_b = 1 + (img_brightness - mask_brightness) / 255
    dst_mask = change_brightness(dst_mask, delta_b)

    # Adjust Saturation
    mask_saturation = get_avg_saturation(img)
    img_saturation = get_avg_saturation(image_face)
    delta_s = 1 - (img_saturation - mask_saturation) / 255
    dst_mask = change_saturation(dst_mask, delta_s)

    # Apply mask
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(image, image, mask=mask_inv)
    img_fg = cv2.bitwise_and(dst_mask, dst_mask, mask=mask)
    out_img = cv2.add(img_bg, img_fg[:, :, 0:3])
    if "empty" in type or "inpaint" in type:
        out_img = img_bg
    # Plot key points

    if "inpaint" in type:
        out_img = cv2.inpaint(out_img, mask, 3, cv2.INPAINT_TELEA)
        # dst_NS = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

    if debug:
        for i in six_points:
            cv2.circle(out_img, (i[0], i[1]), radius=4, color=(0, 0, 255), thickness=-1)

        for i in dst_mask_points:
            cv2.circle(
                out_img, (i[0][0], i[0][1]), radius=4, color=(0, 255, 0), thickness=-1
            )

    return out_img, mask



detector = dlib.get_frontal_face_detector()
path_to_dlib_model = "dlib_models/shape_predictor_68_face_landmarks.dat"
if not os.path.exists(path_to_dlib_model):
    download_dlib_model()
predictor = dlib.shape_predictor(path_to_dlib_model)

def mask_image(image):    
    face_locations = detector(image, 1)

    masked_images = []
    for (i, face_location) in enumerate(face_locations):
        shape = predictor(image, face_location)
        shape = face_utils.shape_to_np(shape)
        face_landmarks = shape_to_landmarks(shape)
        face_location = rect_to_bb(face_location)
        
        six_points_on_face, angle = get_six_points(face_landmarks, image)


        l = ["surgical", "N95", "KN95", "cloth"]
        mask_type = l[random.randint(0, len(l)-1)]

        if len(masked_images) > 0:
            image = masked_images.pop(0)

        image, mask_binary = mask_face(
            image, face_location, six_points_on_face, angle, type=mask_type
        )
        return image
#         masked_images.append(image)
#     return masked_images#, mask, mask_binary_array, original_image


import argparse

def arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output",required=True, type=str)

        
    return parser.parse_args()
def main():

    args = arguments()


    import copy
    with open(args.input, 'rb') as f:
        data_org = pickle.load(f)

    data = copy.deepcopy(data_org)
    n_faces = 0

    os.makedirs(args.output, exist_ok=True)


    import tqdm
    for k in tqdm.tqdm(list(data.keys())):
        if data[k][1] == 0:
            continue
        image = cv2.imdecode(data[k][0], cv2.IMREAD_COLOR)
        faces = data[k][2]
        data[k].append([])
        for face in faces:

            face = [int(point) for point in face]
            x,y, w,h = face
            subim = image[y:y+h,x:x+w]
            mask = mask_image(subim)
            if type(mask) == type(None) or random.randint(0,10) <=2 :
                data[k][3].append(0)
                continue
            data[k][3].append(1)

            image[y:y+h, x:x+w] = mask
        img_encode = cv2.imencode('.jpg', image)[1]
        data[k][0] = img_encode


        with open(os.path.join(args.output,f"{k}.pkl"), 'wb') as f:
            pickle.dump(data[k],f)
        
main()