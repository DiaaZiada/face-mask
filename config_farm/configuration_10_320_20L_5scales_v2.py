# -*- coding: utf-8 -*-
 
import math

'''
    algorithm
'''
# the number of image channels
param_num_image_channel = 3

# the number of output scales (loss branches)
param_num_output_scales = 5

# feature map size for each scale
param_feature_map_size_list = [159, 79, 39, 19, 9]

# bbox lower bound for each scale
param_bbox_small_list = [10, 20, 40, 80, 160]
assert len(param_bbox_small_list) == param_num_output_scales

# bbox upper bound for each scale
param_bbox_large_list = [20, 40, 80, 160, 320]
assert len(param_bbox_large_list) == param_num_output_scales

# bbox gray lower bound for each scale
param_bbox_small_gray_list = [math.floor(v * 0.9) for v in param_bbox_small_list]
# bbox gray upper bound for each scale
param_bbox_large_gray_list = [math.ceil(v * 1.1) for v in param_bbox_large_list]

# the RF size of each scale used for normalization, here we use param_bbox_large_list for better regression
param_receptive_field_list = param_bbox_large_list
# RF stride for each scale
param_receptive_field_stride = [4, 8, 16, 32, 64]
# the start location of the first RF of each scale
param_receptive_field_center_start = [3, 7, 15, 31, 63]

# the sum of the number of output channels, 2 channels for classification and 4 for bbox regression
param_num_output_channels = 6