# Face Mask Detector

## Step
`pip3 install -r requirements.txt`

## Run
`python3 run.py -h`
```
optional arguments:
  -h, --help         show this help message and exit
  --version VERSION  The version of pretrained model, now support "v1" and
                     "v2".
  --mode MODE        The format of input data, now support "image" of jpg and
                     "video" of mp4.
  --data DATA        The path of input and output file.
```

## Training
the training process is done using jupyter notebook you can find all the step in the links below

[Local Notebook](https://github.com/DiaaZiada/face-mask/blob/main/Avidbeam.ipynb)
[colab Notebook](https://colab.research.google.com/drive/1uPOkG-kQvi5T6Xnqm8aq4KD9l04wPb-R?usp=sharing)

**dataset:** [Here](https://www.kaggle.com/omkargurav/face-mask-dataset)

the model reached **90%** accuracy on the validation set

loss graph

![graph](https://github.com/DiaaZiada/face-mask/blob/main/images/download.png)

## Preformance 
**LFFD** : ~ **3** FPS

**Face Mask detection** ~ **130** FPS

**Resolution** :  854x480

## hardware specs
CPU: Inter Core i7 7th gen 4 cores

RAM: 8GB

## Results
![image](https://github.com/DiaaZiada/face-mask/blob/main/images/021121_ts_mask_feat_result.png)

![image](https://github.com/DiaaZiada/face-mask/blob/main/images/1586526013004_result.png)


![image](https://github.com/DiaaZiada/face-mask/blob/main/images/960x0_result.png)

![image](https://github.com/DiaaZiada/face-mask/blob/main/images/face-masks-give-back-kr-2x1-tease-200602-1575506_result.png)

![image](https://github.com/DiaaZiada/face-mask/blob/main/images/katie-holmes_result.png)

![image](https://github.com/DiaaZiada/face-mask/blob/main/images/stock-photo-set-of-people-faces-498141919_result.png)
