# Face Mask Detector based on LFFD architecture

## Dataset
after long research, I didn't found a dataset suitable for our situation

so I decided to create one 

you can find the codes from [here](https://github.com/DiaaZiada/face-mask/tree/face-maskV2/MaskTheFace)

you can find the full generated dataset from [here](https://drive.google.com/drive/folders/1u5seFz2lsbkjbXoE1uzS6eF27m1xDpy5)

and here some examples from the dataset

![image](https://github.com/DiaaZiada/face-mask/blob/face-maskV2/images/0.jpg)



![image](https://github.com/DiaaZiada/face-mask/blob/face-maskV2/images/1.jpg)


![image](https://github.com/DiaaZiada/face-mask/blob/face-maskV2/images/44.jpg)



![image](https://github.com/DiaaZiada/face-mask/blob/face-maskV2/images/22.jpg)



![image](https://github.com/DiaaZiada/face-mask/blob/face-maskV2/images/32.jpg)



![image](https://github.com/DiaaZiada/face-mask/blob/face-maskV2/images/48.jpg)

## Model Architecture

you can find the modified architecture [here](https://github.com/DiaaZiada/face-mask/blob/face-maskV2/lffd-pytorch/face_detection/net_farm/naivenet.py)

made the block layer return score, bbox, mask instead fo score, bbox only 


```py
def forward(self, x):
	x = self.conv1(x)
	x = self.relu1(x)
	x = self.layer1(x)
	score1, bbox1, mask1 = self.branch1(x)
	
	x = self.layer2(x)
	score2, bbox2, mask2 = self.branch2(x)
	
	x = self.layer3(x)
	score3, bbox3, mask3 = self.branch3(x)
	
	x = self.layer4(x)
	score4, bbox4, mask4 = self.branch4(x)
	
	x = self.layer5(x)
	score5, bbox5, mask5 = self.branch5(x)
	
	outs = [score1, bbox1, mask1, score2, bbox2, mask2, score3, bbox3, mask3, score4, bbox4, mask4, score5, bbox5, mask5]
	return outs
```


- [x] DataSet
- [x] Model Architecture
- [ ] Data Loader
- [ ] Loss Function

### DataLoader 
modify the existing data loader to accept an extra attribute which the face mask ground truth and do the preprocessing to it as the score attribute 

### Loss Function
calculate the loss of the face mask using Cross-Entropy loss and add it to the total loss
