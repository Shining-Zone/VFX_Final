# VFX_Final

### 
train on KITTI dataset video: ```00, 01, 02, 05, 08, 09```  
valid on KITTI dataset video: ```04, 06, 07, 10```  
test on KITTI dataset video: ```04, 05, 07, 09, 10```  
test on Self-made dataset video: ```ntu, room, campus1, campus2```  

****
## <center>Pytorch DeepVo Implementation</center>
TBD

## Usage
1. Download our dataset  
https://drive.google.com/drive/folders/1DVB0K2cufUY0mSzXrByesJdHrs4bZqDf?usp=sharing  
2. Download our pretrain model
```
wget
```

## Result
#### 0. Learning Curve
<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/record/lc_0_100.png" width="840">
<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/record/lc_100_200.png" width="840">
<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/record/lc_200_250.png" width="630">

#### 1. test on KITTI dataset video: ```04, 05, 07, 09, 10```  
paper result

|pre-trained model from [alexart13](https://github.com/alexart13) |  our model  |
| ------------- | -------------------- |
|<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/ref_result/route_04_gradient.png" width="420"> | <img src="https://github.com/Shining-Zone/VFX_Final/blob/master/result/route_04_gradient.png" width="420"> |
|<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/ref_result/route_05_gradient.png" width="420"> | <img src="https://github.com/Shining-Zone/VFX_Final/blob/master/result/route_05_gradient.png" width="420">|
|<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/ref_result/route_07_gradient.png" width="420"> |<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/result/route_07_gradient.png" width="420"> |
|<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/ref_result/route_09_gradient.png" width="420"> | <img src="https://github.com/Shining-Zone/VFX_Final/blob/master/result/route_09_gradient.png" width="420"> |
|<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/ref_result/route_10_gradient.png" width="420"> | <img src="https://github.com/Shining-Zone/VFX_Final/blob/master/result/route_10_gradient.png" width="420"> |

#### 2. test on Self-made dataset video: ```ntu, room, campus1, campus2```  
4 images

#### 3. Rviz visualizing demo
video

****
|Author|陳健倫|李尚倫|李佳蓮|
|---|---|---|---|
****

reference
