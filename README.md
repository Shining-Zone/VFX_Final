# VFX_Final

description
****

## I.Usage
1. Download our dataset  
https://drive.google.com/drive/folders/1DVB0K2cufUY0mSzXrByesJdHrs4bZqDf?usp=sharing  
2. Download our pretrain model
```
wget
```

## II.Data Usage
train on KITTI dataset video: ```00, 01, 02, 05, 08, 09```  
valid on KITTI dataset video: ```04, 06, 07, 10```  
test on KITTI dataset video: ```04, 05, 07, 09, 10```  
test on Self-made dataset video: ```ntu, room, campus1, campus2```  

## III.Result
#### 0. Learning Curve
<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/record/lc_0_100.png" width="420"><img src="https://github.com/Shining-Zone/VFX_Final/blob/master/record/lc_100_200.png" width="420">
<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/record/lc_200_250.png" width="320">

#### 1. test on KITTI dataset video: ```04, 05, 07, 09, 10```  
paper result

|paper result | pre-trained model from [alexart13](https://github.com/alexart13) |  our model  |
| ------------- |--------------| -------------------- |
| <img src="https://github.com/Shining-Zone/VFX_Final/blob/master/paper_result/04.png" width="280"> |<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/ref_result/route_04_gradient.png" width="280"> | <img src="https://github.com/Shining-Zone/VFX_Final/blob/master/result/route_04_gradient.png" width="280"> |
|<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/paper_result/05.png" width="280">|<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/ref_result/route_05_gradient.png" width="280"> | <img src="https://github.com/Shining-Zone/VFX_Final/blob/master/result/route_05_gradient.png" width="280">| 
|<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/paper_result/07.png" width="280"> |<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/ref_result/route_07_gradient.png" width="280"> |<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/result/route_07_gradient.png" width="280"> | 
|  |<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/ref_result/route_09_gradient.png" width="280"> | <img src="https://github.com/Shining-Zone/VFX_Final/blob/master/result/route_09_gradient.png" width="280"> |
|<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/paper_result/10.png" width="280">|<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/ref_result/route_10_gradient.png" width="280"> | <img src="https://github.com/Shining-Zone/VFX_Final/blob/master/result/route_10_gradient.png" width="280"> | 

#### 2. test on Self-made dataset video: ```ntu, room, campus1, campus2```  
| ntu           | ntu-ref      |  room                |  room-ref                |
| ------------- |--------------| -------------------- |-------------------- |
|<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/result/route_ntu_gradient.png" width="210">| <img src="https://github.com/Shining-Zone/VFX_Final/blob/master/result/ntu_GT.png" width="210"> | <img src="https://github.com/Shining-Zone/VFX_Final/blob/master/result/route_room_gradient.png" width="210"> | <img src="https://github.com/Shining-Zone/VFX_Final/blob/master/result/room_GT.png" width="210"> |
| **campus1** | **campus1-ref** | **campus2** | **campus2-ref** |
|<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/result/route_campus1_gradient.png" width="210">| <img src="https://github.com/Shining-Zone/VFX_Final/blob/master/result/campus1_GT.png" width="210"> | <img src="https://github.com/Shining-Zone/VFX_Final/blob/master/result/route_campus2_gradient.png" width="210"> | <img src="https://github.com/Shining-Zone/VFX_Final/blob/master/result/campus2_GT.png" width="210"> |

#### 3. Rviz visualizing demo
video

## IV.Reference
[1] S. Wang, R. Clark, H. Wen and N. Trigoni, "DeepVO: Towards end-to-end visual odometry with deep Recurrent Convolutional Neural Networks," 2017 IEEE International Conference on Robotics and Automation (ICRA), Singapore, 2017, pp. 2043-2050.  
[2] https://github.com/ChiWeiHsiao/DeepVO-pytorch  

****
|Author|陳健倫|李尚倫|李佳蓮|
|---|---|---|---|
****
