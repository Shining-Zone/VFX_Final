## VFX_Final: DeepVO pytorch implementation and Rviz visualization
<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/doc/model1.png" height="170"><img src="https://github.com/Shining-Zone/VFX_Final/blob/master/doc/model2.png" height="170">
description
****

## I.Usage
1. Download KITTI dataset  
```
cd KITTI/
bash downloader.sh
```
2. Download our dataset, uzip them and put them into KITTI/image/  
```
download https://drive.google.com/drive/folders/1DVB0K2cufUY0mSzXrByesJdHrs4bZqDf?usp=sharing  
ntu_30fstep unzip as KITTI/image/ntu
room_1fstep unzip as KITTI/image/room
campus1_2fstep unzip as KITTI/image/campus1
campus2_2fstep unzip as KITTI/image/campus2
```
3. Download our pretrain mode, and put it into model/
```
mkdir model
cd model
wget https://www.dropbox.com/s/0or826j6clrbh3h/DeepVo_Epoch_Last.pth?dl=0
```
4. Specify your path in ```myMain.py, myTest.py, myTestNoGT.py, myVisualize.py, myVisualizeNoGT.py```
```
GTpost...TBD
```
5. (optional) Training your own model (you may need [flownet pretrain model](https://drive.google.com/drive/folders/0B5EC7HMbyk3CbjFPb0RuODI3NmM)
```
python3 myMain.py
```
6. Predict the KITTI dataset pose and our dataset pose
```
python3 myTest.py
python3 myTestNoGT.py
```
7. Visualize the prediction of KITTI and our dataset
```
python3 myVisualize.py
python3 myVisualizeNoGT.py
```
8. Visualize poses dynamically by Rviz (ROS Kinetic required)
```
cd catkin_ws/src
git clone https://github.com/shannon112/ros_odometry_visualizer.git
vim ros_odometry_visualizer/launch/odometry_kitti_visualizer    #edit your own path
roscd
cd ..
catkin_make
roslaunch ros_odomtry_visualizer odometry_kitti_visualizer.launch
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
