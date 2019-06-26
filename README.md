## VFX_Final: DeepVO pytorch implementation and Rviz visualization
<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/doc/model1.png" height="170"><img src="https://github.com/Shining-Zone/VFX_Final/blob/master/doc/model2.png" height="170">
description
****

## I.Usage
1. Download KITTI dataset image as ```KITTI/image/```    
```
cd KITTI/
bash downloader.sh
```
2. Download KITTI dataset ground truth label as ```KITTI/pose_GT/```   
```
cd KITTI/
download: http://www.cvlibs.net/download.php?file=data_odometry_poses.zip
rename as KITTI/pose_GT 
```
3. Transfer ground truth pose from [R|t] to rpyxyzR as .npy into ```KITTI/pose_GT/``` for training  
3.5. Transfer .npy ground truth to rpyxyz into ```/GT_pose_rpyxyz``` for visualizing  
```
python3 preprocess.py
python3 myGTtxt_generator.py     # Need to specify your path
```
4. Download our dataset, uzip them and put them into ```KITTI/image/```  
```
cd KITTI/
download: https://drive.google.com/drive/folders/1DVB0K2cufUY0mSzXrByesJdHrs4bZqDf?usp=sharing  
images/ntu_15fstep unzip as KITTI/image/ntu
images/room_1fstep unzip as KITTI/image/room
images/campus1_2fstep unzip as KITTI/image/campus1
images/campus2_2fstep unzip as KITTI/image/campus2
images/ntu3_15tstep unzip as KITTI/image/ntu3
images/ntu4_15fstep unzip as KITTI/image/ntu4
move all things in pose_GT to KITTI/pose_GT
```
5. Download our pretrain model ```DeepVo_Epoch_Last.pth```, and put it into ```model/```
```
mkdir model
cd model
wget https://www.dropbox.com/s/0or826j6clrbh3h/DeepVo_Epoch_Last.pth?dl=1
```
6. Specify your path in ```myMain.py, myTest.py, myTestNoGT.py, myVisualize.py, myVisualizeNoGT.py```
```
gt_root to KITTI/pose_GT
img_root to KITTI/images
pose_GT_dir to KITTI/pose_GT
```
7. (optional) Training your own model (you may need [flownet pretrain model](https://drive.google.com/drive/folders/0B5EC7HMbyk3CbjFPb0RuODI3NmM)
```
python3 myMain.py
```
8. Predict the KITTI dataset pose and our dataset pose
```
python3 myTest.py
python3 myTestNoGT.py
```
9. Visualize the prediction of KITTI and our dataset
```
python3 myVisualize.py
python3 myVisualizeNoGT.py
```
10. Visualize poses dynamically by Rviz (ROS Kinetic required)
```
mv ros_odometry_visualizer catkin_ws/src/ros_odometry_visualizer
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

## III.Our dataset
```ntu, campus1, campus2``` are recorded by iPhone8 with riding bicycle  
```room``` is recorded by iPhone8 through walking  
All videos are processed by Blender to 1241x376 resolution png sequences  
<img src="https://github.com/Shining-Zone/VFX_Final/blob/master/doc/blender_setting.png" width="620">

## IV.Result
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

## V.Reference
[1] S. Wang, R. Clark, H. Wen and N. Trigoni, "DeepVO: Towards end-to-end visual odometry with deep Recurrent Convolutional Neural Networks," 2017 IEEE International Conference on Robotics and Automation (ICRA), Singapore, 2017, pp. 2043-2050.  
[2] https://github.com/ChiWeiHsiao/DeepVO-pytorch  

****
|Author|陳健倫|李尚倫|李佳蓮|
|---|---|---|---|
****
