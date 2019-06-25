#include <ros/ros.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>

#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace std;

// define infoMap as nx6 2d string array
vector<vector<string> > parsedCsv;

int main(int argc, char** argv){

  /* =============initailize ros=============*/
  ros::init(argc, argv, "odometry_publisher");
  //ros node
  ros::NodeHandle n;
  //ros publisher
  ros::Publisher odom_pub = n.advertise<nav_msgs::Odometry>("odom", 50);
  //ros tf broadcaster
  tf::TransformBroadcaster odom_broadcaster;
  //ros img publisher
  image_transport::ImageTransport it(n);
  image_transport::Publisher img_pub = it.advertise("camera/image", 10);
  //ros marker publisher
  ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 1);
  //ros time
  ros::Time current_time, last_time;
  current_time = ros::Time::now();
  last_time = ros::Time::now();

  /* =============initailize pose=============*/
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  double vx = 0.0;
  double vy = 0.0;
  double vz = 0.0;
  double vroll = 0.0;
  double vpitch = 0.0;
  double vyaw = 0.0;

  /* =============reading pose=============*/
  //get odom output file from ros parameter
  string odomFile_path;
  n.getParam("odomFile_path", odomFile_path);
  ROS_INFO("get the odomFile at: %s",odomFile_path.c_str());

  //readfile, read odometry points
  ifstream input(odomFile_path.c_str());
  string line;
  while(getline(input,line))
  {
      stringstream lineStream(line);
      string cell;
      vector<string> parsedRow;
      while(getline(lineStream,cell,' '))
      {
          string cell_editted = cell.substr (0,cell.length()-1);
          parsedRow.push_back(cell_editted);
      }
      parsedCsv.push_back(parsedRow);
  }
  // print 2d array
  /*for(int i=0; i<parsedCsv.size(); i++){
    for(int j=0; j<parsedCsv[0].size(); j++){
      cout << parsedCsv[i][j]<<",";
    }
    cout << endl;
  }*/
  ROS_INFO("Odometry points is loaded successfully...");
  ROS_INFO_STREAM("Odometry points number: "<<parsedCsv.size());

  /* =============reading image path=============*/
  //get odom output file from ros parameter
  string imgFile_path;
  n.getParam("imgFile_path", imgFile_path);
  ROS_INFO("get the imageFile at: %s",imgFile_path.c_str());
  DIR *dir;
  struct dirent *ent;
  std::vector<std::string> img_filename_array;
  dir = opendir (imgFile_path.c_str());
  ent = readdir (dir); //this is .
  ent = readdir (dir); //this is ..
  while ((ent = readdir (dir)) != NULL){
    stringstream image_path;
    image_path << imgFile_path << "/" << ent->d_name;
    img_filename_array.push_back(image_path.str());
  }
  std::sort(img_filename_array.begin(), img_filename_array.end());
  ROS_INFO("Image is loaded successfully...");
  ROS_INFO_STREAM("Image number: "<<img_filename_array.size());

  /* ==========================================*/
  /* ===============start looping===============*/
  /* ==========================================*/
  int iter = 0;
  ros::Rate r(10.0);
  while(n.ok()){
    ros::spinOnce();// check for incoming messages
    current_time = ros::Time::now();

    /* =============calculating pose=============*/
    //compute odometry in a typical way given the velocities of the robot
    x = atof(parsedCsv[iter][3].c_str());
    y = atof(parsedCsv[iter][4].c_str());
    z = atof(parsedCsv[iter][5].c_str());
    roll = atof(parsedCsv[iter][0].c_str());
    pitch = atof(parsedCsv[iter][1].c_str());
    yaw = atof(parsedCsv[iter][2].c_str());

    //since all odometry is 6DOF we'll need a quaternion created from yaw
    tf::Quaternion odom_quat_tf = tf::createQuaternionFromRPY(roll,pitch,yaw);
    geometry_msgs::Quaternion odom_quat;
    quaternionTFToMsg(odom_quat_tf, odom_quat);

    /* =============broadcast pose tf=============*/
    //first, we'll publish the transform over tf
    geometry_msgs::TransformStamped odom_trans;
    odom_trans.header.stamp = current_time;
    odom_trans.header.frame_id = "odom_top";
    odom_trans.child_frame_id = "camera_link";

    odom_trans.transform.translation.x = x;
    odom_trans.transform.translation.y = y;
    odom_trans.transform.translation.z = z;
    odom_trans.transform.rotation = odom_quat;

    //send the transform
    odom_broadcaster.sendTransform(odom_trans);

    /* =============publish pose topic=============*/
    //next, we'll publish the odometry message over ROS
    nav_msgs::Odometry odom;
    odom.header.stamp = current_time;
    odom.header.frame_id = "odom_top";

    //set the position
    odom.pose.pose.position.x = x;
    odom.pose.pose.position.y = y;
    odom.pose.pose.position.z = z;
    odom.pose.pose.orientation = odom_quat;

    //set the velocity
    odom.child_frame_id = "camera_link";
    odom.twist.twist.linear.x = vx;
    odom.twist.twist.linear.y = vy;
    odom.twist.twist.linear.z = vz;
    odom.twist.twist.angular.x = vroll;
    odom.twist.twist.angular.y = vpitch;
    odom.twist.twist.angular.z = vyaw;

    //publish the message
    odom_pub.publish(odom);

    /* =============publish image topic=============*/
    //std::string image_path_s = image_path.str();
    cv::Mat image = cv::imread(img_filename_array[iter], CV_LOAD_IMAGE_COLOR);
    cv::waitKey(30);
    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    img_pub.publish(img_msg);

    /* =============publish marker topic=============*/
    while(marker_pub.getNumSubscribers() == 0){
      ROS_INFO("wait for marker!");
    }
    visualization_msgs::Marker object;
    object.color.r = 1.0;
    object.color.g = 1.0;
    object.color.b = 0.0;
    object.pose.position= odom.pose.pose.position;
    object.header.frame_id = "odom_top";
    object.header.stamp = ros::Time::now();
    object.color.a = 0.5;
    object.pose.orientation.w = 1.0;
    object.scale.x = 2.5;
    object.scale.y = 2.5;
    object.scale.z = 1.1;
    object.type = visualization_msgs::Marker::CYLINDER;
    object.action = visualization_msgs::Marker::ADD;
    object.id = iter;
    marker_pub.publish(object);

    /* =============the end of loop=============*/
    iter++;
    last_time = current_time;
    r.sleep();
  }
}
