#include <ros/ros.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <nav_msgs/Odometry.h>

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
  //ros pos publisher
  ros::Publisher odom_pub = n.advertise<nav_msgs::Odometry>("odom", 50);
  //ros tf broadcaster
  tf::TransformBroadcaster odom_broadcaster;
  //ros img publisher
  image_transport::ImageTransport it(n);
  image_transport::Publisher img_pub = it.advertise("camera/image", 10);
  //ros time
  ros::Time current_time, last_time;
  current_time = ros::Time::now();
  last_time = ros::Time::now();

  /* =============initailize pose=============*/
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;
  double qw = 0.0;
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
  getline(input,line);
  getline(input,line);
  getline(input,line);
  while(getline(input,line))
  {
      stringstream lineStream(line);
      string cell;
      vector<string> parsedRow;
      while(getline(lineStream,cell,' '))
      {
          parsedRow.push_back(cell);
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
  ROS_INFO("get the odomFile at: %s",imgFile_path.c_str());
  DIR *dir;
  struct dirent *ent;
  dir = opendir (imgFile_path.c_str());

  /* ==========================================*/
  /* ===============start looping===============*/
  /* ==========================================*/
  int iter = 0;
  ros::Rate r(10.0);//10Hz
  while(n.ok()){
    ros::spinOnce();// check for incoming messages
    current_time = ros::Time::now();

    /* =============calculating pose=============*/
    //compute odometry in a typical way given the velocities of the robot
    x = atof(parsedCsv[iter][1].c_str());
    y = atof(parsedCsv[iter][2].c_str());
    z = atof(parsedCsv[iter][3].c_str());
    qx = atof(parsedCsv[iter][4].c_str());
    qy = atof(parsedCsv[iter][5].c_str());
    qz = atof(parsedCsv[iter][6].c_str());
    qw = atof(parsedCsv[iter][7].c_str());

    //since all odometry is 6DOF we'll need a quaternion created from yaw
    geometry_msgs::Quaternion odom_quat;
    odom_quat.x = qx;
    odom_quat.y = qy;
    odom_quat.z = qz;
    odom_quat.w = qw;

    /* =============broadcast tf=============*/
    //first, we'll publish the transform over tf
    geometry_msgs::TransformStamped odom_trans;
    odom_trans.header.stamp = current_time;
    odom_trans.header.frame_id = "odom";
    odom_trans.child_frame_id = "base_link";

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
    odom.header.frame_id = "odom";

    //set the position
    odom.pose.pose.position.x = x;
    odom.pose.pose.position.y = y;
    odom.pose.pose.position.z = z;
    odom.pose.pose.orientation = odom_quat;

    //set the velocity
    odom.child_frame_id = "base_link";
    odom.twist.twist.linear.x = vx;
    odom.twist.twist.linear.y = vy;
    odom.twist.twist.linear.z = vz;
    odom.twist.twist.angular.x = vroll;
    odom.twist.twist.angular.y = vpitch;
    odom.twist.twist.angular.z = vyaw;

    //publish the message
    odom_pub.publish(odom);

    /* =============publish image topic=============*/
    stringstream image_path;
    if ((ent = readdir (dir)) != NULL){
      image_path << imgFile_path << "/" << ent->d_name;
      std::string image_path_s = image_path.str();
      cv::Mat image = cv::imread(image_path_s, CV_LOAD_IMAGE_COLOR);
      cv::waitKey(30);
      sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
      img_pub.publish(img_msg);
    }

    iter++;
    last_time = current_time;
    r.sleep();
  }
}
