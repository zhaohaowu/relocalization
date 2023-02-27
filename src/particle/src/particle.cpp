#include <bits/stdc++.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <nav_msgs/Odometry.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_datatypes.h>

#define pi 3.1415926
using namespace std;

class Particle{
    ros::NodeHandle n;
    ros::Subscriber sub_cloud;
    ros::Subscriber sub_init_odom;
    ros::Subscriber sub_wheel;
    ros::Publisher pub_odom;
    deque<sensor_msgs::PointCloud2> cloudQueue;
    deque<nav_msgs::Odometry> initOdomQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;
    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudIn;
    double x0,y0,z0,roll0,pitch0,yaw0;
    double particle_num;
    double submap_size;
    double resolution;
    double v, w;
    vector<vector<double>> particles;
    ros::Time current_time, last_time = ros::Time::now();
    
public:
    Particle():
    laserCloudIn(new pcl::PointCloud<pcl::PointXYZI>)
    {
        sub_cloud = n.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 5, &Particle::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        sub_init_odom = n.subscribe<nav_msgs::Odometry>("/initial_odom", 5, &Particle::initOdomHandler, this, ros::TransportHints().tcpNoDelay());
        sub_wheel = n.subscribe<nav_msgs::Odometry>("/odom", 5, &Particle::getWheelHandler, this, ros::TransportHints().tcpNoDelay());
        pub_odom = n.advertise<nav_msgs::Odometry>("/particle_odom",5);
        n.param<double>("x0", x0, 0);
        n.param<double>("y0", y0, 0);
        n.param<double>("yaw0", yaw0, 0);
        n.param<double>("particle_num", particle_num, 1600);
        n.param<double>("submap_size", submap_size, 4);
        n.param<double>("resolution", resolution, 0.2);
    }
    void initOdomHandler(const nav_msgs::OdometryConstPtr& msgIn){
        static bool initPose = true;
        if(initPose){
            initPose = false;
            x0 = msgIn->pose.pose.position.x;
            y0 = msgIn->pose.pose.position.y;
            z0 = msgIn->pose.pose.position.z;
            tf::Quaternion orientation;
            tf::quaternionMsgToTF(msgIn->pose.pose.orientation, orientation);
            tf::Matrix3x3(orientation).getRPY(roll0, pitch0, yaw0);
        }
    }
    void getWheelHandler(const nav_msgs::OdometryConstPtr& msgIn){
        v = msgIn->twist.twist.linear.x;
        w = msgIn->twist.twist.angular.z;
    }
    void laserCloudInfoHandler(const sensor_msgs::PointCloud2ConstPtr& msgIn){
        pcl::fromROSMsg(*msgIn, *laserCloudIn);
        
        particles = initParticle();

        particles = motionModel(particles);
    
        if(v>0.2){
            particles = update_weights(particles, laserCloudIn);

            particles = resample(particles);
        }
        cout << laserCloudIn->points[0].intensity << ' '
        << laserCloudIn->points[1].intensity << ' '
        << laserCloudIn->points[2].intensity << ' '
        << laserCloudIn->points[3].intensity << ' ';
    }
    vector<vector<double>> initParticle(){
        vector<vector<double>> particles(particle_num, vector<double>(4, 0));
        for(double i=-submap_size; i<submap_size; i+=resolution){
            for(double j=-submap_size; j<submap_size; j+=resolution){
                double theta = -pi + 2.0 * pi * rand()/double(RAND_MAX);
                particles.push_back({i+x0, j+y0, theta+yaw0, 1/particle_num});
            }
        }
        // for(int i=0; i<particle.size(); i++){
        //     cout << particle[i][0] << ' ' << particle[i][1] << ' ' << particle[i][2] << ' ' << particle[i][3]<< endl;
        // }
        return particles;
    }
    vector<vector<double>> motionModel(vector<vector<double>> particles){
        current_time = ros::Time::now();
        double dt = current_time.toSec() - last_time.toSec();
        last_time = current_time;
        for(int i=0; i<particles.size(); i++){
            double delta_x = v * cos(particles[i][2]) * dt;
            double delta_y = v * sin(particles[i][2]) * dt;
            double delta_th = w * dt;
            particles[i][0] += delta_x;
            particles[i][1] += delta_y;
            particles[i][2] += delta_th;
        }
        return particles;
    }
    vector<vector<double>> update_weights(vector<vector<double>> particles, pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudIn){
        
    }
    vector<vector<double>> resample(vector<vector<double>> particles){

    }
};
int main(int argc, char** argv){
    ros::init(argc, argv, "particle");
    Particle particle;
    ROS_INFO("\033[1;32m----> particle start.\033[0m");
    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();
    return 0;
}
