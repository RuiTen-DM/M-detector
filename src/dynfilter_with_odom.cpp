#include <ros/ros.h>
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <iostream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <types.h>
#include <m-detector/DynObjFilter.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <pcl/filters/random_sample.h>
#include <Eigen/Eigen>
#include <eigen_conversions/eigen_msg.h>

#include <deque>

// #include "preprocess.h"

using namespace std;

shared_ptr<DynObjFilter> DynObjFilt(new DynObjFilter());
M3D cur_rot = Eigen::Matrix3d::Identity();
V3D cur_pos = Eigen::Vector3d::Zero();

int     QUAD_LAYER_MAX  = 1;
int     occlude_windows = 3;
int     point_index = 0;
float   VER_RESOLUTION_MAX  = 0.01;
float   HOR_RESOLUTION_MAX  = 0.01;
float   angle_noise     = 0.001;
float   angle_occlude     = 0.02;
float   dyn_windows_dur = 0.5;
bool    dyn_filter_en = true, dyn_filter_dbg_en = true;
string  points_topic, odom_topic;
string  out_folder, out_folder_origin;
double  lidar_end_time = 0;
int     dataset = 0;
int     cur_frame = 0;

deque<M3D> buffer_rots;
deque<V3D> buffer_poss;
deque<double> buffer_times;
deque<boost::shared_ptr<PointCloudXYZI>> buffer_pcs;
deque<double> buffer_pc_times;

// Pose of the LiDAR in the odometry body frame. FAST-LIO publishes /Odometry and
// /cloud_registered_body in the IMU frame, while M-detector's depth images must be
// centred on the LiDAR origin. Default (identity) keeps the original behaviour.
M3D  lidar_in_body_R = M3D::Identity();
V3D  lidar_in_body_T = V3D::Zero();
bool use_lidar_extrinsic = false;
double pair_time_tol = 1e-3;

ros::Publisher pub_pcl_dyn, pub_pcl_dyn_extend, pub_pcl_std, pub_pcl_static_frame; 

void OdomCallback(const nav_msgs::Odometry &cur_odom)
{
    Eigen::Quaterniond cur_q;
    geometry_msgs::Quaternion tmp_q;
    tmp_q = cur_odom.pose.pose.orientation;
    tf::quaternionMsgToEigen(tmp_q, cur_q);
    cur_rot = cur_q.matrix();
    cur_pos << cur_odom.pose.pose.position.x, cur_odom.pose.pose.position.y, cur_odom.pose.pose.position.z;
    if (use_lidar_extrinsic)
    {
        // body pose -> LiDAR pose
        cur_pos = cur_rot * lidar_in_body_T + cur_pos;
        cur_rot = cur_rot * lidar_in_body_R;
    }
    buffer_rots.push_back(cur_rot);
    buffer_poss.push_back(cur_pos);
    lidar_end_time = cur_odom.header.stamp.toSec();
    buffer_times.push_back(lidar_end_time);
}

void PointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg_in)
{
    boost::shared_ptr<PointCloudXYZI> feats_undistort(new PointCloudXYZI());
    pcl::fromROSMsg(*msg_in, *feats_undistort);
    if (use_lidar_extrinsic)
    {
        // body (IMU) frame -> LiDAR frame
        const M3D R_lb = lidar_in_body_R.transpose();
        for (auto &pt : feats_undistort->points)
        {
            V3D p_l = R_lb * (V3D(pt.x, pt.y, pt.z) - lidar_in_body_T);
            pt.x = p_l(0); pt.y = p_l(1); pt.z = p_l(2);
        }
    }
    buffer_pcs.push_back(feats_undistort); 
    buffer_pc_times.push_back(msg_in->header.stamp.toSec());
}


void TimerCallback(const ros::TimerEvent& e)
{
    // FAST-LIO stamps the scan and its odometry with the same lidar_end_time: drop
    // whichever side has no partner instead of pairing the queues blindly by index.
    while (buffer_pcs.size() > 0 && buffer_times.size() > 0)
    {
        double dt = buffer_pc_times.front() - buffer_times.front();
        if (fabs(dt) <= pair_time_tol) break;
        if (dt < 0)
        {
            ROS_WARN("dynfilter: drop cloud %.6f without odometry", buffer_pc_times.front());
            buffer_pcs.pop_front();
            buffer_pc_times.pop_front();
        }
        else
        {
            buffer_rots.pop_front();
            buffer_poss.pop_front();
            buffer_times.pop_front();
        }
    }
    if(buffer_pcs.size() > 0 && buffer_poss.size() > 0 && buffer_rots.size() > 0 && buffer_times.size() > 0)
    {
        boost::shared_ptr<PointCloudXYZI> cur_pc = buffer_pcs.at(0);
        buffer_pcs.pop_front();
        buffer_pc_times.pop_front();
        auto cur_rot = buffer_rots.at(0);
        buffer_rots.pop_front();
        auto cur_pos = buffer_poss.at(0);
        buffer_poss.pop_front();
        auto cur_time = buffer_times.at(0);
        buffer_times.pop_front();
        string file_name = out_folder;
        stringstream ss;
        ss << setw(6) << setfill('0') << cur_frame ;
        file_name += ss.str(); 
        file_name.append(".label");
        string file_name_origin = out_folder_origin;
        stringstream sss;
        sss << setw(6) << setfill('0') << cur_frame ;
        file_name_origin += sss.str(); 
        file_name_origin.append(".label");

        if(file_name.length() > 15 || file_name_origin.length() > 15)
            DynObjFilt->set_path(file_name, file_name_origin);

        DynObjFilt->filter(cur_pc, cur_rot, cur_pos, cur_time);
        DynObjFilt->publish_dyn(pub_pcl_dyn, pub_pcl_dyn_extend, pub_pcl_std, cur_time);
        if (pub_pcl_static_frame.getNumSubscribers() > 0)
        {
            // per-frame static points in world frame (complement of /m_detector/frame_out)
            PointCloudXYZI::Ptr steady = DynObjFilt->cluster_coupled ? DynObjFilt->laserCloudSteadObj_clus
                                                                     : DynObjFilt->laserCloudSteadObj;
            sensor_msgs::PointCloud2 msg;
            pcl::toROSMsg(*steady, msg);
            msg.header.stamp = ros::Time().fromSec(cur_time);
            msg.header.frame_id = DynObjFilt->frame_id;
            pub_pcl_static_frame.publish(msg);
        }
        cur_frame ++;
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "dynfilter_odom");
    ros::NodeHandle nh;
    nh.param<string>("dyn_obj/points_topic", points_topic, "");
    nh.param<string>("dyn_obj/odom_topic", odom_topic, "");
    nh.param<string>("dyn_obj/out_file", out_folder,"");
    nh.param<string>("dyn_obj/out_file_origin", out_folder_origin, "");
    vector<double> ext_T, ext_R;
    nh.param<vector<double>>("dyn_obj/lidar_in_body_T", ext_T, vector<double>());
    nh.param<vector<double>>("dyn_obj/lidar_in_body_R", ext_R, vector<double>());
    nh.param<double>("dyn_obj/pair_time_tol", pair_time_tol, 1e-3);
    if (ext_T.size() == 3)
    {
        lidar_in_body_T << ext_T[0], ext_T[1], ext_T[2];
        use_lidar_extrinsic = true;
    }
    if (ext_R.size() == 9)
    {
        lidar_in_body_R << ext_R[0], ext_R[1], ext_R[2],
                           ext_R[3], ext_R[4], ext_R[5],
                           ext_R[6], ext_R[7], ext_R[8];
        use_lidar_extrinsic = true;
    }
    if (use_lidar_extrinsic)
        ROS_INFO_STREAM("dynfilter: LiDAR-in-body extrinsic T = " << lidar_in_body_T.transpose()
                        << ", R =\n" << lidar_in_body_R);

    DynObjFilt->init(nh);    
    /*** ROS subscribe and publisher initialization ***/
    pub_pcl_dyn_extend = nh.advertise<sensor_msgs::PointCloud2>("/m_detector/frame_out", 10000);  
    pub_pcl_dyn = nh.advertise<sensor_msgs::PointCloud2> ("/m_detector/point_out", 100000);
    pub_pcl_std  = nh.advertise<sensor_msgs::PointCloud2> ("/m_detector/std_points", 100000);   
    pub_pcl_static_frame = nh.advertise<sensor_msgs::PointCloud2> ("/m_detector/static_points", 100000);
    ros::Subscriber sub_pcl = nh.subscribe(points_topic, 200000, PointsCallback);
    ros::Subscriber sub_odom = nh.subscribe(odom_topic, 200000, OdomCallback);
    ros::Timer timer = nh.createTimer(ros::Duration(0.01), TimerCallback);

    ros::spin();
    return 0;
}
