#include "lidar_localization/loam_fea/image_projection.hpp"


namespace lidar_localization{

ImageProjection::ImageProjection(const LidarParam& lidar_param)
    :  lidar_param_(lidar_param)
{

    initParam();


    nan_point_.x = std::numeric_limits<float>::quiet_NaN();
    nan_point_.y = std::numeric_limits<float>::quiet_NaN();
    nan_point_.z = std::numeric_limits<float>::quiet_NaN();
    nan_point_.intensity = -1;

    allocateMemory();
    resetParameters();
}

void ImageProjection::initParam(){
    char abs_path[PATH_MAX];
    realpath(__FILE__, abs_path);
    std::string dirpath(abs_path);
    dirpath = dirpath.substr(0, dirpath.find_last_of("/"));
    dirpath = dirpath.substr(0, dirpath.find_last_of("/"));
    dirpath = dirpath.substr(0, dirpath.find_last_of("/"));
    std::string config_file_path = dirpath + "/config/image_projection/image_projection.yaml";
    YAML::Node config_node = YAML::LoadFile(config_file_path);
    if(!config_node)
        std::cout << "cannot Load ImageProjection.yaml." << std::endl; 

    sensor_min_range_ = config_node["sensor_min_range"].as<float>();
    sensor_segment_range_ = config_node["sensor_segment_range"].as<float>();
    sensor_mount_angle_ = config_node["sensor_mount_range"].as<float>();
    float degree = config_node["segment_theta"].as<float>();
    segment_theta_ = degree / 180.0* M_PI ;
    segment_valid_point_num_ = config_node["segment_valid_point_num"].as<int>();
    segment_valid_line_num_ = config_node["segment_valid_line_num"].as<int>();

}

void ImageProjection::allocateMemory(){
    laser_cloud_in_.reset(new pcl::PointCloud<PointType>());

    full_cloud_.reset(new pcl::PointCloud<PointType>());
    full_info_cloud_.reset(new pcl::PointCloud<PointType>());

    ground_cloud_.reset(new pcl::PointCloud<PointType>());
    ground_removed_cloud_.reset(new pcl::PointCloud<PointType>());
    segmented_cloud_.reset(new pcl::PointCloud<PointType>());
    segmented_cloud_pure_.reset(new pcl::PointCloud<PointType>());
    outlier_cloud_.reset(new pcl::PointCloud<PointType>());
    
    full_cloud_->points.resize(lidar_param_.N_SCAN * lidar_param_.Horizon_SCAN);
    full_info_cloud_->points.resize(lidar_param_.N_SCAN * lidar_param_.Horizon_SCAN);

    seg_msg_.startRingIndex.assign(lidar_param_.N_SCAN, 0);
    seg_msg_.endRingIndex.assign(lidar_param_.N_SCAN, 0);
    seg_msg_.segmentedCloudGroundFlag.assign(lidar_param_.N_SCAN*lidar_param_.Horizon_SCAN, false);
    seg_msg_.segmentedCloudColInd.assign(lidar_param_.N_SCAN*lidar_param_.Horizon_SCAN, 0);
    seg_msg_.segmentedCloudRange.assign(lidar_param_.N_SCAN*lidar_param_.Horizon_SCAN, 0);


    queue_ind_x_ = new uint16_t[lidar_param_.N_SCAN * lidar_param_.Horizon_SCAN];
    queue_ind_y_ = new uint16_t[lidar_param_.N_SCAN * lidar_param_.Horizon_SCAN];

    all_pushed_ind_x_ = new uint16_t[lidar_param_.N_SCAN * lidar_param_.Horizon_SCAN];
    all_pushed_ind_y_ = new uint16_t[lidar_param_.N_SCAN * lidar_param_.Horizon_SCAN];

    std::pair<int8_t, int8_t> neighbor;
     neighbor.first = -1; neighbor.second =  0; neighbor_iterator_.push_back(neighbor);
    neighbor.first =  0; neighbor.second =  1; neighbor_iterator_.push_back(neighbor);
    neighbor.first =  0; neighbor.second = -1; neighbor_iterator_.push_back(neighbor);
    neighbor.first =  1; neighbor.second =  0; neighbor_iterator_.push_back(neighbor);
}


void ImageProjection::resetParameters(){
    laser_cloud_in_->clear();
    ground_cloud_->clear();
    ground_removed_cloud_->clear();
    segmented_cloud_->clear();
    segmented_cloud_pure_->clear();
    outlier_cloud_->clear();

    range_mat_ = cv::Mat(lidar_param_.N_SCAN, lidar_param_.Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
    ground_mat_ = cv::Mat(lidar_param_.N_SCAN, lidar_param_.Horizon_SCAN, CV_8S, cv::Scalar::all(0));
    label_mat_ = cv::Mat(lidar_param_.N_SCAN, lidar_param_.Horizon_SCAN, CV_32S, cv::Scalar::all(0));

    label_count_ = 1;
    
    std::fill(full_cloud_->points.begin(), full_cloud_->points.end(), nan_point_);
    std::fill(full_info_cloud_->points.begin(), full_info_cloud_->points.end(), nan_point_);
}


void ImageProjection::processCloud(pcl::PointCloud<PointType>::Ptr cloud_in){
    laser_cloud_in_ = cloud_in;

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*laser_cloud_in_, *laser_cloud_in_, indices);
    // Eigen::AngleAxisd pitch_angle(10.0 * 3.1415926 / 360.0, Eigen::Vector3d::UnitY());
    // Eigen::Isometry3d trans_mat = Eigen::Isometry3d::Identity();
    // trans_mat.prerotate(pitch_angle);
    // pcl::transformPointCloud(*laser_cloud_in_, *laser_cloud_in_, trans_mat.matrix());
    
    findStartEndAngle();
    projectPointCloud();
    groundRemoval();
    cloudSegmentation();
    // ?????????????????????
    // ??????get??????
    // TODO???????????????????????????resetParameters???????????????????????????
}

void ImageProjection::removeGround(pcl::PointCloud<PointType>::Ptr cloud_in){
    laser_cloud_in_ = cloud_in;

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*laser_cloud_in_, *laser_cloud_in_, indices);
    
    findStartEndAngle();
    projectPointCloud();
    groundRemoval();
    // ??????get????????????????????????????????????
    // TODO???????????????????????????resetParameters???????????????????????????
}

// ????????????????????????????????????????????????????????????
void ImageProjection::findStartEndAngle(){
    // ????????????
    seg_msg_.startOrientation  = - atan2(laser_cloud_in_->points[0].y, laser_cloud_in_->points[0].x);
    // ???????????????
    size_t point_num = laser_cloud_in_->points.size();
    seg_msg_.endOrientation  = - atan2(laser_cloud_in_->points[point_num-1].y, laser_cloud_in_->points[point_num-1].x) + 2 * M_PI;

    if (seg_msg_.endOrientation - seg_msg_.startOrientation > 3 * M_PI) {
        seg_msg_.endOrientation -= 2 * M_PI;
    } else if (seg_msg_.endOrientation - seg_msg_.startOrientation < M_PI)
        seg_msg_.endOrientation += 2 * M_PI;
    seg_msg_.orientationDiff = seg_msg_.endOrientation - seg_msg_.startOrientation;
}

void ImageProjection::projectPointCloud(){
    // range image projection
    float vertical_angle, horizon_angle, range;
    size_t row_ind, column_ind, index, cloud_size; 
    PointType this_point;

    cloud_size = laser_cloud_in_->points.size();

    for(size_t i = 0; i < cloud_size; ++i){
        this_point.x = laser_cloud_in_->points[i].x;
        this_point.y = laser_cloud_in_->points[i].y;
        this_point.z = laser_cloud_in_->points[i].z;
        // find the row and column index in the iamge for this point

        vertical_angle = atan2(this_point.z, sqrt(this_point.x * this_point.x + this_point.y * this_point.y)) * 180 / M_PI;
        // ?????????????????????0???-15???????????????????????lidar_param_.ang_bottom
        row_ind = (vertical_angle - lidar_param_.ang_bottom_) / lidar_param_.ang_res_y_;
        if (row_ind < 0 || row_ind >= lidar_param_.N_SCAN)
                continue;
        
        // ????????????????????????
        horizon_angle = atan2(this_point.x, this_point.y) * 180 / M_PI;

        column_ind = -round((horizon_angle-90.0)/lidar_param_.ang_res_x_) + lidar_param_.Horizon_SCAN/2;
        // ?????????????????????id?????????lidar_param_.Horizon_SCAN???????????????????????????????????????????????????
        if (column_ind >= lidar_param_.Horizon_SCAN)
                column_ind -= lidar_param_.Horizon_SCAN;

         if (column_ind < 0 || column_ind >= lidar_param_.Horizon_SCAN)
                continue;

        range = sqrt(this_point.x * this_point.x + this_point.y * this_point.y + this_point.z * this_point.z);
        if (range < sensor_min_range_)
                continue;

        range_mat_.at<float>(row_ind, column_ind) = range;
        this_point.intensity = (float)row_ind + (float)column_ind / 10000.0;

        index = column_ind + row_ind * lidar_param_.Horizon_SCAN; // ?????????id????????? * ???????????? + ??????
        full_cloud_->points[index] = this_point;
        // full_info_cloud???intensity ???????????????????????? range
        full_info_cloud_->points[index] = this_point;
        full_info_cloud_->points[index].intensity = range;
    }
}

void ImageProjection::groundRemoval(){
    size_t lower_ind, upper_ind;
    float diff_x, diff_y, diff_z, angle;
    // groundMat
    // -1, no valid info to check if ground of not
    //  0, initial value, after validation, means not ground
    //  1, ground
    for (size_t j = 0; j < lidar_param_.Horizon_SCAN; ++j){
        for (size_t i = 0; i < lidar_param_.ground_scan_id_; ++i){    // ???0~id-1?????????????????????
            // ????????????????????????????????????
            lower_ind = j + ( i ) * lidar_param_.Horizon_SCAN;   // 
            upper_ind = j + ( i + 1) * lidar_param_.Horizon_SCAN;

            if(full_cloud_->points[lower_ind].intensity == -1 || 
                full_cloud_->points[upper_ind].intensity == -1){
                    // no info to check, invalid points
                    ground_mat_.at<int8_t>(i,j) = -1;
                    continue;
            }
            // ????????????????????????
            diff_x = full_cloud_->points[upper_ind].x - full_cloud_->points[lower_ind].x;
            diff_y = full_cloud_->points[upper_ind].y - full_cloud_->points[lower_ind].y;
            diff_z = full_cloud_->points[upper_ind].z - full_cloud_->points[lower_ind].z;
            // ????????????????????????
            angle = atan2(diff_z, sqrt(diff_x*diff_x + diff_y*diff_y) ) * 180 / M_PI;
            // ??????????????????????????????10?????????????????????????????????????????????????????
            if (abs(angle - sensor_mount_angle_) <= 5){
                    ground_mat_.at<int8_t>(i,j) = 1;
                    ground_mat_.at<int8_t>(i+1,j) = 1;
                }
        }
    }

    // extract ground cloud (groundMat == 1)
    // mark entry that doesn't need to label (ground and invalid point) for segmentation
    // note that ground remove is from 0~lidar_param_.N_SCAN-1, need range_mat_ for mark label matrix for the 16th scan
    for (size_t i = 0; i < lidar_param_.N_SCAN; ++i){
        for (size_t j = 0; j < lidar_param_.Horizon_SCAN; ++j){
            if (ground_mat_.at<int8_t>(i,j) == 1 || range_mat_.at<float>(i,j) == FLT_MAX){
                label_mat_.at<int>(i,j) = -1;
            }
        }
    }

     for (size_t i = 0; i <= lidar_param_.ground_scan_id_; ++i){
        for (size_t j = 0; j < lidar_param_.Horizon_SCAN; ++j){
            if (ground_mat_.at<int8_t>(i,j) == 0)
                ground_removed_cloud_->push_back(full_cloud_->points[j + i*lidar_param_.Horizon_SCAN]);
        }
    }
    // ???????????????
    if (make_ground_cloud_flag_){
        for (size_t i = 0; i <= lidar_param_.ground_scan_id_; ++i){
            for (size_t j = 0; j < lidar_param_.Horizon_SCAN; ++j){
                if (ground_mat_.at<int8_t>(i,j) == 1)
                    ground_cloud_->push_back(full_cloud_->points[j + i*lidar_param_.Horizon_SCAN]);
            }
        }
    }
}

void ImageProjection::cloudSegmentation(){
    // segmentation process
    // // ???????????????????????????????????????????????????
    for (size_t i = 0; i < lidar_param_.N_SCAN; ++i)
            for (size_t j = 0; j < lidar_param_.Horizon_SCAN; ++j)
                if (label_mat_.at<int>(i,j) == 0)
                    labelComponents(i, j);

    int seg_cloud_size = 0;
    // extract segmented cloud for lidar odometry
    for(size_t i = 0; i < lidar_param_.N_SCAN; ++i){
        
        seg_msg_.startRingIndex[i] = seg_cloud_size - 1 + 5; // (start-1) + 5 ?????????????????????5??????????????????????????????

        for(size_t j = 0; j < lidar_param_.Horizon_SCAN; ++j){

            if(ground_mat_.at<int8_t>(i,j) != 1 && 
                full_info_cloud_->points[j + i*lidar_param_.Horizon_SCAN].intensity < sensor_segment_range_)
                    continue;

            if(label_mat_.at<int>(i,j) > 0 || ground_mat_.at<int8_t>(i,j) == 1){
                // outliers that will not be used for optimization (always continue)
                // j%5 ???????????????????????????????????????????????????????????????????????????????????????
                if (label_mat_.at<int>(i,j) == 999999){
                    if (i > lidar_param_.ground_scan_id_ && j % 5 == 0){
                        outlier_cloud_->push_back(full_cloud_->points[j + i*lidar_param_.Horizon_SCAN]);
                        continue;
                    }else{
                        continue;
                    }
                }
                // majority of ground points are skipped
                // ????????????????????????????????????ground?????????segmented points
                if (ground_mat_.at<int8_t>(i,j) == 1){
                    if (j%5!=0 && j>5 && j<lidar_param_.Horizon_SCAN-5)
                        continue;
                }
                
                // mark ground points so they will not be considered as edge features later
                seg_msg_.segmentedCloudGroundFlag[seg_cloud_size] = (ground_mat_.at<int8_t>(i,j) == 1);
                // mark the points' column index for marking occlusion later
                seg_msg_.segmentedCloudColInd[seg_cloud_size] = j;
                // save range info
                seg_msg_.segmentedCloudRange[seg_cloud_size]  = range_mat_.at<float>(i,j);
                // save seg cloud
                segmented_cloud_->push_back(full_cloud_->points[j + i*lidar_param_.Horizon_SCAN]);
                // size of seg cloud
                ++seg_cloud_size;
            }
        }

        seg_msg_.endRingIndex[i] = seg_cloud_size - 1 - 5;  // (end -1) -5 ?????????????????????5??????????????????????????????
    }
    // extract segmented cloud for visualization
    if (make_pure_cloud_flag_){
        for (size_t i = 0; i < lidar_param_.N_SCAN; ++i){
            for (size_t j = 0; j < lidar_param_.Horizon_SCAN; ++j){
                if (label_mat_.at<int>(i,j) > 0 && label_mat_.at<int>(i,j) != 999999){
                    segmented_cloud_pure_->push_back(full_cloud_->points[j + i*lidar_param_.Horizon_SCAN]);
                    segmented_cloud_pure_->points.back().intensity = label_mat_.at<int>(i,j);
                }
            }
        }
    }
}


// ???????????????????????????????????????????????????
void ImageProjection::labelComponents(int row, int col){
    // use std::queue std::vector std::deque will slow the program down greatly
    float d1, d2, alpha, angle;
    int from_ind_x, from_ind_y, this_ind_x, this_ind_y;
    bool line_count_flag[lidar_param_.N_SCAN] = {false};

    queue_ind_x_[0] = row;
    queue_ind_y_[0] = col;
    int queue_size = 1;
    int queue_start_ind = 0;
    int queue_end_ind = 1;

    all_pushed_ind_x_[0] = row;
    all_pushed_ind_y_[0] = col;
    int all_pushed_ind_size = 1;

    while(queue_size > 0){
        // Pop point
        from_ind_x = queue_ind_x_[queue_start_ind];
        from_ind_y = queue_ind_y_[queue_start_ind];
        --queue_size;
        ++queue_start_ind;
        // Mark popped point
        // ??????????????????????????? ???label_count_???
        label_mat_.at<int>(from_ind_x, from_ind_y) = label_count_;
        // Loop through all the neighboring grids of popped grid
        // ????????????????????????????????????????????????????????????
        for(auto iter = neighbor_iterator_.begin(); iter != neighbor_iterator_.end(); ++ iter){
            // new index
            this_ind_x = from_ind_x + (*iter).first;
            this_ind_y = from_ind_y + (*iter).second;
            // index should be within the boundary
            if (this_ind_x < 0 || this_ind_x >= lidar_param_.N_SCAN)
                continue;
            // at range image margin (left or right side)
            if (this_ind_y < 0)
                this_ind_y = lidar_param_.Horizon_SCAN - 1;
            if (this_ind_y >= lidar_param_.Horizon_SCAN)
                this_ind_y = 0;
            // prevent infinite loop (caused by put already examined point back)
            // ?????????????????????????????????????????????????????????????????????
            if (label_mat_.at<int>(this_ind_x, this_ind_y) != 0)
                continue;
            // from_ind???????????????????????????this_ind?????????????????????
            // d1?????????????????????????????????d2???????????????????????????
            d1 = std::max(range_mat_.at<float>(from_ind_x, from_ind_y), 
                            range_mat_.at<float>(this_ind_x, this_ind_y));
            d2 = std::min(range_mat_.at<float>(from_ind_x, from_ind_y), 
                            range_mat_.at<float>(this_ind_x, this_ind_y));
            
            if ((*iter).first == 0) // ??????????????????scan????????????x??1????????????segment_alpha_x_
                alpha = segment_alpha_x_;
            else    // ?????????????????????scan????????????y??1????????????segment_alpha_y_
                alpha = segment_alpha_y_;
            
            //???????????????????????????????????????????????????
            angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));
            // TODO: ??????1??????????????????
            // ????????????????????????????????????????????????// XXX :??????????????????????????????????????????????????????????????????????????????????????????
            if (angle > segment_theta_){

                queue_ind_x_[queue_end_ind] = this_ind_x;
                queue_ind_y_[queue_end_ind] = this_ind_y;
                ++queue_size;
                ++queue_end_ind;

                label_mat_.at<int>(this_ind_x, this_ind_y) = label_count_;
                line_count_flag[this_ind_x] = true;

                all_pushed_ind_x_[all_pushed_ind_size] = this_ind_x;
                all_pushed_ind_y_[all_pushed_ind_size] = this_ind_y;
                ++all_pushed_ind_size;  //??????????????????1
            }
        }
    }

     // check if this segment is valid
    bool feasible_segment = false;
    // ??????30????????????????????????????????????????????????????????????
    if (all_pushed_ind_size >= 30)
        feasible_segment = true;
    // ???????????????5???????????????????????????????????????????????????????????????????????????
    else if (all_pushed_ind_size >= segment_valid_point_num_){
        int line_count = 0;
        for (size_t i = 0; i < lidar_param_.N_SCAN; ++i)
            if (line_count_flag[i] == true)
                ++line_count;
        if (line_count >= segment_valid_line_num_)
            feasible_segment = true;            
    }
    // segment is valid, mark these points
    // ?????????????????????????????????
    if (feasible_segment == true){
        ++label_count_;
    // ????????????????????????????????????????????????
    }else{ // segment is invalid, mark these points
        for (size_t i = 0; i < all_pushed_ind_size; ++i){
            label_mat_.at<int>(all_pushed_ind_x_[i], all_pushed_ind_y_[i]) = 999999;
        }
    }
}

pcl::PointCloud<PointType>::Ptr ImageProjection::getFullCloud(){
    return full_cloud_;
}

pcl::PointCloud<PointType>::Ptr ImageProjection::getFullInfoCloud(){
    return full_info_cloud_;
}

pcl::PointCloud<PointType>::Ptr ImageProjection::getGroundCloud(){
    return ground_cloud_;
}

pcl::PointCloud<PointType>::Ptr ImageProjection::getGroundRemovedCloud(){
    return ground_removed_cloud_;
}

pcl::PointCloud<PointType>::Ptr ImageProjection::getSegmentedCloud(){
    return segmented_cloud_;
}

pcl::PointCloud<PointType>::Ptr ImageProjection::getSegmentedCloudPure(){
    return segmented_cloud_pure_;
}

pcl::PointCloud<PointType>::Ptr ImageProjection::getOutlierCloud(){
    return outlier_cloud_;
}

cloud_msgs::cloud_info ImageProjection::getSegmentedCloudInfo(){
    return seg_msg_;
}


} // end of namespace
