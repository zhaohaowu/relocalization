首先git clone整个工作空间

```
cd
git clone https://github.com/zhaohaowu/relocalization.git
```
安装第三方库
```
#安装python需要的库
cd ~/relocalization
sudo apt-get update 
sudo apt-get install libgl1-mesa-glx
sudo apt-get install libusb-1.0   # open3d 0.12.0 dependency
sudo apt-get install -y python3-pip
pip3 install --upgrade pip
pip3 install -r 3rdparty/requirements.txt #安装第三方库
#安装protobuf
cd 3rdparty/protobuf-3.14.0
./autogen.sh
./configure
make
sudo make install
sudo ldconfig
protoc --version
#安装g2o
cd ../g2o
mkdir build
cd build
cmake ..
make
sudo make install
#安装GeographicLib
cd ../../GeographicLib-1.48
mkdir build
cd build
cmake ..
make
sudo make install
```
编译程序
```
cd ~/relocalization
catkin_make
source devel/setup.bash
```

bag下载地址

链接：https://pan.baidu.com/s/1DE8PUkGEX55W_lFk68A1Vg?pwd=slam 
提取码：slam

生成特征地图，关键帧bin和pcd，关键帧位姿，生成Scans, SCDs, flat_cloud_map.pcd, sharp_cloud_map.pcd, pose.txt, pose_kitti.txt保存至lio_sam的data文件夹下
```
cd ~/relocalization
source devel/setup.bash
roslaunch lio_sam run.launch
rosbag play park.bag
```

pcd转为bin，生成bin保存至lio_sam的data文件夹下

```
cd ~/relocalization
source devel/setup.bash
rosrun lio_sam pcd2bin
```

然后将data复制到data_park数据集下

```
cd ~/relocalization/src/lio_sam
cp -r data data_park
```

建mesh地图

```
cd ~/relocalization
source devel/setup.bash
cd ~/relocalization/src/particle/scipts
python3 build_mesh_map.py
```

生成初始initial_odom

```
cd ~/relocalization
source devel/setup.bash
rosrun lio_sam sc_pose
```

生成particle_odom

```
cd ~/relocalization
source devel/setup.bash
roslaunch particle particle.launch
```

基于特征的重定位

```
cd ~/relocalization
source devel/setup.bash
roslaunch lidar_localization matching_loam.launch
```

