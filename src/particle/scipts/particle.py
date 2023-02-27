#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import os
import sys
import yaml
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import rospy
import open3d as o3d
from map_module import MapModule
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from map_renderer import MapRenderer, MapRenderer_instanced
from sensor_msgs import point_cloud2
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseArray, Pose, Twist
import copy
import time
import os
class Particle:
    def __init__(self):
        self.initPose = False
        rospy.init_node('particle', anonymous=True)
        rospy.Subscriber("/rslidar_points", PointCloud2, self.laserCloudInfoHandler)
        rospy.Subscriber("/initial_odom", Odometry, self.initOdomHandler)
        self.gazebo_environment = False
        if self.gazebo_environment:
            rospy.Subscriber("/odom", Odometry, self.getWheelHandler)
        else:
            rospy.Subscriber("/wheel/data", Twist, self.getWheelHandler)
        self.pub_particle_odom = rospy.Publisher("/particle_odom", Odometry, queue_size=10)
        self.pub_particles = rospy.Publisher("/particles", PoseArray, queue_size=10)
        # self.x0 = 12
        # self.y0 = 9
        self.current_time = rospy.get_rostime().to_sec()
        self.last_time = rospy.get_rostime().to_sec()
        self.vx = 0
        self.vy = 0
        self.w = 0
        

        # 获取当前工作目录
        current_path = os.path.abspath(__file__)
        current_path = os.path.dirname(current_path)
        current_path = os.path.dirname(current_path)
        current_path = os.path.dirname(current_path)
        self.map_file = current_path+'/lio_sam/data_park/park.ply'
        self.pose_file = current_path+'/lio_sam/data_park/pose_kitti.txt'
        self.height = 64
        self.width = 900
        if self.gazebo_environment:
            self.fov_up = 15
            self.fov_down = -15
            self.max_range = 131
            self.min_range = 0.3
        else:#雷达参数
            self.fov_up = 15
            self.fov_down = -15
            self.max_range = 150
            self.min_range = 0.4
        self.is_converged = False
        self.mapInit = True
        self.poses = self.load_poses(self.pose_file)
        print(self.poses.shape)
        self.start = time.time()
        self.timeInit = True
        # visualize = Visualize(self.particles, self.poses)
    def laserCloudInfoHandler(self, msgIn):
        if self.initPose == False:
            return
        if self.mapInit:
            self.mapInit = False
            self.map_module = MapModule(self.poses, self.map_file)
            self.renderer = MapRenderer(self.fov_up, self.fov_down, self.max_range, self.min_range)
            self.renderer.set_mesh(self.map_module.mesh)
            self.particles = self.initParticle()
        self.particles = self.motion_model(self.particles)
        cloud_msg = PoseArray()
        cloud_msg.header.frame_id = "map"
        cloud_msg.header.stamp = rospy.get_rostime()
        p = Pose()
        for i in range(len(self.particles)):
            p.position.x = self.particles[i][0]
            p.position.y = self.particles[i][1]
            q = R.from_euler('z', self.particles[i][2], degrees=False).as_quat()
            q= q.ravel()
            p.orientation.x = q[0]
            p.orientation.y = q[1]
            p.orientation.z = q[2]
            p.orientation.w = q[3]
            cloud_msg.poses.append(copy.deepcopy(p))
        self.pub_particles.publish(cloud_msg)
        # if abs(self.vx) > 0.1 or abs(self.vy) > 0.1 or abs(self.w) > 0.1:
        if 1:
            # print("update weight and resample:")
            self.particles = self.update_weights(self.particles, msgIn)
            self.particles = self.resample(self.particles)
        self.num = 0
        for i in self.particles[:,3]:
            if(i<0.9 and i>0.3):
                self.num+=1
        if self.timeInit==True:
            print(self.num)
            print(self.particles[:,3])
        if self.num == 0 and self.timeInit==True:
            self.timeInit=False
            self.end = time.time()
            print("time:", self.end-self.start, "s")
            x_sum = 0
            y_sum = 0
            yaw_sum = 0
            n = 0
            for particle in self.particles:
                if particle[3] > 0.9:
                    x_sum += particle[0] * particle[3]
                    y_sum += particle[1] * particle[3]
                    yaw_sum += particle[2] * particle[3]
                    n += 1
            msg = Odometry()
            msg.header.frame_id = "map"
            msg.header.stamp = rospy.get_rostime()
            msg.pose.pose.position.x = x_sum / n
            msg.pose.pose.position.y = y_sum / n
            q = R.from_euler('z', yaw_sum / n, degrees=False).as_quat()
            q= q.ravel()
            msg.pose.pose.orientation.x = q[0]
            msg.pose.pose.orientation.y = q[1]
            msg.pose.pose.orientation.z = q[2]
            msg.pose.pose.orientation.w = q[3]
            self.pub_particle_odom.publish(msg)
            print("x:",msg.pose.pose.position.x)
            print("y:", msg.pose.pose.position.y)
            print("yaw:", yaw_sum / n + 6.28)
    def initOdomHandler(self, msgIn):
        if self.initPose == False:
            self.initPose = True
            self.x0 = msgIn.pose.pose.position.x
            self.y0 = msgIn.pose.pose.position.y
            # x = msgIn.pose.pose.orientation.x
            # y = msgIn.pose.pose.orientation.y
            # z = msgIn.pose.pose.orientation.z
            # w = msgIn.pose.pose.orientation.w
            # quat = [x,y,z,w]
            # #不知道怎么将四元数归一化
            # self.yaw0 = R.from_quat(quat).as_euler('zyx',degrees=False) 
            # print("self.yaw0:", self.yaw0)
    def getWheelHandler(self, msgIn):
        if self.gazebo_environment:
            self.vx = math.sqrt(msgIn.twist.twist.linear.x ** 2 + msgIn.twist.twist.linear.y ** 2)
            self.vy = 0
            self.w = msgIn.twist.twist.angular.z
        else:
            self.vx = msgIn.linear.x
            self.vy = 0
            self.w = msgIn.angular.z
    def load_poses(self, pose_path):
        poses = []
        with open(pose_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                T = np.fromstring(line, dtype=float, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return np.array(poses)
    def initParticle(self, resolution=0.1, submap_size=4):
        coords = []
        for x_coord in np.arange(-submap_size, submap_size, resolution):
            for y_coord in np.arange(-submap_size, submap_size, resolution):
                rand = np.random.rand
                theta = -np.pi + 2 * np.pi * rand(1)
                coords.append([x_coord+self.x0, y_coord+self.y0, theta, 1])
        coords = np.array(coords)
        return coords
    def motion_model(self, particles):
        self.current_time = rospy.get_rostime().to_sec()
        dt = self.current_time - self.last_time
        self.last_time = self.current_time
        for particle in particles:
            delta_x = (self.vx*math.cos(particle[2]) - self.vy*math.sin(particle[2])) * dt
            delta_y = (self.vx*math.sin(particle[2]) + self.vy*math.cos(particle[2])) * dt
            delta_th = self.w * dt
            particle[0] += delta_x
            particle[1] += delta_y
            particle[2] += delta_th
        return particles
    def update_weights(self, particles, msgIn):
        assert isinstance(msgIn, PointCloud2)
        gen = point_cloud2.read_points(msgIn, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        points = []
        for p in gen:
            points.append(p)
        points = np.array(points)
        current_range = self.range_projection(points, self.fov_up, self.fov_down, self.height, self.width, self.max_range)
        scores = np.ones(len(particles)) * 0.00001
        tiles_collection = []
        for idx in range(len(particles)):
            particle = particles[idx]
            if particle[0] < self.map_module.map_boundaries[0] or \
                particle[0] > self.map_module.map_boundaries[1] or \
                particle[1] < self.map_module.map_boundaries[2] or \
                particle[1] > self.map_module.map_boundaries[3]:
                continue
            tile_idx = self.map_module.get_tile_idx([particle[0], particle[1]])
            if not self.map_module.tiles[tile_idx].valid:
                continue
            if tile_idx not in tiles_collection:
                tiles_collection.append(tile_idx)
            start = self.map_module.tiles[tile_idx].vertices_buffer_start
            size = self.map_module.tiles[tile_idx].vertices_buffer_size
            particle_pose = np.identity(4)
            particle_pose[0, 3] = particle[0]
            particle_pose[1, 3] = particle[1]
            particle_pose[2, 3] = self.map_module.tiles[tile_idx].z
            particle_pose[:3, :3] = R.from_euler('z', particle[2], degrees=False).as_matrix()
            self.renderer.render_with_tile(particle_pose, start, size)
            particle_depth = self.renderer.get_depth_map()
            diff = abs(particle_depth - current_range)
            # print(np.mean(diff[current_range > 0]))
            scores[idx] = np.exp(-0.5 * np.mean(diff[current_range > 0]) ** 2 / (2.0 ** 2))
        particles[:, 3] = particles[:, 3] * scores
        particles[:, 3] = particles[:, 3] / np.max(particles[:, 3])
        # if len(tiles_collection) < 2 and not self.is_converged:
        if 1:
            self.is_converged = True
            # print('Converged!')
            idxes = np.argsort(particles[:, 3])[::-1]
            particles = particles[idxes[:100]]
        return particles

    def range_projection(self, current_vertex, fov_up, fov_down, proj_H, proj_W, max_range):
        """ Project a pointcloud into a spherical projection, range image.
            Args:
            current_vertex: raw point clouds
            Returns:
            proj_range: projected range image with depth, each pixel contains the corresponding depth
            proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
            proj_intensity: each pixel contains the corresponding intensity
            proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
        """
        # laser parameters
        fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
        fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians
        
        # get depth of all points
        depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
        current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
        depth = depth[(depth > 0) & (depth < max_range)]
        
        # get scan components
        scan_x = current_vertex[:, 0]
        scan_y = current_vertex[:, 1]
        scan_z = current_vertex[:, 2]
        intensity = current_vertex[:, 3]
        # print(intensity[:20])
        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)
        
        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
        
        # scale to image size using angular resolution
        proj_x *= proj_W  # in [0.0, W]
        proj_y *= proj_H  # in [0.0, H]
        
        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        
        proj_y = np.floor(proj_y)
        proj_y = np.minimum(proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        
        # order in decreasing depth
        order = np.argsort(depth)[::-1]#将升序索引变为降序索引
        depth = depth[order]
        intensity = intensity[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]
        
        scan_x = scan_x[order]
        scan_y = scan_y[order]
        scan_z = scan_z[order]
        
        indices = np.arange(depth.shape[0])
        indices = indices[order]
        
        proj_range = np.full((proj_H, proj_W), -1,
                            dtype=np.float32)  # [H,W] range (-1 is no data)
        proj_vertex = np.full((proj_H, proj_W, 4), -1,
                                dtype=np.float32)  # [H,W] index (-1 is no data)
        proj_idx = np.full((proj_H, proj_W), -1,
                            dtype=np.int32)  # [H,W] index (-1 is no data)
        proj_intensity = np.full((proj_H, proj_W), -1,
                                dtype=np.float32)  # [H,W] index (-1 is no data)
        
        proj_range[proj_y, proj_x] = depth
        proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
        proj_idx[proj_y, proj_x] = indices
        proj_intensity[proj_y, proj_x] = intensity
    
        return proj_range
    def resample(self, particles):
        weights = particles[:, 3]
        weights = weights / sum(weights)
        eff_N = 1 / sum(weights ** 2)
        new_particles = np.zeros(particles.shape)
        i = 0
        if eff_N < len(particles)*3.0/4.0:#粒子的权重和太小，开始更新粒子
            r = np.random.rand(1) * 1.0/len(particles)
            c = weights[0]
            for idx in range(len(particles)):
                u = r + idx/len(particles)
                while u > c:
                    if i >= len(particles) - 1:
                        break
                    i += 1
                    c += weights[i]
                new_particles[idx] = particles[i]#对每个粒子遍历。每次找个随机数u，如果u的取值落在i的权重到i+1的权重之间，说明i的权重比较大，就将i对应的粒子赋值给u
        else:
            new_particles = particles
        return new_particles
class Visualize:
    def __init__(self, particles, poses):
        min_x = int(np.round(np.min(poses[:, 0, 3])))-10
        max_x = int(np.round(np.max(poses[:, 0, 3])))+10
        min_y = int(np.round(np.min(poses[:, 1, 3])))-10
        max_y = int(np.round(np.max(poses[:, 1, 3])))+10
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = np.array([min_x, max_x])
        y = np.array([min_y, max_y])
        print(x, y)
        ax.set(xlim=x, ylim=y)
        currentAxis = plt.gca()
        plt.plot(poses[:, 0, 3], poses[:, 1, 3])
        plt.plot(particles[:, 0], particles[:, 1])
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.show()
if __name__ == '__main__':
    particle = Particle()

    rospy.spin()
