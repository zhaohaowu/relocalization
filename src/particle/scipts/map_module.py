#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Brief: this is the map module for mesh-based Monte Carlo localization.

import numpy as np
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from map_renderer import Mesh, OffscreenWindow

class MapModule:
  """ Mesh map module.
  We use a triangular mesh as a map of the environment.
  A triangular mesh provides us with a compact representation
  that enables us to render the synthetic range images for particles.
  In the map module, we split the global mesh-based map into tiles to accelerate
  the Monte Carlo localization by more efficient rendering.
  """
  def __init__(self, map_poses, mesh_file, max_distance=50, keep_tile_maps=False):
    """ Constructor input:
     map_poses: the ground truth to build the mesh map
     mesh_file: path to the prebuild mesh map the command in the form [rot1 trasl rot2] or real odometry [v, w]
     max_distance: maximum distance for range image rendering
     keep_tile_maps: if false, we don't keep tile maps in CPU while keep only the start vertex index and the size
    """
    # even though we are not requiring the window, we still have to create on with glfw to get an context.
    # however, that should not be so relevant.
    window = OffscreenWindow(show=False)
    
    # several properties
    self.offset_x = 0
    self.offset_y = 0
    self.numTiles_x = 0
    self.numTiles_y = 0
    self.tile_size = 100/5#片状地图大小
    self.max_distance = max_distance/5#距离图像渲染的最大距离
    self.keep_tile_maps = keep_tile_maps#内存是否保留片状地图，如果否，只保存开始顶点索引和数量
    self.map_boundaries = [0, 0, 0, 0]  # [x_min, x_max, y_min, y_max]#mesh地图边界，最大的点云位姿加上渲染最大距离
    
    # load meshes  # we don't need to maintain a global mesh anymore
    o3d_mesh = o3d.io.read_triangle_mesh(mesh_file)
    o3d_mesh.compute_vertex_normals()
    
    vertices = np.asarray(o3d_mesh.vertices, dtype=np.float32)#表示所有点的三维坐标
    normals = np.asarray(o3d_mesh.vertex_normals, dtype=np.float32)#表示所有点的向量
    # print(normals.shape)
    # print(vertices.shape)
    triangles = np.asarray(o3d_mesh.triangles, dtype=np.int32)#表示n个三角形的三个点的索引
    
    rearranged_vertices = vertices[triangles]#表示n个三角形的三个点的三维坐标
    rearranged_normals = normals[triangles]#表示n个三角形的三个点的向量
    # print(rearranged_normals[:2])
    # load poses
    self.poses = map_poses
    
    # initialize tiles
    self.tiles = self.tile_init(self.poses, tile_size=self.tile_size, max_distance=self.max_distance)#根据真值二维坐标将绘制片状地图，每个片状地图大小为100乘100，每个片状地图存储对应的激光帧索引
    
    # calculate z for the tile maps
    self.calculate_tile_height()#计算片状地图的对应真值位姿的z平均值
    
    # instead of saving submaps
    self.generate_tile_map(o3d_mesh, max_distance=self.max_distance)#将原始mesh地图按照每个片状地图中心进行裁剪
    
    # we now store vertices in each tile map
    self.mesh = self.generate_buffer_for_all_vertices()#mesh存储每个三角形的三维坐标和向量
    
    # clean the tile maps
    if not keep_tile_maps:
      self.clean_tile_maps()
  
  class Tile:
    """ Class of tile map. """
    def __init__(self, i, j, x, y):
      self.i = i  # [i, j] tile coordinates
      self.j = j
      self.x = x  # [x, y] actual world coordinates.
      self.y = y
      self.z = 0  # default as zero, will be updated after calling calculate_tile_height()
      
      self.valid = False  # if one tile contains at least one scan, it's valid
      self.scan_indexes = []  # scan indexes
      self.neighbor_indexes = []  # neighbor tile indexes
      self.tile_map = o3d.geometry.TriangleMesh()  # corresponding submap mesh
      self.vertices = []  # vertices of triangles
      self.normals = []  # normals of triangles
      self.particle_indexes = []  # indexes of particles locate in this tile
      self.vertices_buffer_start = 0  # start point of the tile map in the vertices buffer
      self.vertices_buffer_size = 0  # size of vertices of this tile map
  
  def tile_init(self, poses, tile_size=100, max_distance=50, plot_tiles=False):
    """ Initialize tile maps. """
    # get boundary of poses
    bound_x_min = min(poses[:, 0, 3]) - max_distance
    bound_y_min = min(poses[:, 1, 3]) - max_distance
    bound_x_max = max(poses[:, 0, 3]) + max_distance
    bound_y_max = max(poses[:, 1, 3]) + max_distance
    
    self.map_boundaries = [bound_x_min, bound_x_max, bound_y_min, bound_y_max]#mesh地图边界，最大的点云位姿加上渲染最大距离
    
    print("lower bound:", [bound_x_min, bound_y_min], "upper bound:", [bound_x_max, bound_y_max])
    
    offset_x = np.ceil((abs(bound_x_min) - 0.5 * tile_size) / tile_size) * tile_size + 0.5 * tile_size#x偏移量，150
    offset_y = np.ceil((abs(bound_y_min) - 0.5 * tile_size) / tile_size) * tile_size + 0.5 * tile_size#y偏移量，150
    
    numTiles_x = int(np.ceil((abs(bound_x_min) - 0.5 * tile_size) / tile_size) + \
                     np.ceil((bound_x_max - 0.5 * tile_size) / tile_size) + 1)#x方向片状地图个数
    numTiles_y = int(np.ceil((abs(bound_y_min) - 0.5 * tile_size) / tile_size) + \
                     np.ceil((bound_y_max - 0.5 * tile_size) / tile_size) + 1)#y方向片状地图y个数
    # print(offset_x)
    # print(offset_y)
    # print(numTiles_x)
    # print(numTiles_y)
    tiles = {}
    for idx_x in range(numTiles_x):
      # print(idx_x)
      for idx_y in range(numTiles_y):
        idx_tile = idx_x + idx_y * numTiles_x
        tiles[idx_tile] = self.Tile(idx_x, idx_y,
                                    idx_x * tile_size - offset_x + 0.5 * tile_size,
                                    idx_y * tile_size - offset_y + 0.5 * tile_size)#16个片状地图的生成(-100,-100)到(200,200)
        # print(idx_x * tile_size - offset_x + 0.5 * tile_size)
        # print(idx_y * tile_size - offset_y + 0.5 * tile_size)
    # check which poses are included by which tile
    e = [0.5 * tile_size, 0.5 * tile_size]
    # print(len(poses))
    # print(len(tiles))
    for idx_scan in range(len(poses)):
      for idx_tile in range(len(tiles)):
        q = abs(poses[idx_scan, :2, 3] - [tiles[idx_tile].x, tiles[idx_tile].y])
        
        # if max(q) > e[0] + max_distance: continue  # definitely outside.
        # if min(q) < e[0] or np.linalg.norm(q - e) < max_distance:
        if np.linalg.norm(q) < max_distance + 0.5 * e[0]:
          # print(idx_scan)
          # print(idx_tile)
          # print(poses[idx_scan, :2, 3])
          # print([tiles[idx_tile].x, tiles[idx_tile].y])
          tiles[idx_tile].scan_indexes.append(idx_scan)#把片状地图中心坐标与真值轨迹距离小的对应激光点云索引push进片状地图中
    # print(len(tiles))
    # print(len(poses))
    # print(q)
    # print(np.linalg.norm(q))
    # check validity of tiles
    tile_counter = 0
    for idx_tile in range(len(tiles)):
      # print(idx_tile)
      # print([tiles[idx_tile].x, tiles[idx_tile].y])
      if len(tiles[idx_tile].scan_indexes) > 1:
        # print(idx_tile)
        tiles[idx_tile].valid = True#这些索引下的片状地图有对应的激光点云
        tile_counter += 1
    
    print("number of tiles = ", tile_counter)
    
    self.offset_x = offset_x
    self.offset_y = offset_y#偏移量
    self.numTiles_x = numTiles_x
    self.numTiles_y = numTiles_y#片状地图个数共16个

    if plot_tiles:
      self.plot_valid_tiles(tiles, poses)#把真值轨迹附近的片状地图绘制出来
    
    return tiles

  def crop_mesh_with_bbox(self, mesh, position, length=50, width=50, z_level=5, offset=1):
    """ Crop the mesh. """
    bbox = o3d.geometry.AxisAlignedBoundingBox(
      min_bound=(-length + position[0] - offset, -width + position[1] - offset, -z_level),
      max_bound=(+length + position[0] + offset, +width + position[1] + offset, +z_level))
    return mesh.crop(bbox)

  def generate_tile_map(self, mesh, max_distance=50, extended_border=50):
    """ Generate submap mesh for tile map. """
    for idx in range(len(self.tiles)):
      if self.tiles[idx].valid:
        self.tiles[idx].tile_map = self.crop_mesh_with_bbox(mesh,
                                                            [self.tiles[idx].x, self.tiles[idx].y],
                                                            max_distance + extended_border,
                                                            max_distance + extended_border)#以每个片状地图中心的-100和100范围对原始mesh进行裁剪

  def generate_buffer_for_all_vertices(self):
    """ generate a buffer for all vertices of the global mesh. """
    # get the total number of vertices we stored
    num_triangles = self.get_num_triangles()
    print("total number of triangles: ", num_triangles)
  
    # rearrange the vertices and assign them to the vertex buffer
    rearranged_vertices_buffer = np.empty(num_triangles * 9, dtype=np.float32)
    rearranged_normals_buffer = np.empty(num_triangles * 9, dtype=np.float32)
    counter = 0
  
    for tile_idx in range(len(self.tiles)):
      tile_mesh = self.tiles[tile_idx].tile_map
      vertices = np.asarray(tile_mesh.vertices, dtype=np.float32)#每个片状地图的n个顶点坐标
      normals = np.asarray(tile_mesh.vertex_normals, dtype=np.float32)#每个片状地图的n个法向量
      triangles = np.asarray(tile_mesh.triangles, dtype=np.int32)#每个片状地图的n个三角形的三个点的索引
      # print(vertices.shape)
      # print(normals.shape)
      # print(triangles.shape)
      rearranged_vertices = vertices[triangles]#n*3*3，n个三角形的三个点的三维坐标
      rearranged_normals = normals[triangles]#n*3*3，n个三角形的三个点的向量
    
      num_triangles_tile = len(rearranged_vertices)
      # print(rearranged_vertices.shape)
      rearranged_vertices_buffer[counter * 9:(counter + num_triangles_tile) * 9] = rearranged_vertices.reshape(-1)#将每个片状地图的n个三角形的三个点的三维坐标摊平为一维向量存储到buffer
      rearranged_normals_buffer[counter * 9:(counter + num_triangles_tile) * 9] = rearranged_normals.reshape(-1)#将每个片状地图的n个三角形的三个点的向量摊平为一维向量存储到buffer
      # print(rearranged_vertices_buffer.shape)
      # assign start point and vertices size of the tile map
      self.tiles[tile_idx].vertices_buffer_start = counter * 9#每个片状地图的开始顶点索引
      self.tiles[tile_idx].vertices_buffer_size = num_triangles_tile * 9#每个片状地图存储n个三角形的顶点的三维坐标的个数
      counter += num_triangles_tile
    
      # clean the tile maps
      if not self.keep_tile_maps:
        self.tiles[tile_idx].tile_map = None#清除每个索引下的片状地图
    
    mesh = Mesh()#mesh存储每个三角形的三维坐标和向量
    mesh._buf_vertices.assign(rearranged_vertices_buffer)
    mesh._buf_normals.assign(rearranged_normals_buffer)
  
    return mesh

  def generate_tile_map_vertex(self, rearranged_vertices, rearranged_normals, max_distance=50):
    """ Old version, we save submap vertices"""
    for idx in tqdm(range(len(rearranged_vertices))):
      # get tile index of each vertex of the triangle
      tile_idx_A = self.get_tile_idx([rearranged_vertices[idx, 0, 0], rearranged_vertices[idx, 0, 1]])  # vertex A(x, y)
      tile_idx_B = self.get_tile_idx([rearranged_vertices[idx, 1, 0], rearranged_vertices[idx, 1, 1]])  # vertex B(x, y)
      tile_idx_C = self.get_tile_idx([rearranged_vertices[idx, 2, 0], rearranged_vertices[idx, 2, 1]])  # vertex C(x, y)
    
      # add vertices to the corresponding tile map
      self.tiles[tile_idx_A].vertices.append(rearranged_vertices[idx])
      self.tiles[tile_idx_A].normals.append(rearranged_normals[idx])
    
      # for the same triangle we store only once in one tile map
      if tile_idx_B != tile_idx_A:
        self.tiles[tile_idx_B].vertices.append(rearranged_vertices[idx])
        self.tiles[tile_idx_B].normals.append(rearranged_normals[idx])
    
      if tile_idx_C != tile_idx_A and tile_idx_C != tile_idx_B:
        self.tiles[tile_idx_C].vertices.append(rearranged_vertices[idx])
        self.tiles[tile_idx_C].normals.append(rearranged_normals[idx])
  
    # get the total number of vertices we stored
    num_triangles = self.get_num_triangles()
    print("total number of triangles: ", num_triangles)
  
    # rearrange the vertices and assign them to the vertex buffer
    rearranged_vertices_buffer = np.empty(num_triangles * 9, dtype=np.float32)
    rearranged_normals_buffer = np.empty(num_triangles * 9, dtype=np.float32)
    counter = 0
    for tile_idx in range(len(self.tiles)):
      num_triangles_tile = len(self.tiles[tile_idx].vertices)
      rearranged_vertices_buffer[counter * 9:(counter + num_triangles_tile) * 9] = np.array(
        self.tiles[tile_idx].vertices).reshape(-1)
      rearranged_normals_buffer[counter * 9:(counter + num_triangles_tile) * 9] = np.array(
        self.tiles[tile_idx].normals).reshape(-1)
      counter += num_triangles_tile
  
    return rearranged_vertices_buffer, rearranged_normals_buffer

  def get_local_map(self, tile_idx):
    """ Get the tile map sub-mesh. """
    return self.tiles[tile_idx].tile_map

  def get_global_map(self):
    """ Get the global mesh map. """
    global_map = o3d.geometry.TriangleMesh()
    for tile_idx in range(len(self.tiles)):
      global_map += self.tiles[tile_idx].tile_map
  
    return global_map

  def get_particles(self, tile_idx):
    """ Get the indexes of particles of give tile. """
    return self.tiles[tile_idx].particle_indexes

  def get_tile_idx(self, position):
    """ Get the index of a tile of give position. """
    # world coordinates to tile index
    i = round((position[0] + self.offset_x - 0.5 * self.tile_size) / self.tile_size)
    j = round((position[1] + self.offset_y - 0.5 * self.tile_size) / self.tile_size)
    tile_idx = int(round(i + j * self.numTiles_x))
    #i和j分别表示x方向tile索引和y方向tile索引
    return tile_idx

  def get_num_triangles(self, use_tile_map=True):
    """ Get the number of triangles. """
    num_triangles = 0
  
    if use_tile_map:
      # print('use cropped tile map to rearrange vertices buffer.')
      for tile_idx in range(len(self.tiles)):
        tile_mesh = self.tiles[tile_idx].tile_map
        vertices = np.asarray(tile_mesh.vertices, dtype=np.float32)
        triangles = np.asarray(tile_mesh.triangles, dtype=np.int32)
      
        rearranged_vertices = vertices[triangles]
        num_triangles += len(rearranged_vertices)
  
    else:
      # print('use vertices directly.')
      for tile_idx in range(len(self.tiles)):
        num_triangles += len(self.tiles[tile_idx].vertices)
        print(len(self.tiles[tile_idx].vertices))
  
    return num_triangles

  def clean_tile_maps(self):
    """ Release tile maps in CPU. """
    for tile_idx in range(len(self.tiles)):
      self.tiles[tile_idx].tile_map = None

  def calculate_tile_height(self):
    """ Calculate the height for each tile. """
    for tile_idx in range(len(self.tiles)):
      # print(tile_idx)
      if len(self.tiles[tile_idx].scan_indexes) == 0: continue
      poses = self.poses[self.tiles[tile_idx].scan_indexes]
      self.tiles[tile_idx].z = np.mean(poses[:, 2, 3])#每个有效片状地图的真值坐标的平均值
      # print(self.tiles[tile_idx].z)
    
  def plot_valid_tiles(self, tiles, poses):
    """ Visualize supmaps together with trajectory. """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    currentAxis = plt.gca()
    plt.plot(poses[:, 0, 3], poses[:, 1, 3])
    for idx in range(len(tiles)):
      if tiles[idx].valid:
        currentAxis.add_patch(Rectangle((tiles[idx].x - self.max_distance, tiles[idx].y - self.max_distance),
                                        self.tile_size, self.tile_size, alpha=1, fill=None))
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Visualization of trajectory and submaps")
    plt.show()

  def vis_mesh(self, mesh, crop_mesh=False):
    """ Visualize mesh. """
    if crop_mesh:
      bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-50, -50, -5),
                                                 max_bound=(+50, +50, +5))
      mesh = mesh.crop(bbox)
  
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

  def vis_mesh_traj(self, mesh):
    """ Visualize mesh together with trajectory. """
    pose_points = self.poses[:, :3, 3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pose_points))
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)
    # pcd.estimate_normals()
    o3d.visualization.draw_geometries([mesh, pcd, origin])


# debugging
def test_tile_map_vertex():
  mesh_file = '/path/to/mesh_file'
  pose_file = '/path/to/pose_file'
  poses = load_poses(pose_file)
  
  submap_test = MapModule(poses, mesh_file)
  
  for idx in range(len(submap_test.tiles)):
    tile = submap_test.tiles[idx]
    
    pcd = o3d.geometry.PointCloud()
    vertices = np.array(tile.vertices).reshape((-1, 3))
    pcd.points = o3d.utility.Vector3dVector(vertices)
    
    o3d.visualization.draw_geometries([pcd])


def test_get_map():
  mesh_file = '/path/to/mesh_file'
  pose_file = '/path/to/pose_file'
  poses = load_poses(pose_file)
  
  submap_test = MapModule(poses, mesh_file, keep_tile_maps=True)
  
  # get global map
  global_mesh = submap_test.get_global_map()
  o3d.visualization.draw_geometries([global_mesh])
  
  # get local map
  for idx in range(len(submap_test.tiles)):
    global_mesh = submap_test.get_local_map(idx)
    o3d.visualization.draw_geometries([global_mesh])


def test_get_rearranged_vertices_buffer():
  mesh_file = '/path/to/mesh_file'
  pose_file = '/path/to/pose_file'
  poses = load_poses(pose_file)
  
  submap_test = MapModule(poses, mesh_file)
  global_vertices = np.array(submap_test.global_vertices).reshape((-1, 3))
  
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(global_vertices)
  o3d.visualization.draw_geometries([pcd])


def test_average_height():
  mesh_file = '/path/to/mesh_file'
  pose_file = '/path/to/pose_file'
  poses = load_poses(pose_file)
  
  submap_test = MapModule(poses, mesh_file)
  # get local map
  for idx in range(len(submap_test.tiles)):
    print("tile_idx: ", idx, " z: ", submap_test.tiles[idx].z)


if __name__ == '__main__':
  # test_tile_map_vertex()
  # test_get_map()
  # test_get_rearranged_vertices_buffer()
  # test_average_height()
  pass
