project(particle)
find_package(catkin REQUIRED COMPONENTS
 roscpp
 rospy
 pcl_conversions
 tf_conversions
)
find_package(PCL REQUIRED QUIET)
find_package(Eigen3 REQUIRED)

catkin_package(
 CATKIN_DEPENDS
)

include_directories(
 ${catkin_INCLUDE_DIRS}
 ${PCL_INCLUDE_DIRS}
 ${EIGEN3_INCLUDE_DIR}
)
add_executable(particle src/particle.cpp)
target_link_libraries(particle ${catkin_LIBRARIES} ${PCL_LIBRARIES})


