cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)


project(des_mubot)


cmake_policy(SET CMP0054 NEW)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")


find_package(catkin REQUIRED COMPONENTS
  rospy
  gazebo_ros
  std_msgs
  roscpp
  roslaunch
)
find_package(gazebo REQUIRED)

catkin_package(
    CATKIN_DEPENDS
    DEPENDS
    rospy
    gazebo_ros
    std_msgs
    roscpp
    roslaunch
)

catkin_python_setup(
)


include_directories(include ${catkin_INCLUDE_DIRS} )
include_directories(${roscpp_INCLUDE_DIRS})
include_directories(${std_msgs_INCLUDE_DIRS})
include_directories(${GAZEBO_INCLUDE_DIRS})


link_directories(${GAZEBO_LIBRARY_DIRS})

# Build our plugin
add_library(actuator_plugin SHARED actuator_plugin.cc)
target_link_libraries(actuator_plugin ${GAZEBO_LIBRARIES} ${roscpp_LIBRARIES})
add_library(Hydro_plugin SHARED Hydro_plugin.cc)
target_link_libraries(Hydro_plugin ${GAZEBO_LIBRARIES} ${roscpp_LIBRARIES})

catkin_install_python(PROGRAMS
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch worlds urdf meshes config Database
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(TARGETS
    actuator_plugin
    Hydro_plugin
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    )

  