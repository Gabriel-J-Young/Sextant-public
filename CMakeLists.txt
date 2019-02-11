cmake_minimum_required(VERSION 3.1)
# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(OpenCV_DIR ../opencv/installation/OpenCV-master/lib/cmake/opencv4)

project (sextant)
find_package (OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable (galacticpositioningsystem galacticpositioningsystem.cpp)
target_link_libraries(galacticpositioningsystem ${OpenCV_LIBS})

add_executable (starmapper starmapper.cpp)
target_link_libraries(starmapper ${OpenCV_LIBS})