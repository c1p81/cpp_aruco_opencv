cmake_minimum_required(VERSION 3.5.0)
project(quincinetto VERSION 0.1.0 LANGUAGES C CXX)

#set(OpenCV_DIR "/home/luca/opencv/build")
#set(OpenCV_DIR "/home/luca/open/build/unix-install/")
find_package(OpenCV REQUIRED)
message(STATUS "Using OpenCV version ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

add_executable(quincinetto main.cpp)

target_include_directories(quincinetto PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(quincinetto PRIVATE ${OpenCV_LIBS})

include(CTest)
enable_testing()
