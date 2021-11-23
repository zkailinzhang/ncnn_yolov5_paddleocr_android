project(nanodetncnn)

#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

cmake_minimum_required(VERSION 3.10)

#D:\code\android\opencv-mobile-3.4.15-android
#set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.5.1-android/sdk/native/jni)

set(OpenCV_DIR D:\\code\\android\\opencv-mobile-3.4.15-android\\sdk\\native\\jni)

find_package(OpenCV REQUIRED core imgproc highgui video)


#set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20210124-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
set(ncnn_DIR D:\\code\\android\\ncnn-20210525-android-vulkan\\${ANDROID_ABI}\\lib\\cmake\\ncnn)

find_package(ncnn REQUIRED)

add_library(nanodetncnn SHARED nanodetncnn.cpp nanodet.cpp ndkcamera.cpp common.cpp clipper.cpp)

target_link_libraries(nanodetncnn ncnn ${OpenCV_LIBS} camera2ndk mediandk)
