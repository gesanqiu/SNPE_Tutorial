# create by Ricardo Lu in 05-17-2022

cmake_minimum_required(VERSION 3.10)

project(SNPETask)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
  set(PKG_CONFIG_EXECUTABLE ${VCPKG_DIR}/installed/x64-windows/tools/pkgconf/pkgconf.exe)
  set(SNPE_INCLUDE_DIR D:/Tools/snpe-2.6.0.4131/include/zdl)
  set(SNPE_LIBRARY_DIR D:/Tools/snpe-2.6.0.4131/lib/aarch64-windows-vc19)
  set(UTILITY_SOURCES ${PROJECT_SOURCE_DIR}/utility/unpack.c
		      ${PROJECT_SOURCE_DIR}/utility/unparse.c
                      ${PROJECT_SOURCE_DIR}/utility/gettimeofday.c)
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
  set(OpenCV_DIR "/opt/thundersoft/opencv-4.5.5/lib/cmake/opencv4")
  set(SNPE_INCLUDE_DIR /usr/include/zdl)
  set(SNPE_LIBRARY_DIR /usr/lib/aarch64-linux-gnu)
  set(PTHREAD_DL_LIBS pthread dl)
endif()
find_package(OpenCV REQUIRED)
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
  find_package(gflags REQUIRED)
  find_package(jsoncpp REQUIRED)
endif()
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

include(FindPkgConfig)
pkg_check_modules(GST     REQUIRED gstreamer-1.0)
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
  pkg_check_modules(GFLAGS  REQUIRED gflags)
  pkg_check_modules(JSONCPP REQUIRED jsoncpp)
endif()
pkg_check_modules(GSTAPP  REQUIRED gstreamer-app-1.0)
pkg_check_modules(JSON    REQUIRED json-glib-1.0)

include_directories(
    ${SNPE_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/inc
    ${PROJECT_SOURCE_DIR}/snpetask
    ${PROJECT_SOURCE_DIR}/utility
    ${PROJECT_SOURCE_DIR}/yolov5s/inc
    ${OpenCV_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}
    ${JSON_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS}
    ${GSTAPP_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}     # jsoncpp header directory
)

link_directories(
    ${OpenCV_LIBRARY_DIRS}
    ${GFLAGS_LIBRARY_DIRS}
    ${fmt_LIBRARY_DIRS}
    ${spdlog_LIBRARY_DIRS}
    ${JSON_LIBRARY_DIRS}
    ${GST_LIBRARY_DIRS}
    ${GSTAPP_LIBRARY_DIRS}
    ${JSONCPP_LIBRARY_DIRS}     # jsoncpp library directory
    ${PROJECT_SOURCE_DIR}/lib
    ${SNPE_LIBRARY_DIR}
)

set(LIBRARY_OUTPUT_PATH  ${CMAKE_SOURCE_DIR}/lib)

# Config Logger
if(NOT DEFINED LOG_LEVEL)
    message(STATUS "Not define log print level, default is 'info'")
    set(LOG_LEVEL "info")
endif()
add_definitions(-DLOG_LEVEL="${LOG_LEVEL}")
message(STATUS "log level: ${LOG_LEVEL}")

option(DUMP_LOG "Dump log into a file." OFF)
option(MULTI_LOG "Dump log and stdout." OFF)

if(DUMP_LOG OR MULTI_LOG)
    if(NOT DEFINED LOG_PATH)
        message(STATUS "Not define log path, use default")
        set(LOG_PATH "./log")
        message(STATUS "log path: ${LOG_PATH}")
    endif()
    if(NOT DEFINED LOG_FILE_PREFIX)
        message(STATUS "Not define log name prefix, use default")
        set(LOG_FILE_PREFIX ${PROJECT_NAME})
        message(STATUS "log file prefix: ${LOG_FILE_PREFIX}")
    endif()

    add_definitions(
        -DDUMP_LOG
        -DLOG_PATH="${LOG_PATH}"
        -DLOG_FILE_PREFIX="${LOG_FILE_PREFIX}"
    )
    if(MULTI_LOG)
        message(STATUS "Multi log set.")
        add_definitions(-DMULTI_LOG)
    endif()
endif()

# Compile YOLOv5s.so target
add_subdirectory(yolov5s)

# Compile test program
add_subdirectory(test/test_image)
add_subdirectory(test/test_video)

# Compile standard algorithm module
add_subdirectory(alg/yolov5s)
