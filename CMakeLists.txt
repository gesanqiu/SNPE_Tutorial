# create by Ricardo Lu in 05-17-2022

cmake_minimum_required(VERSION 3.10)

project(SNPETask)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")

set(OpenCV_DIR "/opt/thundersoft/opencv-4.5.5/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

include(FindPkgConfig)
pkg_check_modules(GFLAGS REQUIRED gflags)
pkg_check_modules(JSON   REQUIRED json-glib-1.0)

include_directories(
    /usr/include/zdl
    ${PROJECT_SOURCE_DIR}/inc
    ${PROJECT_SOURCE_DIR}/snpetask
    ${PROJECT_SOURCE_DIR}/utility
    ${PROJECT_SOURCE_DIR}/yolov5s/inc
    ${OpenCV_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}
    ${JSON_INCLUDE_DIRS}
)

link_directories(
    ${OpenCV_LIBRARY_DIRS}
    ${GFLAGS_LIBRARY_DIRS}
    ${spdlog_LIBRARY_DIRS}
    ${JSON_LIBRARY_DIRS}
    ${PROJECT_SOURCE_DIR}/lib
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
add_executable(test-yolov5
    ${PROJECT_SOURCE_DIR}/test/main.cpp
)

target_link_libraries(test-yolov5
    PUBLIC
    pthread
    dl
    ${OpenCV_LIBS}
    ${GFLAGS_LIBRARIES}
    ${JSON_LIBRARIES}
    ${spdlog_LIBRARIES}
    YOLOv5s
)