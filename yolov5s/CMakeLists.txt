# create by Ricardo Lu in 07-09-2022

cmake_minimum_required(VERSION 3.10)

project(YOLOv5s)

add_library(${PROJECT_NAME}
    SHARED
    ${PROJECT_SOURCE_DIR}/src/YOLOv5s.cpp
    ${PROJECT_SOURCE_DIR}/src/YOLOv5sImpl.cpp
    ${CMAKE_SOURCE_DIR}/snpetask/SNPETask.cpp
)

target_link_libraries(${PROJECT_NAME}
    SNPE
    pthread
    dl
    ${OpenCV_LIBS}
    ${spdlog_LIBRARIES}
)

install(
    TARGETS YOLOv5s
    LIBRARY DESTINATION /opt/thundersoft/algs/lib
)