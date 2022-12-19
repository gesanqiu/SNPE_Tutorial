# Create by Ricardo Lu in 12/16/2022

PROJECT(AlgYolov5s)

add_library(${PROJECT_NAME}
    SHARED
    AlgYolov5s.cpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${GST_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${JSON_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ./
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
    ${OpenCV_LIBS}
    ${GLIB_LIBRARIES}
    ${JSON_LIBRARIES}
    YOLOv5s
)

install(
    TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION /opt/thundersoft/algs/lib
)

install(
    FILES yolov5s.json ${CMAKE_SOURCE_DIR}/model/yolov5s_labels.txt
    DESTINATION /opt/thundersoft/configs
)

install(
    FILES ${CMAKE_SOURCE_DIR}/model/yolov5s.dlc
    DESTINATION /opt/thundersoft/alg/model
)