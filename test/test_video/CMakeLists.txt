PROJECT(test-video)

add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/VideoPipeline.cpp
    ${PROJECT_SOURCE_DIR}/VideoAnalyzer.cpp
    ${PROJECT_SOURCE_DIR}/main.cpp
    ${UTILITY_SOURCES}
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${JSONCPP_INCLUDE_DIRS}     # jsoncpp header directory
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
    ${PTHREAD_DL_LIBS}
    fmt::fmt
    ${OpenCV_LIBS}
    ${GFLAGS_LIBRARIES}
    ${JSON_LIBRARIES}
    ${spdlog_LIBRARIES}
    ${GST_LIBRARIES}
    ${GSTAPP_LIBRARIES}
    YOLOv5s
    jsoncpp
    mosquitto
)
