PROJECT(test-image)

add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/main.cpp
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
    ${PTHREAD_DL_LIBS}
    fmt::fmt
    ${OpenCV_LIBS}
    ${GFLAGS_LIBRARIES}
    ${JSON_LIBRARIES}
    ${spdlog_LIBRARIES}
    YOLOv5s
)
