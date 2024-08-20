cmake_minimum_required(VERSION 3.0.0)
project(TeshTransTestingTaskCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (WIN32)
    set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")
    set(OpenCV_BIN_DIR "C:/opencv/build/x64/vc16/bin")

    set(OpenCV_DEBUG_DLL_FILENAME opencv_world480d.dll)
    set(OpenCV_RELEASE_DLL_FILENAME opencv_world480.dll)

    set(ONNXRUNTIME_DIR "C:/onnxruntime-win-x64-1.19.0")
else()
    set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
    set(ONNXRUNTIME_DIR "/onnxruntime")
endif()

find_package(OpenCV REQUIRED)

include_directories(include)

file(GLOB_RECURSE CURR_SOURCES src/*.cpp)

add_executable(TeshTransTestingTaskCPP ${CURR_SOURCES})

target_include_directories(TeshTransTestingTaskCPP PRIVATE "${ONNXRUNTIME_DIR}/include")

target_link_libraries(TeshTransTestingTaskCPP ${OpenCV_LIBS})

if (WIN32)
    target_link_libraries(TeshTransTestingTaskCPP "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")

    add_custom_command(TARGET TeshTransTestingTaskCPP POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:TeshTransTestingTaskCPP>")

    add_custom_command(TARGET TeshTransTestingTaskCPP POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_BIN_DIR}/${OpenCV_DEBUG_DLL_FILENAME}"
        "$<TARGET_FILE_DIR:TeshTransTestingTaskCPP>")

    add_custom_command(TARGET TeshTransTestingTaskCPP POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_BIN_DIR}/${OpenCV_RELEASE_DLL_FILENAME}"
        "$<TARGET_FILE_DIR:TeshTransTestingTaskCPP>")
endif()

if (UNIX)
    target_link_libraries(TeshTransTestingTaskCPP "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif()