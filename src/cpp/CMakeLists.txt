cmake_minimum_required(VERSION 3.20)

project(QRscanner_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# ZXing
# ============================================================

set(ZXING_READERS ON CACHE BOOL "" FORCE)
set(ZXING_WRITERS OFF CACHE STRING "" FORCE)
set(ZXING_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZXING_BLACKBOX_TESTS OFF CACHE BOOL "" FORCE)
set(ZXING_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(ZXING_PYTHON_MODULE OFF CACHE BOOL "" FORCE)
set(ZXING_C_API OFF CACHE BOOL "" FORCE)

add_subdirectory(
    zxing-cpp_finder/zxing-cpp
    EXCLUDE_FROM_ALL
)

# ============================================================
# OpenCV
# ============================================================

include(FetchContent)

set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(BUILD_JAVA OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    opencv
    URL https://github.com/opencv/opencv/archive/refs/tags/4.12.0.tar.gz
)

FetchContent_MakeAvailable(opencv)

# Add include directories of all OpenCV modules.
# This is needed because <opencv2/opencv.hpp> includes headers
# from modules other than core/imgproc/imgcodecs/objdetect.
file(GLOB OPENCV_MODULE_INCLUDE_DIRS
    "${opencv_SOURCE_DIR}/modules/*/include"
)

# ============================================================
# QR scanner library
# ============================================================

add_library(qrscanner_cpp
    tools.cpp
    math_base.cpp
    homographic_correct.cpp
    interpolation_correct.cpp
    TPS_correct.cpp
    vectorfield_correct.cpp
    binarizer.cpp
    decoder.cpp
    find_qr.cpp
    QRscanner.cpp
)

target_include_directories(
    qrscanner_cpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${opencv_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}
        ${OPENCV_MODULE_INCLUDE_DIRS}
)

target_link_libraries(
    qrscanner_cpp
    PUBLIC
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_objdetect
        ZXing::ZXing
)

# ============================================================
# CLI
# ============================================================

add_executable(qrscanner_cli main.cpp)

target_link_libraries(
    qrscanner_cli
    PRIVATE
        qrscanner_cpp
)