cmake_minimum_required(VERSION 3.20)

project(QRscanner_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ZXING_READERS ON CACHE BOOL "" FORCE)
set(ZXING_WRITERS OFF CACHE STRING "" FORCE)
set(ZXING_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZXING_BLACKBOX_TESTS OFF CACHE BOOL "" FORCE)
set(ZXING_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(ZXING_PYTHON_MODULE OFF CACHE BOOL "" FORCE)
set(ZXING_C_API OFF CACHE BOOL "" FORCE)
add_subdirectory(zxing-cpp_finder/zxing-cpp EXCLUDE_FROM_ALL)

find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs objdetect)

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

target_include_directories(qrscanner_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qrscanner_cpp
    PUBLIC
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_objdetect
        ZXing::ZXing
)

add_executable(qrscanner_cli main.cpp)
target_link_libraries(qrscanner_cli PRIVATE qrscanner_cpp)
