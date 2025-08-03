cmake_minimum_required(VERSION 3.10)
project(siamrpn_onnx_tracker)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

add_executable(tracker siamrpn_tracker.cpp)
target_link_libraries(tracker ${OpenCV_LIBS} onnxruntime)
