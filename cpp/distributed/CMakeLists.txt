cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(dist-mnist)


#add_subdirectory(./gloo)
include_directories(./gloo)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(program_name example)
#set(program_name example)
set(src example.cpp)
#set(src example.cpp)
#add_executable(dist-mnist dist-mnist.cpp)
add_executable(${program_name} ${src})
target_link_libraries(${program_name} ${TORCH_LIBRARIES} /data/build/examples/cpp/distributed/libtorch/lib/libc10d.a /data/build/examples/cpp/distributed/libtorch/lib/libgloo.a)
#target_link_libraries(${program_name} ${TORCH_LIBRARIES})
set_property(TARGET ${program_name} PROPERTY CXX_STANDARD 14)
