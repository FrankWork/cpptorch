project(bert)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# torch 
find_package(Torch REQUIRED)
# gflags
find_package(gflags REQUIRED)


# jsoncpp
include_directories($ENV{HOME}/local/include)
link_directories($ENV{HOME}/local/lib64)

# binary file
add_executable(bert model.cc)

# torch
target_link_libraries(bert "${TORCH_LIBRARIES}")

# jsoncpp
target_link_libraries (bert jsoncpp)
# gflags
target_link_libraries(bert gflags)


set_property(TARGET bert PROPERTY CXX_STANDARD 11)


