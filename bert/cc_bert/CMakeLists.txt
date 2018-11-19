cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(bert)

find_package(Torch REQUIRED)

add_executable(bert model.cc)
target_link_libraries(bert "${TORCH_LIBRARIES}")
set_property(TARGET bert PROPERTY CXX_STANDARD 11)

