cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(minitext)

# set cxx flags
set(CMAKE_CXX_FLAGS "-Wall -O2 -std=c++11 -g")

# set include path
include_directories(minitext)

# set the path to the library folder
add_library(dictionary SHARED dictionary.cc dictionary.h args.h matrix.cc densematrix.cc vector.cc utils.cc)

add_executable(app app.cc)

# link the libraries to the executable
target_link_libraries (app dictionary)

