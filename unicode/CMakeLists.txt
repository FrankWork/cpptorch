cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)


set(CMAKE_CXX_FLAGS "-g -pipe -W -Wall -fPIC -std=c++11 -mfpmath=sse -msse -msse2 -msse3 -fopenmp")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-Bsymbolic -rdynamic -zmuldefs")

add_subdirectory(third_party)

set(SRC
main.cc
string_util.h
string_util.cc
)

#file(GLOB SRCS "*h *.cc")
#SET_SOURCE_FILES_PROPERTIES(${SRCS} PROPERTIES GENERATED TRUE)

add_executable(app ${SRC})
set_property(TARGET app PROPERTY CXX_STANDARD 11)


