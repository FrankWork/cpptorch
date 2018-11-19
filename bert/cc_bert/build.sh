cd build
cmake -DCMAKE_PREFIX_PATH=${path_to_libtorch} .. 
make -j4

#./bert
