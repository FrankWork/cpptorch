https://pytorch.org/cppdocs

https://pytorch.org/cppdocs/installing.html


```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip

example-app/
  CMakeLists.txt
  example-app.cc

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/full/path/to/libtorch .. 
make -j4

./example-app
```


https://github.com/goldsborough/examples/tree/cpp/cpp