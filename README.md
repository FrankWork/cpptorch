https://pytorch.org/cppdocs

https://pytorch.org/cppdocs/installing.html


```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip

example-app/
  CMakeLists.txt
  example-app.cc

cd example-app/
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/full/path/to/libtorch .. 
make -j4

./example-app
```


https://github.com/goldsborough/examples/tree/cpp/cpp


# libtorch

```
include/
	- ATen/       : The foundational tensor and mathematical operation library
	- c10/        : Device type, meta programming, logging, flags, ...
	- caffe2/     : A New Lightweight, Modular, and Scalable Deep Learning Framework
	- torch/csrc/autograd: Augments ATen with automatic differentiation
```

# bugs

mac:

dyld: Library not loaded: @rpath/libmklml.dylib

https://github.com/intel/mkl-dnn/releases
1. Downloaded the a .tar.gz of the latest release of mklml for my platform
2. Extract the .tar.gz.
3. Copy the .dylib files from lib/ within the extracted folder to libtorch/lib/

