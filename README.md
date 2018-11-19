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

include/
	ATen/       : The foundational tensor and mathematical operation library
	c10/        : Device type, meta programming, logging, flags, ...
	caffe2/     : A New Lightweight, Modular, and Scalable Deep Learning Framework
	torch/csrc/autograd: Augments ATen with automatic differentiation

