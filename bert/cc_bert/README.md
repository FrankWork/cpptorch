## 依赖

jsoncpp

```bash
wget https://github.com/open-source-parsers/jsoncpp/archive/1.8.4.tar.gz
tar zxvf 1.8.4.tar.gz
cd jsoncpp-1.8.4
mkdir -p build/release
cd build/release
cmake -DCMAKE_BUILD_TYPE=release \
	-DBUILD_STATIC_LIBS=OFF \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_INSTALL_PREFIX=${HOME}/local \
	-G "Unix Makefiles" \
	../..
make
make install 
#-DCMAKE_INSTALL_INCLUDEDIR=include/jsoncpp


#程序运行时的动态库加载路径
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${HOME}/local/lib64"
```

gflags

```bash
wget https://github.com/gflags/gflags/archive/v2.2.2.tar.gz
tar zxvf v2.2.2.tar.gz
cd gflags-2.2.2
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=ON \
	-DCMAKE_INSTALL_PREFIX=${HOME}/local \
	..
        
make
make install

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${HOME}/local/lib"
```
## 编译

```bash
cd cc_bert/
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=${path_to_libtorch} .. 
make -j4

./bert
```
