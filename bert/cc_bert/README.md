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
