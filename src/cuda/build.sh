#!/usr/bin/env bash
# NOTE: This bash is related to build.rs

rm -rf build

mkdir build
cd build

cmake ..

make -j12

#cd back to cuda/. dir
cd ..