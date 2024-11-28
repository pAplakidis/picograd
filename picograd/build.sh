#!/bin/bash

BUILD_TYPE="Release"
if [ "$1" == "debug" ]; then
    BUILD_TYPE="Debug"
fi

echo "[+] Building $BUILD_TYPE"

mkdir -p ./lib/
rm -rf ./lib/*
cd lib/

cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
make
