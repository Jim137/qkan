#!/bin/bash

mkdir -p cifar10
cd cifar10

wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

# Unzip the downloaded file
tar -xzf cifar-10-python.tar.gz
