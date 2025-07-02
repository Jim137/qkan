#!/bin/bash

mkdir -p cifar100
cd cifar100

wget -c https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

# Unzip the downloaded file
tar -xzf cifar-100-python.tar.gz
