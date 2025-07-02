#!/bin/bash

mkdir -p mnist
cd mnist

wget -c https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget -c https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
wget -c https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
wget -c https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz

# Unzip the downloaded files
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
