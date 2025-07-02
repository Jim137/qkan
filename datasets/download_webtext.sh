#!/bin/bash

mkdir -p webtext
cd webtext

wget -c https://openaipublic.blob.core.windows.net/gpt-2/output-dataset/v1/webtext.test.jsonl
wget -c https://openaipublic.blob.core.windows.net/gpt-2/output-dataset/v1/webtext.train.jsonl
wget -c https://openaipublic.blob.core.windows.net/gpt-2/output-dataset/v1/webtext.valid.jsonl
