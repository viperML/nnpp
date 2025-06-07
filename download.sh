#! /usr/bin/env bash
set -eux
shopt -s failglob

dir="$(cd "$(dirname ${BASH_SOURCE[0]})"; pwd)"

mkdir -p "$dir/data"

curl -L \
    -o "$dir/data/mnist-dataset.zip" \
    https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset

pushd "$dir/data"
unzip *.zip
