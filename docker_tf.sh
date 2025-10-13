#!/bin/bash
docker run -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v $HOME:$HOME -u $(id -u $USER):$(id -g $USER) -it --gpus all --ipc=host  --rm --workdir $(pwd) nvcr.io/nvidia/tensorflow:25.02-tf2-py3 /bin/bash
