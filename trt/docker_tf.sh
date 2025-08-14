#!/bin/bash
docker run -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v $HOME:$HOME -u $(id -u $USER):$(id -g $USER) -it --gpus all --ipc=host  --rm --workdir $(pwd) tensorrt_25.06-py3:25.05_pytorch /bin/bash
