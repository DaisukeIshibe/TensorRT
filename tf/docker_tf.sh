#!/bin/bash
docker run -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v $HOME:$HOME -u $(id -u $USER):$(id -g $USER) -it --gpus all --ipc=host  --rm --workdir $(pwd) tensorflow:20251004 /bin/bash
