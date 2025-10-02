FROM nvcr.io/nvidia/tensorflow:24.10-tf2-py3

RUN apt-get update
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt-get install -y less vim git wget curl unzip 
RUN pip3 install --upgrade pip
RUN pip3 install scikit-learn scipy statsmodels pandas matplotlib tqdm opencv-python pillow tf2onnx onnxruntime-gpu
