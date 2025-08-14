FROM nvcr.io/nvidia/tensorrt:25.06-py3

RUN apt-get update
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt-get install -y less vim git wget curl unzip 
RUN pip3 install --upgrade pip
RUN pip3 install scikit-learn scipy statsmodels pandas matplotlib tqdm opencv-python pillow torch torchvision onnx
