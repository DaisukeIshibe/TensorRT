# tensorflowrt
Practice repository for TensorFlowRT

Build command for tensorrt 25.06 with python3 and pytorch 25.05 Docker image:
> docker build ./ -t tensorrt_25.06-py3:25.05_pytorch

Make a model for TensorRT:
1. Go to "tf" directory and run "docker_tf.sh"
2. Run "python cifar10.py" -> This creates a model "cifar10_vgg_model" with SavedModel format.
3. Run "python -m tf2onnx.convert --saved-model cifar10_vgg_model --opset 18 --output model.onnx" -> This converts the Saved model to ONNX format.
4. Run "trtexec --onnx=model.onnx --saveEngine=model.trt" to convert an onnx to a trt format.
5. Exit the docker container.
6. Go to "trt" directory and run "docker_trt.sh"
7. Run "g++ -I/usr/include/aarch64-linux-gnu -I/usr/local/cuda/include -std=c++14 -o trt_cpp_test trt_cpp_test.cpp -lnvinfer -L/usr/local/cuda-12.9/targets/x86_64-linux/lib -lcudart"
