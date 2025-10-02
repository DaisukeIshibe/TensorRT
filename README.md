# tensorflowrt
Practice repository for TensorFlowRT

Build command for tensorrt 25.06 with python3 and pytorch 25.05 Docker image:
> docker build ./ -t tensorrt_25.06-py3:25.05_pytorch

Make a model for TensorRT:
1. Go to "tf" directory
2. Run "python cifar10.py" -> This creates a model "cifar10_vgg_model" with SavedModel format.
3. Run "python -m tf2onnx.convert --saved-model cifar10_vgg_model --opset 18 --output model.onnx" -> This converts the Saved model to ONNX format.
