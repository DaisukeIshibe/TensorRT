import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()

output = cuda.mem_alloc(1024)  # Example allocation, adjust size as neededcuda.mem_alloc(1024)  # Example allocation, adjust size as needed
print(f'Output memory allocated: {output}')  # Debugging output
