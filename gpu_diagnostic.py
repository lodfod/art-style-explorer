import torch
import cv2
import sys
import platform

print("=== GPU Diagnostic Report ===")
print(f"Python version: {platform.python_version()}")
print(f"PyTorch version: {torch.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print("\n=== PyTorch GPU Detection ===")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("No CUDA-compatible GPU detected by PyTorch")

print("\n=== OpenCV GPU Detection ===")
try:
    cv_cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"CUDA-enabled devices: {cv_cuda_count}")
    
    if cv_cuda_count > 0:
        for i in range(cv_cuda_count):
            device_info = cv2.cuda.getDevice()
            print(f"Device {i} info: {device_info}")
    else:
        print("No CUDA-enabled devices detected by OpenCV")
        
except Exception as e:
    print(f"Error checking OpenCV CUDA support: {e}")
    print("This might indicate that OpenCV was not built with CUDA support")

print("\n=== Possible Solutions ===")
if not torch.cuda.is_available():
    print("1. Check if your system has a CUDA-compatible GPU")
    print("2. Install CUDA toolkit and cuDNN")
    print("3. Reinstall PyTorch with CUDA support: https://pytorch.org/get-started/locally/")

if torch.cuda.is_available() and cv2.cuda.getCudaEnabledDeviceCount() == 0:
    print("1. OpenCV might not be built with CUDA support")
    print("2. Try reinstalling OpenCV with CUDA support:")
    print("   pip uninstall opencv-python")
    print("   pip install opencv-contrib-python") 