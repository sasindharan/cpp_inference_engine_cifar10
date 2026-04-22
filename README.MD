This project implements a deep learning inference engine in C++ using FP32 precision. The goal was to manually implement core neural network operators such as Convolution, Batch Normalization, ReLU, Max Pooling, Fully Connected, and Softmax, and use them to execute a CIFAR-10 model.
The system follows a config-driven approach, where the model architecture is defined in a JSON file (model.json). The engine reads this configuration and executes the model layer by layer. This makes the design flexible and easy to extend.
The implementation also includes unit testing for each operator and full model validation by comparing outputs with a PyTorch implementation.

# FOLDER STRUCTURE

cpp_inference_engine/
│
├── Operators/
│   ├── src/        # Operator implementations (conv, bn, relu, etc.)
│   └── include/    # Header files for operators
│
├── utilities/
│   ├── src/        # Utility functions (I/O, logging, comparison)
│   └── include/
│
├── src/
│   └── config_engine.cpp   # Main execution engine
│
├── include/        # Common headers
│
├── Test_Operators/ # (Optional structure for operator tests)
│
├── tests/          # Python test files (unit + model testing)
│
├── data/
│   ├── input/      # Input tensors
│   ├── weights/    # Model weights
│   ├── output/     # C++ outputs
│   └── reference/  # PyTorch reference outputs
│
├── configs/
│   └── json/
│       ├── model.json        # Full model config
│       └── *_test.json       # Unit test configs
│
├── report/
│   └── report.txt
│
├── build/          # CMake build directory
│
├── CMakeLists.txt
├── Readme.txt
└── .gitignore

# OS / TOOLS / VERSION USED

Operating System: Windows 10/11
Compiler: GCC (MinGW / WinLibs)
Build System: CMake (Ninja generator)
Language: C++ (C++17)
Python Version: Python 3.x
Libraries Used:
NumPy (for testing)
PyTorch (for model validation)

# BUILD AND RUN INSTRUCTIONS

## BUILD THE PROJECT
cd cpp_inference_engine

mkdir build

cd build

cmake ..

cmake --build .

## RUN THE CPP ENGINE
cd build

.\engine.exe

<!-- This will:
Execute the model using model.json
Save outputs to data/output/ -->

## RUN THE UNIT TEST

cd tests

python test_conv.py

python test_bn.py

python test_relu.py

python test_maxpool.py

python test_fc.py

python test_softmax.py

## RUN THE MODEL TEST

cd tests

python test_model.py

## INTERPRETING RESULTS

### UNIT TESTING 
Each test prints:

Max diff: <value>

PASS / FAIL

If PASS, the operator is correct

Small differences (~1e-6) are expected due to floating-point precision

### MODEL TESTING
Max diff : ~1e-7

Mean diff: small

MATCH

MATCH → C++ and PyTorch outputs are equivalent

Differences around 1e-7 are normal for FP32

# EXECUTION LOGS
conv1 → conv → 14 ms → PASS

relu1 → relu → 1 ms → PASS

<!-- These logs show:

Layer execution order
Execution time
Pass/Fail status -->

All operators were implemented manually and validated through unit testing. The full CIFAR-10 model was successfully executed and verified against a PyTorch implementation with high precision (~1e-7), confirming correctness.