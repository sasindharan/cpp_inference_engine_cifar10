import numpy as np
import json
import os

EPS = 1e-5

def batchnorm_numpy(x, gamma, beta, mean, var):
    out = np.zeros_like(x)
    for c in range(x.shape[0]):
        out[c] = gamma[c] * (x[c] - mean[c]) / np.sqrt(var[c] + EPS) + beta[c]
    return out


def write_config(C, H, W):
    config = {
        "layers": [
            {
                "name": "bn_test",
                "type": "batchnorm",

                "input": "../data/input/bn_input.bin",
                "output": "../data/output/bn_output.bin",

                "weight": "../data/input/bn_weight.bin",
                "bias": "../data/input/bn_bias.bin",
                "running_mean": "../data/input/bn_mean.bin",
                "running_var": "../data/input/bn_var.bin",

                "input_shape": [1, C, H, W]
            }
        ]
    }

    with open("../configs/json/bn_test.json", "w") as f:
        json.dump(config, f, indent=4)


def run_case(case_id, C, H, W):

    print(f"\nRunning BatchNorm Test Case {case_id}")

    # Generate data
    x = np.random.randn(C, H, W).astype(np.float32)
    gamma = np.random.randn(C).astype(np.float32)
    beta = np.random.randn(C).astype(np.float32)
    mean = np.random.randn(C).astype(np.float32)
    var = np.abs(np.random.randn(C).astype(np.float32))

    # Save inputs
    x.tofile("../data/input/bn_input.bin")
    gamma.tofile("../data/input/bn_weight.bin")
    beta.tofile("../data/input/bn_bias.bin")
    mean.tofile("../data/input/bn_mean.bin")
    var.tofile("../data/input/bn_var.bin")

    # Write config dynamically
    write_config(C, H, W)

    # Run engine
    os.system("cd ../build && engine.exe test_bn")

    # Load outputs
    cpp = np.fromfile("../data/output/bn_output.bin", dtype=np.float32)
    ref = batchnorm_numpy(x, gamma, beta, mean, var).flatten()

    diff = np.max(np.abs(cpp - ref))

    print("Max diff:", diff)
    print("PASS " if diff < 1e-5 else "FAIL ")


# ✅ 5 REQUIRED TEST CASES
run_case(1, 3, 32, 32)
run_case(2, 1, 8, 8)
run_case(3, 16, 16, 16)
run_case(4, 8, 4, 4)
run_case(5, 32, 32, 32)