import numpy as np
import json
import os

# ---------- NUMPY ----------
def maxpool_numpy(x, k, stride):
    C, H, W = x.shape

    H_out = (H - k) // stride + 1
    W_out = (W - k) // stride + 1

    out = np.zeros((C, H_out, W_out), dtype=np.float32)

    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                out[c, i, j] = np.max(
                    x[c, i*stride:i*stride+k, j*stride:j*stride+k]
                )

    return out


# ---------- CONFIG ----------
def write_pool_config(C, H, W, k, stride):

    config = {
        "layers": [
            {
                "name": "pool_test",
                "type": "maxpool",

                "input": "../data/input/pool_input.bin",
                "output": "../data/output/pool_output.bin",

                "input_shape": [1, C, H, W],
                "kernel_size": k,
                "stride": stride
            }
        ]
    }

    with open("../configs/json/pool_test.json", "w") as f:
        json.dump(config, f, indent=4)


# ---------- TEST ----------
def run_case(case_id, C, H, W, k, stride):

    print(f"\nRunning MaxPool Test Case {case_id}")

    x = np.random.randn(C, H, W).astype(np.float32)
    x.tofile("../data/input/pool_input.bin")

    write_pool_config(C, H, W, k, stride)

    os.system("cd ../build && engine.exe test_pool")

    cpp = np.fromfile("../data/output/pool_output.bin", dtype=np.float32)
    ref = maxpool_numpy(x, k, stride).flatten()

    diff = np.max(np.abs(cpp - ref))

    print("Max diff:", diff)
    print("PASS" if diff < 1e-6 else "FAIL")


# ---------- 5 TEST CASES ----------
run_case(1, 3, 32, 32, 2, 2)
run_case(2, 1, 8, 8, 2, 2)
run_case(3, 3, 16, 16, 3, 1)
run_case(4, 2, 10, 10, 2, 1)
run_case(5, 3, 28, 28, 2, 2)