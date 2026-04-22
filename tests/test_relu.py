import numpy as np
import json
import os

# ---------- NUMPY ----------
def relu_numpy(x):
    return np.maximum(0, x)


# ---------- CONFIG ----------
def write_relu_config():
    config = {
        "layers": [
            {
                "name": "relu_test",
                "type": "relu",
                "input": "../data/input/relu_input.bin",
                "output": "../data/output/relu_output.bin"
            }
        ]
    }

    with open("../configs/json/relu_test.json", "w") as f:
        json.dump(config, f, indent=4)


# ---------- TEST ----------
def run_case(case_id, shape):

    print(f"\nRunning ReLU Test Case {case_id}")

    x = np.random.randn(*shape).astype(np.float32)
    x.tofile("../data/input/relu_input.bin")

    write_relu_config()

    os.system("cd ../build && engine.exe test_relu")

    cpp = np.fromfile("../data/output/relu_output.bin", dtype=np.float32)
    ref = relu_numpy(x).flatten()

    diff = np.max(np.abs(cpp - ref))

    print("Max diff:", diff)
    print("PASS" if diff < 1e-6 else "FAIL")


# ---------- 5 TEST CASES ----------
run_case(1, (10,))
run_case(2, (100,))
run_case(3, (1000,))
run_case(4, (50, 50))
run_case(5, (3, 32, 32))