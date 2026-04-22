import numpy as np
import json
import os

# ---------- NUMPY ----------
def fc_numpy(x, w, b):
    return x @ w.T + b


# ---------- CONFIG ----------
def write_fc_config(in_f, out_f):

    config = {
        "layers": [
            {
                "name": "fc_test",
                "type": "fc",

                "input": "../data/input/fc_input.bin",
                "output": "../data/output/fc_output.bin",

                "weights": "../data/input/fc_weight.bin",
                "bias": "../data/input/fc_bias.bin",

                "input_shape": [1, in_f],
                "output_shape": [1, out_f]
            }
        ]
    }

    with open("../configs/json/fc_test.json", "w") as f:
        json.dump(config, f, indent=4)


# ---------- TEST ----------
def run_case(case_id, in_f, out_f):

    print(f"\nRunning FC Test Case {case_id}")

    x = np.random.randn(in_f).astype(np.float32)
    w = np.random.randn(out_f, in_f).astype(np.float32)
    b = np.random.randn(out_f).astype(np.float32)

    x.tofile("../data/input/fc_input.bin")
    w.tofile("../data/input/fc_weight.bin")
    b.tofile("../data/input/fc_bias.bin")

    write_fc_config(in_f, out_f)

    os.system("cd ../build && engine.exe test_fc")

    cpp = np.fromfile("../data/output/fc_output.bin", dtype=np.float32)
    ref = fc_numpy(x, w, b)

    diff = np.max(np.abs(cpp - ref))

    print("Max diff:", diff)
    print("PASS" if diff < 1e-3 else "FAIL")


# ---------- 5 TEST CASES ----------
run_case(1, 10, 5)
run_case(2, 100, 10)
run_case(3, 256, 64)
run_case(4, 512, 128)
run_case(5, 4096, 128)