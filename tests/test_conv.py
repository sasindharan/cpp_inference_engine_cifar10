import numpy as np
import json
import os

# ---------- NUMPY REFERENCE ----------
def conv2d_numpy(x, w, b, stride=1, padding=0):
    C_out, C_in, K, _ = w.shape
    H, W = x.shape[1], x.shape[2]

    H_out = (H - K + 2 * padding) // stride + 1
    W_out = (W - K + 2 * padding) // stride + 1

    out = np.zeros((C_out, H_out, W_out), dtype=np.float32)

    x_padded = np.pad(x, ((0,0),(padding,padding),(padding,padding)))

    for oc in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                region = x_padded[:, i*stride:i*stride+K, j*stride:j*stride+K]
                out[oc,i,j] = np.sum(region * w[oc]) + b[oc]

    return out


# ---------- WRITE CONFIG ----------
def write_config(C_in, H, W, C_out, K, stride, padding):

    config = {
        "layers": [
            {
                "name": "conv_test",
                "type": "conv",

                "input": "../data/input/conv_input.bin",
                "output": "../data/output/conv_output.bin",

                "weights": "../data/input/conv_weight.bin",
                "bias": "../data/input/conv_bias.bin",

                "input_shape": [1, C_in, H, W],
                "output_shape": [1, C_out, 0, 0],  # dummy

                "kernel_size": K,
                "stride": stride,
                "padding": padding
            }
        ]
    }

    with open("../configs/json/conv_test.json", "w") as f:
        json.dump(config, f, indent=4)


# ---------- TEST RUN ----------
def run_case(case_id, C_in, H, W, C_out, K, stride, padding):

    print(f"\nRunning Conv Test Case {case_id}")

    # Generate data
    x = np.random.randn(C_in, H, W).astype(np.float32)
    w = np.random.randn(C_out, C_in, K, K).astype(np.float32)
    b = np.random.randn(C_out).astype(np.float32)

    # Save
    x.tofile("../data/input/conv_input.bin")
    w.tofile("../data/input/conv_weight.bin")
    b.tofile("../data/input/conv_bias.bin")

    # Write dynamic config
    write_config(C_in, H, W, C_out, K, stride, padding)

    # Run engine
    os.system("cd ../build && engine.exe test_conv")

    # Load output
    cpp = np.fromfile("../data/output/conv_output.bin", dtype=np.float32)
    ref = conv2d_numpy(x, w, b, stride, padding).flatten()

    diff = np.max(np.abs(cpp - ref))

    print("Max diff:", diff)
    print("PASS" if diff < 1e-4 else "FAIL")


# ---------- 5 TEST CASES ----------
run_case(1, 3, 32, 32, 8, 3, 1, 1)
run_case(2, 1, 8, 8, 4, 3, 1, 0)
run_case(3, 3, 16, 16, 6, 5, 2, 2)
run_case(4, 2, 10, 10, 3, 1, 1, 0)
run_case(5, 3, 28, 28, 10, 3, 2, 1)