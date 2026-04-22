import json
import numpy as np
import torch
import torch.nn.functional as F
import os

EPS = 1e-5

# ---------- IO ----------
def load_vec(path):
    return np.fromfile(path, dtype=np.float32)

def load_tensor(path, shape):
    return torch.tensor(load_vec(path)).reshape(shape)

def save_vec(path, arr):
    arr.astype(np.float32).tofile(path)


# ---------- OPERATORS ----------
def conv2d(x, w, b, stride, padding):
    return F.conv2d(x, w, b, stride=stride, padding=padding)

def batchnorm(x, gamma, beta, mean, var):
    x = (x - mean[None,:,None,None]) / torch.sqrt(var[None,:,None,None] + EPS)
    return gamma[None,:,None,None]*x + beta[None,:,None,None]

def relu(x):
    return F.relu(x)

def maxpool(x, k, s):
    return F.max_pool2d(x, k, s)

def fc(x, w, b):
    return x @ w.T + b

def softmax(x):
    x = x - torch.max(x, dim=-1, keepdim=True).values
    e = torch.exp(x)
    return e / torch.sum(e, dim=-1, keepdim=True)


# ---------- PYTORCH EXECUTION ----------
def run_pytorch(config_path):

    with open(config_path) as f:
        config = json.load(f)

    tensors = {}

    # Load input
    first_input = config["layers"][0]["input"]
    tensors[first_input] = load_tensor(first_input, (1,3,32,32))

    for layer in config["layers"]:

        typ = layer["type"]

        # ----- CONV -----
        if typ == "conv":
            x = tensors[layer["input"]]

            N,C,H,W = layer["input_shape"]
            OC = layer["output_shape"][1]
            K = layer["kernel_size"]

            w = load_tensor(layer["weights"], (OC,C,K,K))
            b = load_tensor(layer["bias"], (OC,))

            x = conv2d(x, w, b, layer["stride"], layer["padding"])
            tensors[layer["output"]] = x


        # ----- BN -----
        elif typ == "batchnorm":
            x = tensors[layer["input"]]

            C = layer["input_shape"][1]

            gamma = load_tensor(layer["weight"], (C,))
            beta  = load_tensor(layer["bias"], (C,))
            mean  = load_tensor(layer["running_mean"], (C,))
            var   = load_tensor(layer["running_var"], (C,))

            x = batchnorm(x, gamma, beta, mean, var)
            tensors[layer["output"]] = x


        # ----- RELU -----
        elif typ == "relu":
            tensors[layer["output"]] = relu(tensors[layer["input"]])


        # ----- POOL -----
        elif typ == "maxpool":
            x = tensors[layer["input"]]
            tensors[layer["output"]] = maxpool(x, layer["kernel_size"], layer["stride"])


        # ----- ADD (SKIP) -----
        elif typ == "add":
            a = tensors[layer["input1"]]
            b = tensors[layer["input2"]]
            tensors[layer["output"]] = a + b


        # ----- FC -----
        elif typ == "fc":
            x = tensors[layer["input"]]

            if len(x.shape) > 2:
                x = x.view(1, -1)

            in_f  = layer["input_shape"][1]
            out_f = layer["output_shape"][1]

            w = load_tensor(layer["weights"], (out_f, in_f))
            b = load_tensor(layer["bias"], (out_f,))

            x = fc(x, w, b)
            tensors[layer["output"]] = x


        # ----- SOFTMAX -----
        elif typ == "softmax":
            x = tensors[layer["input"]]
            tensors[layer["output"]] = softmax(x)

        else:
            raise Exception(f"Unknown layer {typ}")

    last = config["layers"][-1]["output"]
    return tensors[last].detach().numpy().flatten()


# ---------- MAIN ----------
if __name__ == "__main__":

    config_path = "../configs/json/model.json"

    print("Running PyTorch reference...")
    ref = run_pytorch(config_path)
    save_vec("../data/reference/final_output.bin", ref)

    print("Running C++ engine...")
    os.system("cd ../build && engine.exe")

    print("Comparing outputs...")

    cpp = load_vec("../data/output/final_output.bin")

    diff = np.abs(ref - cpp)

    print("\n===== MODEL CHECK =====")
    print("Max diff :", np.max(diff))
    print("Mean diff:", np.mean(diff))

    if np.max(diff) < 1e-5:
        print("MATCH")
    else:
        print("MISMATCH")