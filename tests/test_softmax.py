import numpy as np
import os
os.system("cd ../build && engine.exe test_softmax")

x = np.fromfile("../data/input/softmax_input.bin", dtype=np.float32)
cpp = np.fromfile("../data/output/softmax_output.bin", dtype=np.float32)

exp_x = np.exp(x - np.max(x))
ref = exp_x / np.sum(exp_x)

diff = np.abs(ref - cpp)

print("Max diff:", np.max(diff))
print("Mean diff:", np.mean(diff))

assert np.allclose(ref, cpp, atol=1e-5)
print("Softmax is CORRECT")