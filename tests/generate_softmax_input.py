import numpy as np

x = np.random.randn(10).astype(np.float32)
x.tofile("../data/input/softmax_input.bin")

print("Saved:", "../data/input/softmax_input.bin")