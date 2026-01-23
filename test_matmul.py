import torch
import time


# Matrix shapes
n = 2**10
m = 2**10

# Create random matrices on CUDA
A = torch.randn(n, n, device="cuda")
X = torch.randn(n, m, device="cuda")

# Warmup
torch.matmul(A, X)

# Single matmul timing
torch.cuda.synchronize()
start = time.time()
Y_single = torch.matmul(A, X)
torch.cuda.synchronize()
single_time = time.time() - start

# Split A into 4 equal parts along rows
A_split = torch.chunk(A, 4, dim=0)

# Accumulate results
torch.cuda.synchronize()
start = time.time()
Y_parts = [torch.matmul(a, X) for a in A_split]
Y_accum = torch.cat(Y_parts, dim=0)
torch.cuda.synchronize()
split_time = time.time() - start

print(f"Single matmul time: {single_time:.4f} seconds")
print(f"Split & accumulate time: {split_time:.4f} seconds")
