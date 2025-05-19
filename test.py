import time
import numpy as np
import cupy as cp

n = 50000
a_cpu = np.random.rand(n, n)
b_cpu = np.random.rand(n, n)

# NumPy
t0 = time.time()
_ = a_cpu @ b_cpu
print("NumPy matmul:", time.time() - t0)

# CuPy
a_gpu = cp.asarray(a_cpu)
b_gpu = cp.asarray(b_cpu)
cp.cuda.Stream.null.synchronize()  # 确保所有 GPU 前期工作完成
t1 = time.time()
_ = a_gpu @ b_gpu
cp.cuda.Stream.null.synchronize()  # 等待 GPU 完成
print("CuPy  matmul:", time.time() - t1)

print("可用 GPU 数量：", cp.cuda.runtime.getDeviceCount())






