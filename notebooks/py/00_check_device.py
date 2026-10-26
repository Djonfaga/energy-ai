# %% [markdown]
# # 00 · Check the compute device
# Run this first. `get_device()` returns the GPU if PyTorch can use one (NVIDIA → `cuda`, Intel → `xpu`), else the CPU.
# The first GPU operation is slow (kernels are compiled once); later calls are fast.

# %%
import time
import torch
from energy_ai import get_device

device = get_device()
print("torch", torch.__version__)

def sync():
    if device.type != "cpu":
        torch.accelerator.synchronize()

# %%
# Same matrix multiplication on the CPU and on the selected device
a = torch.randn(4096, 4096)
b = torch.randn(4096, 4096)
t = time.perf_counter(); c_cpu = a @ b; t_cpu = time.perf_counter() - t

a_d, b_d = a.to(device), b.to(device)
a_d @ b_d; sync()                                   # warm-up
t = time.perf_counter(); c_dev = a_d @ b_d; sync(); t_dev = time.perf_counter() - t

print(f"CPU {t_cpu * 1000:.0f} ms | {device} {t_dev * 1000:.0f} ms | speed-up ×{t_cpu / t_dev:.1f}")
print("results match:", torch.allclose(c_cpu, c_dev.cpu(), rtol=1e-3, atol=1e-2))

# %% [markdown]
# **Rule of thumb:** use the GPU for neural networks and big matrices; keep tiny models (a few hundred numbers) on the
# CPU, where kernel-launch overhead makes the GPU slower. On an Intel Iris Xe laptop GPU we measured 3–14× speed-ups
# for neural networks and large matrix products.
