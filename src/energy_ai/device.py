"""Pick the best available compute device: NVIDIA (cuda), Intel GPU (xpu), Apple (mps) or CPU."""
import torch


def get_device(verbose: bool = True) -> torch.device:
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")
    if verbose:
        name = "CPU"
        if device.type == "xpu":
            name = torch.xpu.get_device_name(0)
        elif device.type == "cuda":
            name = torch.cuda.get_device_name(0)
        print(f"Using device: {device} ({name})")
    return device
