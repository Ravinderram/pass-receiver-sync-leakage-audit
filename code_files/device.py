"""Where the tensors live. One definition, imported by every torch script.

Kept out of config.py on purpose: config is imported by step2 through step7,
none of which touch torch, and importing torch costs a few seconds of startup
every time. Only the scripts that actually train something import this.
"""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ON_GPU = DEVICE.type == "cuda"

# pinned host memory only helps when there is a device to copy to, and it is
# what makes non_blocking=True on .to() actually asynchronous
PIN_MEMORY = ON_GPU

if ON_GPU:
    # every batch in this project has the same shape, so letting cuDNN pick and
    # cache the fastest kernels is free
    torch.backends.cudnn.benchmark = True


def banner():
    """Say plainly what we are running on. Printed once per script."""
    if ON_GPU:
        i = torch.cuda.current_device()
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
        print(f"Using Device: GPU ({name})")
        print(f"  {vram:.1f} GB VRAM · CUDA {torch.version.cuda} · "
              f"torch {torch.__version__}")
    else:
        print(f"Using Device: CPU (torch {torch.__version__})")
        if not torch.cuda.is_available():
            print("  torch.cuda.is_available() is False — this build has no CUDA,"
                  " or no GPU was found")


def seed_everything(seed):
    """Seed both generators. Without the cuda call, GPU runs are not repeatable."""
    torch.manual_seed(seed)
    if ON_GPU:
        torch.cuda.manual_seed_all(seed)


def to_numpy(t):
    """Bring a tensor back to host memory safely.

    detach() drops the autograd graph, cpu() crosses the device boundary. Calling
    .numpy() on a CUDA tensor raises; doing it on one that still requires grad
    raises too. This helper exists so neither can be forgotten at a call site.
    """
    return t.detach().cpu().numpy()


def batch_size():
    """A 35k-parameter model does not fill a GPU. Bigger batches keep it busy.

    On CPU the configured value is right. On GPU, batches of 64 spend more time
    launching kernels than computing, so we raise it.
    """
    import config as C
    return C.BATCH_SIZE * 4 if ON_GPU else C.BATCH_SIZE
