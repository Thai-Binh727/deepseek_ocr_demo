# How to Set Up DeepSeek OCR on Ubuntu

## My Setup

1. Intel i7-12700K
2. 32 GB RAM
3. RTX 5060 Ti - 16 GB VRAM
4. OS: Ubuntu 22.04
5. Python 3.12.3

## Requirements

1. NVIDIA GPU with ≥ 16 GB VRAM
2. ≥ 16 GB system RAM

## Installation

### 1. Install `uv`

With `curl`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

With `wget`:

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

### 2. Install Python

Download here: [Python ≥ 3.11](https://www.python.org/downloads/)

### 3. Install CUDA Toolkit

1. Go to: [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
2. Recommended version: CUDA ≥ 12.9.0 for RTX 5000 series
3. For *Installation Type*, choose `runfile (local)`
4. You’ll see two commands, for the 2nd one, add `--silent` and `--toolkit` so it installs silently and only the toolkit (no extras)
5. After download, set `CUDA_HOME`:

**Permanent**:

```bash
nano ~/.bashrc
```

Add at the bottom (replace `x` with your version):

```bash
export CUDA_HOME=/usr/local/cuda-12.x
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Reload:

```bash
source ~/.bashrc
```

**Temporary (for one terminal session):**

```bash
export CUDA_HOME=/usr/local/cuda-12.x
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

6. Verify:

```bash
nvcc --version
```

### 4. Install Python Dependencies

Use `uv` for all installations.

1. Install PyTorch: its CUDA version must match your CUDA toolkit version
2. Install:

```
transformers
tokenizers
einops
addict
easydict
packaging
```

3. Install `flash-attn`

   - (Optional) Install `ninja` for faster compilation:

   ```bash
   uv add ninja
   ```

   - If your machine has < 96 GB RAM and many CPU cores, limit parallel jobs:

   ```bash
   MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
   ```

   - Or install without using `ninja`:

   ```bash
   taskset -c 0,1,2,3 uv pip install flash-attn --no-build-isolation
   ```