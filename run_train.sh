#!/bin/bash
# DDPM Training launcher with unbuffered output
export PYTHONUNBUFFERED=1
export PATH="/c/Users/LENOVO/AppData/Local/Microsoft/WindowsApps:$PATH"
cd "/c/Users/LENOVO/Desktop/科研项目/Diffusion"
echo "=== DDPM Training Started at $(date) ==="
echo "Python: $(which python3)"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo "---"
python3 -u train_ddpm.py 2>&1
echo "=== Training finished at $(date) ==="
