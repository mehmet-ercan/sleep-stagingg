# ANACONDA

- conda create -n torchgpu python=3.10 -y
- conda activate torchgpu
- conda deactivate
- conda env remove -n torchgpu
- conda env list
 
- python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


# VENV

- python3 -m venv torchgpu
- source torchgpu
- deactivate
- 