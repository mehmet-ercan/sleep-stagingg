# ANACONDA

- conda create -n tensorgpu python=3.10 -y
- conda activate tensorgpu
- conda deactivate
- conda env remove -n tensorgpu
- conda env list
 
- python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


# VENV

- python3 -m venv torchgpu
- source torchgpu
- deactivate
- 