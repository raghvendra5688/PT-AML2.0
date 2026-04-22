#!/bin/bash -l
#SBATCH -J gcn_OnlyPC
#SBATCH -o out_gcn_OnlyPC.log
#SBATCH -e out_gcn_OnlyPC.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -A A100
#SBATCH -q a100_qos
#SBATCH -w crirdcmgpu006

#module load cuda12.4/toolkit/12.6.2
module load cuda12.4/toolkit/12.4.1
python --version
nvidia-smi
nvcc --version

export MAMBA_EXE='/export/home/rmall/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/export/home/rmall/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from micromamba activate
fi
unset __mamba_setup
# <<< mamba initialize <<<

micromamba env list
micromamba activate BeatAML2.0

python3 gcn_model_optuna_OnlyPC.py
