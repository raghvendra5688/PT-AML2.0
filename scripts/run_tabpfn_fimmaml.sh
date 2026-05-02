#!/bin/bash -l
#SBATCH -J tabpfn_fimmaml
#SBATCH -o out_tabpfn_fimmaml_v2.log
#SBATCH -e out_tabpfn_fimmaml_v2.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH --mem=120000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -x crirdchpxd005
#SBATCH -w crirdchpxd001

module load cuda12.6/toolkit/12.6.2
python --version
nvidia-smi
nvcc --version

export MAMBA_EXE='/export/home/rmall/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/export/home/rmall/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"
fi
unset __mamba_setup

micromamba env list
micromamba activate BeatAML2.0

# Runs all four data types (Embed_Feat_Var, PC_Feat_Var, MolFormer_Feat_Var, ChemBERTa_Feat_Var)
# in a single execution — the loop is handled inside tabpfn_fimmaml_model.py
python3 tabpfn_fimmaml_model.py
