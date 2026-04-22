#!/bin/bash -l
#SBATCH -J tabpfn_ablation_cv
#SBATCH -o out_cv_tabpfn_ablation.log
#SBATCH -e out_cv_tabpfn_ablation.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:2
#SBATCH --mem=80000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -x crirdchpxd005
#SBATCH -w crirdchpxd003

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

python3 cv_tabpfn_evaluation.py

