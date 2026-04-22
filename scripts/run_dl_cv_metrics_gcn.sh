#!/bin/bash -l
#SBATCH -J dl_cv_metrics_gcn
#SBATCH -o logs/out_dl_cv_metrics_gcn.log
#SBATCH -e logs/out_dl_cv_metrics_gcn.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH --mem=80000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -x crirdchpxd005
#SBATCH -w crirdchpxd006

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

micromamba activate BeatAML2.0

DATA_TYPES=("Embed_Feat_Var" "ChemBERTa_Feat_Var" "MolFormer_Feat_Var")
STRATIFY_TYPES=("random" "inhibitor" "dbgap_rnaseq_sample")

for data in "${DATA_TYPES[@]}"; do
    for strat in "${STRATIFY_TYPES[@]}"; do
        echo "=============================="
        echo "model=gcn  data=${data}  stratify=${strat}"
        echo "=============================="
        python3 dl_cv_metrics.py \
            --model_type  gcn \
            --data_type   "${data}" \
            --stratify_by "${strat}"
    done
done
