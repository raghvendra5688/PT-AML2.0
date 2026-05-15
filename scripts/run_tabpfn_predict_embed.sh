#!/bin/bash -l
#SBATCH -J tabpfn_pred_embed
#SBATCH -o out_tabpfn_predict_embed.log
#SBATCH -e out_tabpfn_predict_embed.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH --mem=120000
#SBATCH --cpus-per-task=36
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -x crirdchpxd005
#SBATCH -w crirdchpxd003

# ─── Resource justification ──────────────────────────────────────────────────
#
#  Steps 1–4 only (no SHAP).
#
#  GPU  1× H200
#    Step 3: predict train (~33K) + test (~18K) + CI quantiles (~18K) — all GPU.
#    Step 4: get_embeddings on full train (~210 MB) + full test (~117 MB) — GPU.
#
#  Time  ~2 h
#    Step 3 predict calls: ~370 s each × 3 calls ≈ 30 min total.
#    Step 4 embeddings:    ~5–10 min each × 2 splits.
#    Total with buffer: 2 h.
#
# ─────────────────────────────────────────────────────────────────────────────

module load cuda12.6/toolkit/12.6.2
python --version
nvidia-smi
nvcc --version

export MAMBA_EXE='/export/home/rmall/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/export/home/rmall/micromamba'
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"
fi
unset __mamba_setup

micromamba activate BeatAML2.0

python3 -c "import torch; print(f'PyTorch {torch.__version__}  CUDA available: {torch.cuda.is_available()}  device_count: {torch.cuda.device_count()}')"

echo "============================================================"
echo "  TabPFN Predict + Embed (Steps 1–4)"
echo "  Start  : $(date)"
echo "  Node   : $(hostname)"
echo "  GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "  CPUs   : ${SLURM_CPUS_PER_TASK}"
echo "  RAM MB : ${SLURM_MEM_PER_NODE}"
echo "============================================================"

cd /export/cse/rmall/Raghvendra/PT-AML2.0/scripts

export PYTHONUNBUFFERED=1

python3 tabpfn_predict_embed.py

EXIT_CODE=$?
echo "============================================================"
echo "  End    : $(date)"
echo "  Status : $EXIT_CODE"
echo "============================================================"
exit $EXIT_CODE
