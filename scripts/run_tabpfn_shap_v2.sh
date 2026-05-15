#!/bin/bash -l
#SBATCH -J tabpfn_shap_v3
#SBATCH -o out_tabpfn_shap_v2.log
#SBATCH -e out_tabpfn_shap_v2.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH --mem=120000
#SBATCH --cpus-per-task=36
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -x crirdchpxd005
#SBATCH -w crirdchpxd003
#SBATCH -t 36:00:00

# ─── Resource justification ──────────────────────────────────────────────────
#
#  Prerequisites: tabpfn_predict_embed.py must have been run first.
#  The SHAP step reads test_predictions_with_CI.csv and
#  embedding_attention_importance.csv from OUT_DIR.
#
#  GPU  1× H200 (141 GB HBM3e)
#    Each coalition evaluation re-contextualises TabPFN on the full 33 K
#    training context — dominated by the transformer attention pass.
#    Estimated ~3–10 s/coalition on H200 with tabpfn 8.0.2.
#
#  Timing estimate (SHAP_BUDGET = 8192):
#    8192 coalitions × ~5 s ≈ 11 h per sample.
#    For 20 train + 100 test = 120 samples → ~1320 h total.
#
#    Recommended approach: reduce SHAP_TEST_SAMPLES / SHAP_TRAIN_SAMPLES and
#    submit multiple jobs using the crash-safe caching (each job resumes from
#    the latest checkpoint).  A single 36 h job can explain ~3 samples at
#    budget=8192.  Alternatively, lower SHAP_BUDGET to 2048 for noisier but
#    faster estimates (~2.7 h/sample → ~30 samples in 36 h).
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

python3 -c "import torch, tabpfn, shapiq; print(f'PyTorch {torch.__version__}  tabpfn {tabpfn.__version__}  shapiq {shapiq.__version__}  CUDA: {torch.cuda.is_available()}')"

echo "============================================================"
echo "  TabPFN SHAP v3 (Steps 5–6)"
echo "  Start  : $(date)"
echo "  Node   : $(hostname)"
echo "  GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "  CPUs   : ${SLURM_CPUS_PER_TASK}"
echo "  RAM MB : ${SLURM_MEM_PER_NODE}"
echo "============================================================"

cd /export/cse/rmall/Raghvendra/PT-AML2.0/scripts

export PYTHONUNBUFFERED=1

python3 tabpfn_shap_v2.py

EXIT_CODE=$?
echo "============================================================"
echo "  End    : $(date)"
echo "  Status : $EXIT_CODE"
echo "============================================================"
exit $EXIT_CODE
