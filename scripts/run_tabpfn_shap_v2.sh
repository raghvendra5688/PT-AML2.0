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

# ─── What this job does ──────────────────────────────────────────────────────
#
#  Step 2b — Train a fresh TabPFNRegressor with tabpfn 8.0.2 on the full
#             training set (33 K rows × 1561 features), save it.
#             The original saved model cannot be loaded under tabpfn 8.0.2
#             (module-path change: tabpfn.architectures.base.encoders gone).
#
#  Step 5   — SHAP (TabPFNExplainer, PermutationSamplingSV).
#             Full 33 K training context, all 1561 features.
#             Two-level crash-safe cache; re-submitting this job resumes
#             from the deepest checkpoint automatically.
#
#  Step 5b  — SHAP waterfall for notable test samples.
#
#  Step 6   — Combined SHAP + embedding-attention importance summary.
#             Reads embedding_attention_importance.csv from tabpfn_predict_embed.py;
#             gracefully degrades to SHAP-only ranking if that file is absent.
#
# ─── Timing estimates ────────────────────────────────────────────────────────
#
#  Step 2b  fit(): tabpfn 8.0.2, n_estimators=8, H200 → ~5–20 min.
#  Step 5   SHAP:  each coalition = one model.fit(33 K) + predict(1 sample).
#    Estimated ~3–8 s/coalition × 4096 budget ≈ 3.4–9 h per sample.
#    A single 36 h job can explain ~4–10 test samples at SHAP_BUDGET=4096.
#    To cover all 100 test + 20 train samples: ~10–25 sequential 36 h jobs,
#    each resuming from the checkpoint left by the previous one.
#    Alternative: lower SHAP_BUDGET to 1024 → ~4× faster, noisier estimates.
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

python3 -c "
import tabpfn, shapiq, torch
print(f'tabpfn  : {tabpfn.__version__}')
print(f'shapiq  : {shapiq.__version__}')
print(f'PyTorch : {torch.__version__}  CUDA: {torch.cuda.is_available()}')
"

echo "============================================================"
echo "  TabPFN 8.0.2 retrain + SHAP v3 (Steps 2b, 5, 5b, 6)"
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
