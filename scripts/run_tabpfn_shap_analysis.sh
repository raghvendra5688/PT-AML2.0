#!/bin/bash -l
#SBATCH -J tabpfn_shap
#SBATCH -o out_tabpfn_shap_analysis.log
#SBATCH -e out_tabpfn_shap_analysis.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH --mem=120000
#SBATCH --cpus-per-task=64
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -x crirdchpxd005
#SBATCH -w crirdchpxd001

# ─── Resource justification ──────────────────────────────────────────────────
#
#  GPU  1× H200 (141 GB HBM3e)
#    TabPFN is a transformer; every model call in SHAP uses the GPU.
#    The 3.5 GB model + internal KV-cache fits comfortably; GPU forward passes
#    are 10–50× faster than CPU for the batch sizes KernelExplainer uses.
#
#  RAM  120 GB
#    Unpickled model in RAM : ~12 GB
#    SHAP background matrix : 1000 samples × ~500 features ≈ 4 MB
#    SHAP coalition buffers  : up to ~20 GB peak per worker
#    TabPFN embeddings       : n_estimators × n_samples × embed_dim (~2–5 GB)
#    Pandas + numpy working  : ~10 GB
#    120 GB provides a comfortable buffer for all of the above.
#
#  CPUs  64
#    KernelExplainer solves a weighted least-squares problem on CPU between
#    each batch of GPU model calls; 64 threads speed up these numpy/scipy steps.
#    Pandas preprocessing and pickle I/O also benefit from extra threads.
#
#  Time  48 h
#    Breakdown of estimated wall time on one H200:
#      Step 1  load model + scaler              :  ~5 min
#      Step 2  load + preprocess train/test     :  ~2 min
#      Step 3  predictions + CI (quantiles)     :  ~10 min
#      Step 4  SHAP – build KernelExplainer     :  ~2 min
#              SHAP – 5000 train samples        :  ~8–12 h  ← dominant cost
#                  (5000 samples × 512 coalitions × batch fwd pass on GPU)
#              SHAP – 5000 test  samples        :  ~8–12 h
#              Single-sample waterfall plots    :  ~3 min
#      Step 5  embedding importance (fwd pass)  :  ~15 min
#      Step 6  summary plots / CSVs             :  ~2 min
#    Total estimated: ~5.5–6 h → 48 h gives a safe 8× buffer.
#    (SHAP_TEST_SAMPLES=1000 reduced from 5000; uses shapiq KernelSHAP)
#
#    To run faster further, reduce SHAP_TEST_SAMPLES or SHAP_BUDGET in the script:
#      SHAP_TEST_SAMPLES = 500   → ~3 h total
#      SHAP_BUDGET       = 256   → halves coalition cost at slight accuracy cost
# ────────────────────────────────────────────────────────────────────────────

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

# Verify that PyTorch can see the GPU
python3 -c "import torch; print(f'PyTorch {torch.__version__}  CUDA available: {torch.cuda.is_available()}  device_count: {torch.cuda.device_count()}')"

echo "============================================================"
echo "  TabPFN SHAP + Attention Importance + CI Analysis"
echo "  Start  : $(date)"
echo "  Node   : $(hostname)"
echo "  GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "  CPUs   : ${SLURM_CPUS_PER_TASK}"
echo "  RAM MB : ${SLURM_MEM_PER_NODE}"
echo "============================================================"

cd /export/cse/rmall/Raghvendra/PT-AML2.0/scripts

# Force Python stdout/stderr to be line-buffered so all progress prints
# appear in the SLURM log even if the process is killed by a signal.
export PYTHONUNBUFFERED=1

python3 tabpfn_best_model_shap_analysis.py

EXIT_CODE=$?
echo "============================================================"
echo "  End    : $(date)"
echo "  Status : $EXIT_CODE"
echo "============================================================"
exit $EXIT_CODE
