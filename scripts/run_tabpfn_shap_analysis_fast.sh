#!/bin/bash -l
#SBATCH -J tabpfn_shap_fast
#SBATCH -o out_tabpfn_shap_analysis_fast.log
#SBATCH -e out_tabpfn_shap_analysis_fast.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH --mem=120000
#SBATCH --cpus-per-task=64
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH -x crirdchpxd005
#SBATCH -w crirdchpxd001
#SBATCH --time=12:00:00

# ─── Resource justification ──────────────────────────────────────────────────
#
#  GPU  1× H200 (141 GB HBM3e)
#    TabPFNExplainer calls model.fit()+predict() for each coalition — all GPU.
#
#  Time  12 h  (was 48 h)
#    TabPFNExplainer (remove-and-contextualize) vs MarginalImputer:
#      - Context: 200 rows (not full train set) → faster per-coalition eval
#      - Budget: 128 coalitions/sample (was 512)
#      - Samples: 100 train + 300 test = 400 total (was 2000)
#    Estimated wall time: 3–7 h on H200; 12 h gives a safe buffer.
#
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

python3 -c "import torch; print(f'PyTorch {torch.__version__}  CUDA available: {torch.cuda.is_available()}  device_count: {torch.cuda.device_count()}')"

echo "============================================================"
echo "  TabPFN Fast SHAP (TabPFNExplainer) + CI Analysis"
echo "  Start  : $(date)"
echo "  Node   : $(hostname)"
echo "  GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "  CPUs   : ${SLURM_CPUS_PER_TASK}"
echo "  RAM MB : ${SLURM_MEM_PER_NODE}"
echo "============================================================"

cd /export/cse/rmall/Raghvendra/PT-AML2.0/scripts

export PYTHONUNBUFFERED=1

python3 tabpfn_shap_analysis_fast.py

EXIT_CODE=$?
echo "============================================================"
echo "  End    : $(date)"
echo "  Status : $EXIT_CODE"
echo "============================================================"
exit $EXIT_CODE
