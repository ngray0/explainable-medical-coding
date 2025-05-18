#!/bin/bash
#──────────────── Grid‑Engine options ────────────────
#$ -N modernbert_icd                       # job name
#$ -cwd                                    # run from the directory you qsub
#$ -q gpu                                  # GPU queue on Eddie
#$ -l gpu=1                                # one A100 (scheduler sets CUDA_VISIBLE_DEVICES)
#$ -pe smp 4                               # 4 CPU threads for dataloader
#$ -l h_vmem=90G                           # host memory (≈ GPU mem + a bit)
#$ -l h_rt=48:00:00                        # max wall‑time
#$ -o logs/$JOB_NAME.$JOB_ID.out           # stdout
#$ -e logs/$JOB_NAME.$JOB_ID.err           # stderr
# optional:
# #$ -P your_project                       # compute project
# #$ -m beas -M you@ed.ac.uk               # email on begin/end/abort/suspend
#──────────────── shell safety ────────────
set -euo pipefail
#──────────────── module + conda ──────────
. /etc/profile.d/modules.sh
module load cuda/12.2                      # or whatever Eddie provides
# initialise cluster‑wide conda
source /exports/applications/apps/RL9/anaconda/2024.02/etc/profile.d/conda.sh
conda activate /exports/eddie/scratch/${USER}/miniconda/envs/coding
#──────────────── scratch‑based caches ────
export SCRATCH=/exports/eddie/scratch/${USER}
export WANDB_DIR=$SCRATCH/wandb
export WANDB_CACHE_DIR=$WANDB_DIR/cache
export WANDB_ARTIFACT_DIR=$WANDB_DIR/artifacts
export HF_HOME=$SCRATCH/hf_cache
export TMPDIR=$SCRATCH/tmp
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" "$HF_HOME" "$TMPDIR"
# avoid W&B artefact quota crashes
export WANDB_DISABLE_JOB_CREATION=true
#──────────────── performance knobs ───────
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
#──────────────── run training ────────────
poetry run python train_plm.py \
  experiment=mdace_icd9_code/plm_icd \
  model=plm_icd_modernbert \
  data=mimiciv_icd10 \
  gpu=0 \                                   # 0 == the single GPU assigned
  dataloader.max_batch_size=1
