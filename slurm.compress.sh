#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --partition=batch
#SBATCH -J intml_8gpu
#SBATCH -o /ibex/scratch/hoc0a/imagenet18_collective_scheduling_compress/logs/compress_%J.%N.seed.%A_%a.gpu.out
#SBATCH --time=16:00:00
#SBATCH --mem=0G
#SBATCH --gres=gpu:v100:8

#module load singularity/3.5
#module load openmpi/4.0.1/gnu6.4.0_cuda10.1.105
#run the application:
#wandb login 9b716eb863fa8d3439dd39d203b9539d53c9eb2e
source ~/.bashrc
set | grep SLURM | while read line; do echo "# $line"; done
#mpirun --map-by node -np 1 --display-map --tag-output nvidia-smi -L

nvidia-smi -L

#mpirun -np 16 -x NCCL_DEBUG="INFO" -x NCCL_TREE_THRESHOLD="42949672960" -x OMP_NUM_THREADS="1" -x PYTORCH_USE_SPAWN="False" -x WANDB_API_KEY="9b716eb863fa8d3439dd39d203b9539d53c9eb2e"   --mca btl tcp,self --mca btl_tcp_if_exclude lo -map-by slot -bind-to none -nooversubscribe --output-filename /ibex/scratch/hoc0a/imagenet18_logs/intml_2nodes/mpi_logs -wdir /ibex/scratch/hoc0a/imagenet18_logs/intml_2nodes \
#singularity exec --nv -B /local/scratch,/ibex/scratch/hoc0a,/home/hoc0a /ibex/scratch/hoc0a/horovod_intml.simg \
#python /ibex/scratch/hoc0a/training_scripts/imagenet18/training/train_imagenet_nv.py /ibex/scratch/hoc0a/data/imagenet18/data/imagenet --logdir ./test_imagenet18

#python /ibex/scratch/hoc0a/training_scripts/imagenet18/training/train_imagenet_nv.py /ibex/scratch/hoc0a/data/imagenet18/data/imagenet --logdir /ibex/scratch/hoc0a/imagenet18_logs/intml_2nodes --name intml-16gpus --init-bn0 --no-bn-wd --phases gANdcQAofXEBKFgCAAAAZXBxAksAWAIAAABzenEDS4BYAgAAAGJzcQRNAAFYBgAAAHRybmRpcnEFWAcAAAAtc3ovMTYwcQZ1fXEHKGgCSwBLBoZxCFgCAAAAbHJxCUc/+AAAAAAAAEdACAAAAAAAAIZxCnV9cQsoaAJLBmgDS4BoBE0AAlgHAAAAa2VlcF9kbHEMiHV9cQ0oaAJLBmgJR0AIAAAAAAAAdX1xDihoAksLSw2GcQ9oCUdACAAAAAAAAEc/+AAAAAAAAIZxEHV9cREoaAJLDWgDS+BoBEvgaAVYBwAAAC1zei8zNTJxElgJAAAAbWluX3NjYWxlcRNHP7ZFocrAgxJ1fXEUKGgCSw1oCUc/9QAAAAAAAHV9cRUoaAJLEEsXhnEWaAlHP/UAAAAAAABHP8DMzMzMzM2GcRd1fXEYKGgCSxdLHIZxGWgJRz/AzMzMzMzNRz+K4UeuFHrhhnEadX1xGyhoAkscaANNIAFoBEuAaBNHP+AAAAAAAABYCAAAAHJlY3RfdmFscRyIdX1xHShoAkscSx6GcR5oCUc/frhR64UeuEc/SJN0vGp++oZxH3VlLg== --workers=6 --log_all_workers=1
export SEED=${SLURM_ARRAY_TASK_ID}
export CNAT_COMPRESS=1
conda activate /ibex/scratch/hoc0a/imagenet18_collective_scheduling_compress/env
LOG_DIR=/ibex/scratch/hoc0a/imagenet18_collective_scheduling_compress/logs/compress_${SLURM_JOB_ID}_SEED_${SLURM_ARRAY_TASK_ID}
mkdir -p $LOG_DIR
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 /ibex/scratch/hoc0a/imagenet18_collective_scheduling_compress/training/train_imagenet_nv.py /ibex/scratch/hoc0a/data/imagenet --logdir $LOG_DIR --distributed --init-bn0 --no-bn-wd --phases "[{'ep': 0, 'sz': 128, 'bs': 512, 'trndir': '-sz/160'}, {'ep': (0, 7), 'lr': (1.0, 2.0)}, {'ep': (7, 13), 'lr': (2.0, 0.25)}, {'ep': 13, 'sz': 224, 'bs': 224, 'trndir': '-sz/320', 'min_scale': 0.087}, {'ep': (13, 22), 'lr': (0.4375, 0.043750000000000004)}, {'ep': (22, 25), 'lr': (0.043750000000000004, 0.004375)}, {'ep': 25, 'sz': 288, 'bs': 128, 'min_scale': 0.5, 'rect_val': True}, {'ep': (25, 28), 'lr': (0.0025, 0.00025)}]"
