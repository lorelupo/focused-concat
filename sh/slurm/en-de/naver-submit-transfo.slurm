#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu_32g"
#SBATCH --cpus-per-task=24
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lorenzo.lupo@univ-grenoble-alpes.fr
#SBATCH --output=/scratch/1/user/llupo/fairseq/checkpoints/slurm/%j.out

# Activate data collection about GPU energy consumption
/usr/local/bin/dcgmi-slurm-start.sh

# nettoyage des modules charges en interactif et herites par defaut
module purge
 
# chargement des modules
# module load ...

# echo des commandes lancées
set -x

cd $HOME/dev/fairseq

# fix anaconda broken dependencies to use tensorboard
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/lib/

basename=fromscratch/transfo_lr

# for lr in 3e-3 5e-3; do bash sh/run/en-de/iwslt17/transfo_base.sh --t=results --save_dir=standard/$basename$lr --mode=slide_n2n --opt=num-sent --val=4 --scored_checkpoint=avg_closest --cuda=$cuda ; done

for lr in 3e-3 5e-3
do   
    bash sh/run/en-de/iwslt17/transfo_base.sh \
    --t=train \
    --arch=transformer_vaswani_wmt_en_fr \
    --dropout=0.3 \
    --save_dir=standard/$basename$lr \
    --lr=$lr \
    --max_tokens=16000 \
    --update_freq=1 \
    --patience=15
    # --t=finetune \
    # --pretrained=checkpoints/en-de/wmt17/transfo_base/checkpoint.avg_last10.pt \
  
    bash sh/run/en-de/iwslt17/transfo_base.sh \
    --save_dir=standard/$basename$lr \
    --t=average \

    bash sh/run/en-de/iwslt17/transfo_base.sh \
    --t=test-suites \
    --save_dir=standard/$basename$lr \
    --scored_checkpoint=avg_closest --cuda=$cuda

    # bash sh/run/en-de/iwslt17/transfo_base.sh --t=search-lenpen --save_dir=standard/$basename$lr --scored_checkpoint=avg_closest --cuda=$cuda
    # bash sh/run/en-de/iwslt17/transfo_base.sh \
    # --t=test-suites \
    # --save_dir=standard/$basename$lr \
    # --scored_checkpoint=best --cuda=$cuda
done

# Deactivate data collection about GPU energy consumption
# /usr/local/bin/dcgmi-slurm-stop.sh