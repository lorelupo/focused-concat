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
# /usr/local/bin/dcgmi-slurm-start.sh

# nettoyage des modules charges en interactif et herites par defaut
module purge
 
# chargement des modules
# module load ...

# echo des commandes lancées
set -x

cd $HOME/dev/fairseq

# fix anaconda broken dependencies to use tensorboard
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/lib/

# Read arguments
for argument in "$@" 
do
    key=$(echo $argument | cut -f1 -d=)
    value=$(echo $argument | cut -f2 -d=)
    # Use string manipulation to set variable names according to convention   
    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        v="${v/-/_}"
        declare $v="${value}" 
   fi
done
if [ -n "$cuda" ]; then cuda=$cuda ; else cuda=0 ; fi

basename=transfo_lr
lang=en-ru
corpus=voita_opensubs
lr=1e-3
seed=0

for seed in 0 12 345 6789 876 54; do bash sh/run/en-ru/voita_opensubs/context_aware/concat.sh --t=results --save_dir=standard/$basename$lr/seed$seed --mode=slide_n2n --opt=num-sent --val=4 --scored_checkpoint=avg_closest --cuda=$cuda ; done
# for lr in 5e-4 7e-4 9e-4 1e-3 2e-3; do bash sh/run/en-ru/voita_opensubs/context_aware/concat.sh --t=results --save_dir=standard/$basename$lr/seed$seed --mode=slide_n2n --opt=num-sent --val=4 --scored_checkpoint=avg_closest --cuda=$cuda ; done

for seed in 6789 876 54
do   
    bash sh/run/$lang/$corpus/context_agnostic/transfo_base.sh \
    --t=train \
    --dropout=0.3 \
    --save_dir=checkpoints/$lang/$corpus/context_aware/standard/$basename$lr/seed$seed \
    --data_dir=data/$lang/data-bin/$corpus/context_aware/standard \
    --lr=$lr \
    --max_tokens=32000 \
    --update_freq=1 \
    --patience=12 \
    --seed=$seed
       
    bash sh/run/$lang/$corpus/context_agnostic/transfo_base.sh \
    --t=average \
    --save_dir=checkpoints/$lang/$corpus/context_aware/standard/$basename$lr/seed$seed \
    --scored_checkpoint=avg_closest \
    --seed=$seed

    bash sh/run/$lang/$corpus/context_agnostic/transfo_base.sh \
    --t=test-suites \
    --save_dir=checkpoints/$lang/$corpus/context_aware/standard/$basename$lr/seed$seed\
    --data_dir=data/$lang/data-bin/$corpus/context_aware/standard \
    --scored_checkpoint=avg_closest --cuda=$cuda \
    --seed=$seed
done

for lr in 5e-4 7e-4 9e-4 1e-3 2e-3; do bash sh/run/en-ru/voita_opensubs/context_aware/concat.sh --t=results --save_dir=standard/$basename$lr/seed$seed --mode=slide_n2n --opt=num-sent --val=4 --scored_checkpoint=avg_closest --cuda=$cuda ; done


# Deactivate data collection about GPU energy consumption
# /usr/local/bin/dcgmi-slurm-stop.sh