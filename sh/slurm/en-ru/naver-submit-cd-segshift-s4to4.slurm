#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu_32g"
#SBATCH --cpus-per-task=24
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lorenzo.lupo@univ-grenoble-alpes.fr
#SBATCH --output=/scratch/1/user/llupo/fairseq/checkpoints/slurm/%j.out

# nettoyage des modules charges en interactif et herites par defaut
module purge
 
# chargement des modules
# module load ...

# echo des commandes lancées
set -x

# Activate data collection about GPU energy consumption
# /usr/local/bin/dcgmi-slurm-start.sh

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

basename=cd_segshift8_s4to4_lr
cd=0.01
val=4
lr=9e-4
seed=0

# for seed in 0 12 345 6789 876 54; do bash sh/run/en-ru/voita_opensubs/context_aware/concat.sh --t=results --save_dir=standard/$basename$lr/${cd}cd/seed$seed --mode=slide_n2n --opt=num-sent --val=4 --scored_checkpoint=avg_closest --cuda=$cuda ; done
# for cd in 0.01 0.1 0.3 0.5 0.7 0.9; do bash sh/run/en-ru/voita_opensubs/context_aware/concat.sh --t=results --save_dir=standard/$basename$lr/${cd}cd/seed$seed --mode=slide_n2n --opt=num-sent --val=$val --scored_checkpoint=avg_closest --cuda=$cuda ; done
# for lr in 7e-4 9e-4 1e-3 2e-3; do bash sh/run/en-ru/voita_opensubs/context_aware/concat.sh --t=results --save_dir=standard/$basename$lr/${cd}cd/seed$seed --mode=slide_n2n --opt=num-sent --val=$val --scored_checkpoint=avg_closest --cuda=$cuda ; done

for seed in 12 345 6789 876 54
do   
    bash sh/run/en-ru/voita_opensubs/context_aware/concat.sh \
    --t=train \
    --arch=concat_segshift \
    --position-shift=8 \
    --need-seg_label=True \
    --context_discount=$cd \
    --mode=slide_n2n --opt=num-sent --val=$val \
    --save_dir=standard/$basename$lr/${cd}cd/seed$seed \
    --lr=$lr \
    --max_tokens=32000 \
    --update_freq=1 \
    --patience=12 \
    --seed=$seed
       
    bash sh/run/en-ru/voita_opensubs/context_aware/concat.sh \
    --save_dir=standard/$basename$lr/${cd}cd/seed$seed \
    --t=average \
    --mode=slide_n2n --opt=num-sent --val=$val \
    --seed=$seed

    # bash sh/run/en-ru/voita_opensubs/context_aware/concat.sh --t=results --save_dir=standard/$basename$lr/${cd}cd/seed$seed --mode=slide_n2n --opt=num-sent --val=$val --scored_checkpoint=avg_closest --cuda=$cuda
    bash sh/run/en-ru/voita_opensubs/context_aware/concat.sh \
    --t=test-suites \
    --save_dir=standard/$basename$lr/${cd}cd/seed$seed \
    --mode=slide_n2n --opt=num-sent --val=$val \
    --need_seg_label=True \
    --scored_checkpoint=avg_closest --cuda=$cuda \
    --seed=$seed
done

# Deactivate data collection about GPU energy consumption
# /usr/local/bin/dcgmi-slurm-stop.sh