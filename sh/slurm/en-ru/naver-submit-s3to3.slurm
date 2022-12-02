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

basename=s3to3_lr
val=3

# for lr in 1e-4 3e-4 5e-4 7e-4 9e-4 1e-3 2e-3 4e-3; do bash sh/run/en-ru/voita_opensubs/context_aware/concat.sh --t=results --save_dir=standard/$basename$lr --mode=slide_n2n --opt=num-sent --val=4 --scored_checkpoint=avg_closest --cuda=$cuda ; done

for lr in 7e-4 9e-4 1e-3 2e-3 
do   
    # bash sh/run/en-ru/voita_opensubs/context_aware/concat.sh \
    # --t=train \
    # --arch=concat_voita_fairseq \
    # --save_dir=standard/$basename$lr \
    # --lr=$lr \
    # --max_tokens=32000 \
    # --update_freq=1 \
    # --mode=slide_n2n --opt=num-sent --val=$val \
    # --patience=12
       
    # bash sh/run/en-ru/voita_opensubs/context_aware/concat.sh \
    # --save_dir=standard/$basename$lr \
    # --t=average \
    # --mode=slide_n2n --opt=num-sent --val=$val

    # bash sh/run/en-ru/voita_opensubs/context_aware/concat.sh --t=results --save_dir=standard/$basename$lr --mode=slide_n2n --opt=num-sent --val=4 --scored_checkpoint=avg_closest --cuda=$cuda
    bash sh/run/en-ru/voita_opensubs/context_aware/concat.sh \
    --t=test-suites \
    --save_dir=standard/$basename$lr \
    --mode=slide_n2n --opt=num-sent --val=$val \
    --scored_checkpoint=avg_closest --cuda=$cuda
done

# Deactivate data collection about GPU energy consumption
# /usr/local/bin/dcgmi-slurm-stop.sh