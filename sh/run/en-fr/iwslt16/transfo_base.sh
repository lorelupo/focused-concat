#!/bin/bash
#
# "Train" or "test" a standard Trasformer model
# to reproduce Wang et al., 2019' results on IWSLT16 fr-en.

# Read script arguments and assign them to variables
# sh/run_transformer_iwslt16_fr.sh --t=train --src=en --tgt=fr --save_dir=en2fr_iwslt16_wmt14_han/k0 --data_dir=wmt14
for argument in "$@" 
do

    key=$(echo $argument | cut -f1 -d=)
    value=$(echo $argument | cut -f2 -d=)   
    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        declare $v="${value}" 
   fi
done

# Set variables
if [ -n "$cuda" ] ; then export CUDA_VISIBLE_DEVICES=$cuda ; fi
if [ -n "$seed" ]; then seed=$seed ; else seed=0 ; fi
if [ -n "$data_dir" ]; then data_dir=data/data-bin/iwslt16.dnmt.fr-en/$data_dir ; else data_dir=data/data-bin/iwslt16.dnmt.fr-en/standard ; fi
if [[ $save_dir != "checkpoints/"* ]]; then save_dir=checkpoints/$save_dir; fi
src=$src
tgt=$tgt

if [ $t = "train" ]
then
    # train
    if [ $t != "results" ]
then
    mkdir -p $save_dir/logs
fi
    fairseq-train $data_dir \
    --save-dir $save_dir \
    --seed $seed \
    --source-lang $src \
    --target-lang $tgt \
    --arch transformer_iwslt_fr_en \
    --optimizer adam \
    --lr-scheduler inverse_sqrt \
    --lr 1e-4 \
    --warmup-init-lr 1e-7 \
    --warmup-updates 4000\
    --patience 5 \
    --max-tokens 2048 \
    --log-format json \
    | tee -a $save_dir/logs/train.log
elif [ $t = "test" ]
then
    fairseq-generate $data_dir \
    --task translation \
    --source-lang $src \
    --target-lang $tgt \
    --path $save_dir/checkpoint.avg5.pt \
    --batch-size 64 \
    --remove-bpe \
    --beam 4 \
    --lenpen 1 \
    --temperature 1.3 \
    --num-workers 8 \
    | tee $save_dir/logs/test.log
elif [ $t = "score" ]
then
    grep ^S $save_dir/logs/test.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- > $save_dir/logs/gen.out.src
    grep ^T $save_dir/logs/test.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- > $save_dir/logs/gen.out.ref
    grep ^H $save_dir/logs/test.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- > $save_dir/logs/gen.out.sys
    fairseq-score \
    --sentence-bleu \
    --sys $save_dir/logs/gen.out.sys \
    --ref $save_dir/logs/gen.out.ref \
    | tee $save_dir/logs/score.log
elif [ $t = "average" ]
then
    avg_n_epochs=5
    avg_checkpoint=checkpoint.avg$avg_n_epochs.pt
    python scripts/average_checkpoints.py \
        --inputs $save_dir \
        --num-epoch-checkpoints $avg_n_epochs \
        --checkpoint-upper-bound=115 \
        --output $save_dir/$avg_checkpoint
else
    echo "Argument is not valid. Type 'train' or 'test'."
fi

