#!/bin/bash
#
# "Train" or "test" a Hierarchical Attention Network Transformer model
# (Miculicich et al., 2018').

# Set cuda device if (second) argument was given
if [ -n "$2" ] 
then 
    echo "setting CUDA_VISIBLE_DEVICES=$2"
    export CUDA_VISIBLE_DEVICES=$2
fi

what=$1
data_dir=data/data-bin/dummy.tokenized
avg_n_epochs=5
avg_checkpoint=checkpoint.avg$avg_n_epochs.pt

if [ $what = "train" ]
then
    save_dir=checkpoints/dummy_han
    if [ $t != "results" ]
then
    mkdir -p $save_dir/logs
fi
    fairseq-train $data_dir \
    --task translation_han \
    --arch han_transformer_test \
    --optimizer adam --lr-scheduler inverse_sqrt --lr 1e-4 \
    --warmup-init-lr 1e-7 --warmup-updates 4000\
    --patience 2 \
    --max-sentences 50 \
    --max-source-positions 1000 \
    --max-target-positions 1000 \
    --save-dir $save_dir \
    --log-format json \
    --seed 0 \
    | tee -a $save_dir/logs/train.log
elif [ $what = "train_han_from_pretrained" ]
then
    save_dir=checkpoints/dummy_han_from_pretrained
    if [ $t != "results" ]
then
    mkdir -p $save_dir/logs
fi
    fairseq-train $data_dir \
    --task translation_han \
    --arch han_transformer \
    --pretrained-transformer-checkpoint checkpoints/dummy_tr/checkpoint_best.pt \
    --optimizer adam --lr-scheduler inverse_sqrt --lr 1e-4 \
    --warmup-init-lr 1e-7 --warmup-updates 4000\
    --patience 2 \
    --max-tokens 1000 \
    --max-source-positions 1000 \
    --max-target-positions 1000 \
    --save-dir $save_dir \
    --log-format json \
    --seed 0 \
    | tee -a $save_dir/logs/train.log
elif [ $what = "train_tr_base" ]
then
    save_dir=checkpoints/dummy_tr
    if [ $t != "results" ]
then
    mkdir -p $save_dir/logs
fi
    fairseq-train $data_dir \
    --task translation \
    --arch transformer \
    --optimizer adam --lr-scheduler inverse_sqrt --lr 1e-4 \
    --warmup-init-lr 1e-7 --warmup-updates 4000\
    --patience 2 \
    --max-tokens 1000 \
    --max-source-positions 1000 \
    --max-target-positions 1000 \
    --save-dir $save_dir \
    --log-format json \
    --seed 0 \
    | tee -a $save_dir/logs/train.log
elif [ $what = "average" ]
then
    python scripts/average_checkpoints.py \
        --inputs $save_dir \
        --num-epoch-checkpoints $avg_n_epochs \
        --checkpoint-upper-bound=115 \
        --output $save_dir/$avg_checkpoint
elif [ $what = "test" ]
then
    fairseq-generate $data_dir \
    --task translation_han \
    --path $save_dir/checkpoint_best.pt \
    --results-path $save_dir/logs/test \
    --quiet \
    --batch-size 64 \
    --remove-bpe \
    --beam 4 \
    --lenpen 1 \
    --temperature 1.3 \
    | tee $save_dir/logs/test.log
else
    echo "Argument is not valid. Type 'train' or 'test'."
fi