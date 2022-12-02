#!/bin/bash
#
# Train, test and score a Hierarchical Attention Network Transformer model
# (Miculicich et al., 2018').

# Read script arguments and assign them to variables
# sh/run_han_transformer_iwslt16_fr.sh --t=test --save_dir=checkpoints/en_fr_iwslt16_han/k1 --k=1 --src=en --tgt=fr --cuda=0 --pretrained=checkpoints/en_fr_iwslt16_han/k0/checkpoint_best.pt
for argument in "$@"
do
    key=$(echo $argument | cut -f1 -d=)
    value=$(echo $argument | cut -f2 -d=)   
    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        declare $v="${value}" 
   fi
done

# Set cuda device if argument was given
if [ -n "$cuda" ] 
then 
    echo "setting CUDA_VISIBLE_DEVICES=$cuda"
    export CUDA_VISIBLE_DEVICES=$cuda
fi

# Set variables
if [ -n "$cuda" ] ; then export CUDA_VISIBLE_DEVICES=$cuda ; fi
if [ -n "$seed" ] ; then seed=$seed ; else seed=0 ; fi
if [ -n "$data_dir" ] ; then data_dir=data/data-bin/iwslt16.dnmt.fr-en/$data_dir ; else data_dir=data/data-bin/iwslt16.dnmt.fr-en/standard ; fi
if [ -n "$pretrained" ]; then pretrained=checkpoints/$pretrained ; else pretrained='' ; fi
if [[ $save_dir != "checkpoints/"* ]]; then save_dir=checkpoints/$save_dir; fi
src=$src
tgt=$tgt
num_workers=8
n_best_checkpoints=5

if [ $t = "train" ]
then
    # train
    if [ $t != "results" ]
then
    mkdir -p $save_dir/logs
fi
    fairseq-train $data_dir \
    --task translation_han \
    --source-lang $src \
    --target-lang $tgt \
    --n-context-sents $k \
    --pretrained-transformer-checkpoint $pretrained \
    --freeze-transfo-params \
    --arch han_transformer_iwslt_fr_en \
    --optimizer adam \
    --lr-scheduler inverse_sqrt \
    --lr 1e-4 \
    --warmup-init-lr 1e-7 \
    --warmup-updates 4000 \
    --patience 5 \
    --max-tokens 8192 \
    --keep-best-checkpoints 5 \
    --no-epoch-checkpoints \
    --log-format json \
    | tee $save_dir/logs/train.log
    # test
    fairseq-generate $data_dir \
    --task translation_han \
    --source-lang $src \
    --target-lang $tgt \
    --path $save_dir/checkpoint_best.pt \
    --batch-size 64 \
    --remove-bpe \
    --beam 4 \
    --lenpen 1 \
    --temperature 1.3 \
    | tee $save_dir/logs/test.log
    # score sents
    grep ^S $save_dir/logs/test.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- > $save_dir/logs/gen.out.src
    grep ^T $save_dir/logs/test.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- > $save_dir/logs/gen.out.ref
    grep ^H $save_dir/logs/test.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- > $save_dir/logs/gen.out.sys
    fairseq-score \
    --sentence-bleu \
    --sys $save_dir/logs/gen.out.sys \
    --ref $save_dir/logs/gen.out.ref \
    | tee $save_dir/logs/score.log
elif [ $t = "test" ]
then
    fairseq-generate $data_dir \
    --task translation_han \
    --source-lang $src \
    --target-lang $tgt \
    --path $save_dir/checkpoint_best.pt \
    --batch-size 64 \
    --remove-bpe \
    --beam 4 \
    --lenpen 1 \
    --temperature 1.3 \
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
    python scripts/average_checkpoints.py \
        --inputs $save_dir \
        --num-epoch-checkpoints $avg_n_epochs \
        --checkpoint-upper-bound=115 \
        --output $save_dir/$avg_checkpoint
else
    echo "Argument is not valid. Type 'train' or 'test'."
fi