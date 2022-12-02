#!/bin/bash

# This script provides command-line statements for training and testing a
# Transformer model for the translation of a concatenated sequence of
# source and/or target sentences, via the "mode" and "val" arguments

# EXAMPLES OF USAGE
#
# DEBUGGING
# bash sh/run/en-de/iwslt17/concat.sh --t=train \
# --data-dir=data/en-de/data-bin/iwslt17/standard \
# --save-dir=checkpoints/en-de/debug \
# --max-tokens=100 --update-freq=1 \
# --mode=slide_n2n --opt=num-sent --val=3
#
# TRAINING
# bash sh/run/en-de/iwslt17/concat.sh --t=train \
# --data-dir=data/en-de/data-bin/iwslt17/standard \
# --save-dir=checkpoints/en-de/iwslt17/transfo-s2to2 \
# --pretrained=checkpoints/en-de/iwslt17/standard/k0/checkpoint_best.pt \
# --max-tokens=1000 --update-freq=8 \
# --mode=slide_n2n --opt=num-sent --val=3

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

# Global variables
if [ -n "$src" ]; then src=$src ; else src=en ; fi
if [ -n "$tgt" ]; then tgt=$tgt ; else tgt=de ; fi
if [ -n "$corpus" ]; then corpus=$corpus ; else corpus=iwslt17 ; fi
if [ -n "$lang" ]; then lang=$lang; else lang=$src-$tgt ; fi
if [ -n "$this_script" ]; then this_script=$this_script; else this_script=sh/run/$lang/$corpus/concat.sh ; fi

# Common
if [ -n "$cuda" ] ; then export CUDA_VISIBLE_DEVICES=$cuda ; fi
if [ -n "$seed" ]; then seed=$seed ; else seed=0 ; fi
if [ -n "$num_workers" ]; then num_workers=$num_workers ; else num_workers=8 ; fi
if [ -n "$ddp_backend" ]; then ddp_backend=$ddp_backend ; else ddp_backend=c10d ; fi

# Data
if [ -n "$data_dir" ]; then data_dir=$data_dir; else data_dir=data/$lang/data-bin/$corpus/standard ; fi
if [ -n "$max_tokens" ]; then max_tokens=$max_tokens ; else max_tokens=4000; fi
if [ -n "$update_freq" ]; then update_freq=$update_freq ; else update_freq=1; fi

# Model
if [ -n "$arch" ]; then arch=$arch ; else arch=transformer_vaswani_wmt_en_fr ; fi
if [ -n "$pse_segment_dim" ]; then pse_segment_dim=$pse_segment_dim ; else pse_segment_dim=0 ; fi
if [ -n "$position_shift" ]; then position_shift=$position_shift ; else position_shift=0 ; fi
if [ -n "$dropout" ]; then dropout=$dropout ; else dropout= ; fi
if [ -n "$activation_dropout" ]; then activation_dropout=$activation_dropout ; else activation_dropout= ; fi
if [ -n "$attention_dropout" ]; then attention_dropout=$attention_dropout ; else attention_dropout= ; fi

# Loss
if [ -n "$criterion" ]; then criterion=$criterion; else criterion=label_smoothed_cross_entropy; fi
if [ -n "$label_smoothing" ]; then label_smoothing=$label_smoothing; else label_smoothing=0.1; fi
if [ -n "$weight_decay" ]; then weight_decay=$weight_decay; else weight_decay=0.0; fi

# Optimization
if [ -n "$lr_scheduler" ]; then lr_scheduler=$lr_scheduler; else lr_scheduler=inverse_sqrt; fi
if [ -n "$lr" ]; then lr=$lr; else lr=1e-03; fi
if [ -n "$min_lr" ]; then min_lr=$min_lr ; else min_lr=1e-09 ; fi # stop training when the lr reaches this minimum (default -1.0)
if [ -n "$total_num_update" ]; then total_num_update=$total_num_update; else total_num_update=16000; fi # 16000 ups corresponds to 15 epoch with max_tokens=128K
if [ -n "$warmup_updates" ]; then warmup_updates=$warmup_updates ; else warmup_updates=4000 ; fi
if [ -n "$warmup_init_lr" ]; then warmup_init_lr=$warmup_init_lr ; else warmup_init_lr=1e-07 ; fi
if [ -n "$end_learning_rate" ]; then end_learning_rate=$end_learning_rate ; else end_learning_rate=1e-09 ; fi

# Checkpoints
if [[ $save_dir != "checkpoints/"* ]]; then save_dir=checkpoints/$lang/$corpus/$save_dir; fi
if [ -n "$ncheckpoints" ]; then ncheckpoints=$ncheckpoints ; else ncheckpoints=5 ; fi
if [ -n "$patience" ]; then patience=$patience ; else patience=12 ; fi
if [ -n "$max_update"]; then max_update=$max_update ; else max_update=0 ; fi
if [ -n "$keep_last_epochs" ]; then keep_last_epochs=$keep_last_epochs ; else keep_last_epochs=$patience ; fi
if [ -n "$keep_best_checkpoints" ]; then keep_best_checkpoints=$keep_best_checkpoints ; else keep_best_checkpoints=$ncheckpoints ; fi
if [ -n "$save_interval_updates" ]; then save_interval_updates=$save_interval_updates ; else save_interval_updates=0 ; fi
if [ -n "$keep_interval_updates" ]; then keep_interval_updates=$keep_interval_updates ; else keep_interval_updates=-1 ; fi
if [ -n "$scored_checkpoint" ]; then scored_checkpoint=$scored_checkpoint ; else scored_checkpoint=best ; fi
if [ $scored_checkpoint = 'best' ]
then
    checkpoint_path=$save_dir/checkpoint_best.pt
    checkpoint_prefix=best.
elif [ $scored_checkpoint = 'avg_last' ]
then
    checkpoint_path=$save_dir/checkpoint.avg_last$ncheckpoints.pt
    checkpoint_prefix=avg_last$ncheckpoints.
elif [ $scored_checkpoint = 'avg_closest' ]
then
    checkpoint_path=$save_dir/checkpoint.avg_closest$ncheckpoints.pt
    checkpoint_prefix=avg_closest$ncheckpoints.
elif [ $scored_checkpoint = 'last' ]
then
    checkpoint_path=$save_dir/checkpoint_last.pt
    checkpoint_prefix=last.
else
    echo "Argument scored_checkpoint is not valid."
fi

# Test
if [ -n "$mover" ]; then mover=$mover ; else mover="{}" ; fi
if [ -n "$beam" ]; then beam=$beam ; else beam=4 ; fi
if [ -n "$lenpen" ]; then lenpen=$lenpen ; else lenpen=0.6 ; fi
if [ -n "$temperature" ]; then temperature=$temperature ; else temperature=1 ; fi
if [ -n "$batch_size" ]; then batch_size=$batch_size ; else batch_size=64 ; fi
if [ -n "$include_eos" ]; then include_eos=$include_eos ; else include_eos=0 ; fi
if [ -n "$gen_subset" ]; then gen_subset=$gen_subset ; else gen_subset=test ; fi

# Logging
if [ -n "$log_format" ]; then log_format=$log_format ; else log_format=json ; fi
if [ -n "$log_interval" ]; then log_interval=$log_interval ; else log_interval=100 ; fi
if [ -n "$trainlog" ]; then trainlog=$trainlog ; else trainlog=train ; fi
if [ -n "$ftlog" ]; then ftlog=$ftlog ; else ftlog=finetune ; fi
if [ -n "$log_prefix" ]; then log_prefix=$log_prefix ; else log_prefix= ; fi
if [ -n "$testlog" ]; then testlog=$log_prefix$checkpoint_prefix$testlog ; else testlog=$log_prefix$checkpoint_prefix$gen_subset ; fi
if [ -n "$tensorboard_logdir" ]; then tensorboard_logdir=$tensorboard_logdir ; else tensorboard_logdir=$save_dir/logs ; fi
if [ $t != "results" ]
then
    mkdir -p $save_dir/logs
fi

# Task
if [ -n "$task" ]; then task=$task; else task=doc2doc_translation ; fi
if [ -z "$mode" ] && [ $t != "results" ]
then
    echo "You must specify --mode and --val (optional) for the doc2doc task."
    echo "Options are:"
    echo "1) --mode: none, block, slide_block "
    echo "   --val=<desired-num-of-tokens> (default=1000)"
    echo "2) --mode: n2n_block, slide_n2one, slide_n2n"
    echo "   --val=<desired-num-of-sentences> (default=5)"
    exit 0 
fi
if [ "$mode" = 'none' ] || [ "$mode" = 'block' ]  || [ "$mode" = 'slide_block' ]
then
    opt=num-tok
    if [ -n "$val" ]; then val=$val ; else val=1000 ; fi
elif [ "$mode" = 'n2n_block' ] || [ "$mode" = 'slide_n2one' ] || [ "$mode" = 'slide_n2n' ]
then
    opt=num-sent
    if [ -n "$val" ]; then val=$val; else val=4; fi
fi
if [ -n "$max_src_pos" ]; then max_src_pos=$max_src_pos; else max_src_pos=1024; fi
if [ -n "$max_tgt_pos" ]; then max_tgt_pos=$max_tgt_pos; else max_tgt_pos=1024; fi
if [ -n "$need_seg_label" ]; then need_seg_label=$need_seg_label; else need_seg_label=False ; fi
if [ -n "$context_discount" ]; then context_discount=$context_discount; else context_discount=1 ; fi

# Run #########################################################################
if [ $t = "train" ]
then
    echo $@ >> $save_dir/logs/$trainlog.log
    fairseq-train $data_dir \
    --save-dir $save_dir \
    --seed $seed \
    --source-lang $src \
    --target-lang $tgt \
    --num-workers 4 \
    --task $task \
    --mode $mode \
    --$opt $val \
    --need-seg-label $need_seg_label \
    --context-discount $context_discount \
    --original-loss-for-stopping \
    --arch $arch \
    --pse-segment-dim $pse_segment_dim \
    --position-shift $position_shift \
    --dropout $dropout \
    --criterion $criterion --label-smoothing $label_smoothing --weight-decay $weight_decay \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --lr-scheduler $lr_scheduler --lr $lr --warmup-updates $warmup_updates --warmup-init-lr $warmup_init_lr --min-lr $min_lr \
    --max-tokens $max_tokens \
    --update-freq $update_freq \
    --patience $patience \
    --keep-last-epochs $keep_last_epochs \
    --save-interval-updates $save_interval_updates \
    --keep-interval-updates $keep_interval_updates \
    --log-format $log_format \
    --log-interval $log_interval \
    --fp16 \
    --ddp-backend $ddp_backend \
    | tee -a $save_dir/logs/$trainlog.log
    # --max-source-positions $max_src_pos \
    # --max-target-positions $max_tgt_pos \
    # --save-interval-updates $siu \
    # --keep-interval-updates $kiu \
    # --no-epoch-checkpoints \
    # --keep-best-checkpoints $n_best_checkpoints \
    # --ddp-backend $ddp_backend \
    # --dropout $dropout \
    # --activation-dropout $activation_dropout \
    # --attention-dropout $attention_dropout \
    # --tensorboard-logdir $tensorboard_logdir \
###############################################################################
elif [ $t = "finetune" ]
then
    echo $@ >> $save_dir/logs/$trainlog.log
    fairseq-train $data_dir \
    --save-dir $save_dir \
    --seed $seed \
    --source-lang $src \
    --target-lang $tgt \
    --num-workers $num_workers \
    --task $task \
    --mode $mode \
    --$opt $val \
    --need-seg-label $need_seg_label \
    --context-discount $context_discount \
    --arch $arch \
    --pse-segment-dim $pse_segment_dim \
    --position-shift $position_shift \
    --dropout $dropout \
    --finetune-from-model $pretrained \
    --criterion $criterion --label-smoothing $label_smoothing --weight-decay $weight_decay \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --lr-scheduler $lr_scheduler --lr $lr --warmup-updates $warmup_updates --warmup-init-lr $warmup_init_lr --min-lr $min_lr \
    --max-tokens $max_tokens \
    --update-freq $update_freq \
    --patience $patience \
    --keep-last-epochs $keep_last_epochs \
    --save-interval-updates $save_interval_updates \
    --keep-interval-updates $keep_interval_updates \
    --log-format $log_format \
    --log-interval $log_interval \
    --fp16 \
    --ddp-backend $ddp_backend \
    | tee -a $save_dir/logs/$trainlog.log
###############################################################################
elif [ $t = "finefinetune" ]
then
    echo $@ >> $save_dir/logs/$trainlog.log
    fairseq-train $data_dir \
    --save-dir $save_dir \
    --seed $seed \
    --source-lang $src \
    --target-lang $tgt \
    --num-workers $num_workers \
    --task $task \
    --mode $mode \
    --$opt $val \
    --need-seg-label $need_seg_label \
    --context-discount $context_discount \
    --original_loss_for_stopping \
    --arch $arch \
    --pse-segment-dim $pse_segment_dim \
    --position-shift $position_shift \
    --dropout $dropout \
    --finetune-from-model $pretrained \
    --criterion $criterion --label-smoothing $label_smoothing --weight-decay $weight_decay \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --lr-scheduler fixed --lr 2e-04 --fa 1 --lr-shrink 0.99 \
    --max-tokens $max_tokens \
    --update-freq $update_freq \
    --patience $patience \
    --keep-last-epochs $keep_last_epochs \
    --log-format $log_format \
    --log-interval $log_interval \
    --tensorboard-logdir $tensorboard_logdir \
    --fp16 \
    --ddp-backend $ddp_backend \
    | tee -a $save_dir/logs/$trainlog.log
###############################################################################
elif [ $t = "test" ]
then
    echo $@ >> $save_dir/logs/$testlog.log
    fairseq-generate $data_dir \
    --gen-subset $gen_subset \
    --path $checkpoint_path \
    --task $task \
    --model-overrides $mover \
    --source-lang $src \
    --target-lang $tgt \
    --mode $mode \
    --$opt $val \
    --need-seg-label $need_seg_label \
    --batch-size $batch_size \
    --remove-bpe \
    --beam $beam \
    --lenpen $lenpen \
    --temperature $temperature \
    --include-eos $include_eos \
    --num-workers $num_workers \
    --seed $seed \
    | tee $save_dir/logs/$testlog.log
    # --skip-invalid-size-inputs-valid-test \
    # --max-source-positions $max_src_pos \
    # --max-target-positions $max_tgt_pos
    # extract src, tgt (ref) and system hypothesis (sys) from output
    grep ^S $save_dir/logs/$testlog.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- | sacremoses detokenize > $save_dir/logs/$testlog.out.src
    grep ^T $save_dir/logs/$testlog.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- | sacremoses detokenize > $save_dir/logs/$testlog.out.ref
    grep ^H $save_dir/logs/$testlog.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- | sacremoses detokenize > $save_dir/logs/$testlog.out.sys
    # score
    tools/mosesdecoder/scripts/generic/multi-bleu-detok.perl $save_dir/logs/$testlog.out.ref < $save_dir/logs/$testlog.out.sys | tee $save_dir/logs/$testlog.result
    cat $save_dir/logs/$testlog.result
###############################################################################
elif [ $t = "score-ref" ]
then
    fairseq-generate $data_dir \
    --task $task \
    --source-lang $src \
    --target-lang $tgt \
    --mode $mode \
    --$opt $val \
    --need-seg-label $need_seg_label \
    --path $checkpoint_path \
    --score-reference \
    --batch-size $batch_size \
    --remove-bpe '@@ ' \
    --num-workers $num_workers \
    | tee $save_dir/logs/$testlog.log
###############################################################################
elif [ $t = "test-suites" ]
then
    # evaluate on test-set
    bash $this_script --t=test --save_dir=$save_dir --data_dir=$data_dir --cuda=$cuda --lenpen=$lenpen --mover=$mover --mode=$mode --opt=$opt --val=$val --scored_checkpoint=$scored_checkpoint --need_seg_label=$need_seg_label --log_prefix=$log_prefix
    # evaluate on large pronouns test suite (original and with shuffled context)
    test_suites=~/dev/fairseq/data/$lang/data-bin/wmt17/test_suites
    for s in ""; do
        data_dir=$test_suites/large_pronoun/k3$s
        d=large_pronoun$s
        # score reference
        bash $this_script --t=score-ref --save_dir=$save_dir --data_dir=$test_suites/large_pronoun/k3$s --testlog=$d --cuda=$cuda --mover=$mover --mode=$mode --opt=$opt --val=$val --scored_checkpoint=$scored_checkpoint --batch_size=32 --need_seg_label=$need_seg_label --log_prefix=$log_prefix       # --mover="{'n_context_sents':'1'}"
        # evaluate
        echo "extract scores for $d..."
        if [ $mode = 'slide_n2n' ] || [ $mode = 'slide_n2one' ] 
        then
        grep ^H $save_dir/logs/$checkpoint_prefix$d.log | sed 's/^H-//g' | sort -nk 1 | awk 'NR % 4 == 0' | cut -f2 > $save_dir/logs/$checkpoint_prefix$d.score 
        else
        grep ^H $save_dir/logs/$checkpoint_prefix$d.log | sed 's/^H-//g' | sort -nk 1 | cut -f2 > $save_dir/logs/$checkpoint_prefix$d.score
        fi
        echo "evaluate model performance on test-suite by comparing scores..."
        contrapro=data/$lang/ContraPro
        python3 scripts/evaluate_contrapro.py --reference $contrapro/contrapro.json --scores $save_dir/logs/$checkpoint_prefix$d.score --maximize --results-file $save_dir/logs/$checkpoint_prefix$d.results > $save_dir/logs/$checkpoint_prefix$d.result
    done
    # print results
    bash $this_script --t=results --save_dir=$save_dir --mode=$mode --opt=$opt --val=$val --scored_checkpoint=$scored_checkpoint
    rm -rf $save_dir/logs/$checkpoint_prefix$d.full_score
###############################################################################
elif [ $t = "average" ]
then
    # python scripts/average_checkpoints.py \
    #     --inputs $save_dir/checkpoint.best_loss* \
    #     --output $save_dir/checkpoint.avg_best5.pt \
    # | tee $save_dir/logs/average.log
    # python scripts/average_checkpoints.py \
    #     --inputs $save_dir/ \
    #     --num-epoch-checkpoints $ncheckpoints \
    #     --output $save_dir/checkpoint.avg_last$ncheckpoints.pt \
    # | tee -a $save_dir/logs/average.log
    python scripts/average_checkpoints.py \
        --inputs $save_dir/ \
        --num-epoch-checkpoints $ncheckpoints \
        --closest-to-best \
        --output $save_dir/checkpoint.avg_closest$ncheckpoints.pt \
    | tee -a $save_dir/logs/average.log
    # python scripts/average_checkpoints.py \
    # --inputs $save_dir/checkpoint122.pt $save_dir/checkpoint123.pt $save_dir/checkpoint124.pt $save_dir/checkpoint125.pt $save_dir/checkpoint126.pt \
    # --output $save_dir/checkpoint.avg_closest5.pt \
    # | tee $save_dir/logs/average.log
###############################################################################
elif [ $t = "search-lenpen" ]
then
        logname=${checkpoint_prefix}valid_bleu_lenpen
        echo $save_dir >> $save_dir/logs/$logname.summary
        for l in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1 2
        do
        bash $this_script --t=test --save_dir=$save_dir --cuda=$cuda --lenpen=$l --gen_subset=valid --testlog=valid_bleu_lenpen$l --mode=$mode --opt=$opt --val=$val --scored_checkpoint=$scored_checkpoint --need_seg_label=$need_seg_label --mover=$mover
        # gather results
        echo "------- lenpen=$l ---------------------" >> $save_dir/logs/$logname.summary
        cat $save_dir/logs/$logname$l.result >> $save_dir/logs/$logname.summary
        rm $save_dir/logs/$logname$l*.out*
    done
    cat $save_dir/logs/$logname.summary
############################################################################### 
elif [ $t = "search-temperature" ]
then
    logname=${checkpoint_prefix}valid_bleu_temperature
    # 0.5 0.7 0.8 0.9 1 1.1 1.2 1.3 1.5
    for l in 0.8 0.9 1 1.1 1.2
    do
    # bash $this_script --t=test --save_dir=$save_dir --t=test --cuda=$cuda --temperature=$l --gen_subset=valid --testlog=valid_bleu_temperature$l --mode=$mode --opt=$opt --val=$val --scored_checkpoint=$scored_checkpoint --need_seg_label=$need_seg_label 
    # # gather results
    echo "------- temperature=$l ---------------------" >> $save_dir/logs/$logname.summary
    cat $save_dir/logs/$logname$l.result >> $save_dir/logs/$logname.summary
    # rm $save_dir/logs/$logname$l*.out*
    done
    cat $save_dir/logs/$logname.summary
###############################################################################
elif [ $t = "results-devloss" ]
then
    # best valid loss
    echo -n "$lr " >> tmp.results
    grep -oP "(?<=valid_best_loss\": \")[0-9\.]+" $save_dir/logs/train.log | tail -n1 >> tmp.results

    cat tmp.results
    rm tmp.results
    # echo -e "\n"
###############################################################################
elif [ $t = "results" ]
then
    echo "####################################################################"
    echo "RESULTS FOR $save_dir/logs/$checkpoint_prefix"

    # last checkpoint epoch according
    # grep "valid_best_loss" $save_dir/logs/train.log | tail -n1 | grep -oP "(?<=epoch\": )[0-9]+" >> tmp.results
    # best checkpoint epoch
    # grep "valid_best_loss" $save_dir/logs/train.log | grep valid_loss....$(grep -oP "(?<=valid_best_loss\": \")[0-9\.]+" $save_dir/logs/train.log | tail -n1) | grep -oP "(?<=epoch\": )[0-9]+" >> tmp.results
    # best checkpoint valid_current_loss
    grep "valid_best_loss" $save_dir/logs/train.log | grep valid_loss....$(grep -oP "(?<=valid_best_loss\": \")[0-9\.]+\"" $save_dir/logs/train.log | tail -n1) | head -n1 | grep -oP "(?<=valid_current_loss\": \")[0-9\.]+" >> tmp.results

    # best valid loss
    grep -oP "(?<=valid_best_loss\": \")[0-9\.]+" $save_dir/logs/train.log | tail -n1 >> tmp.results
    # grep -oP "(?<=valid_current_loss\": \")[0-9\.]+" $save_dir/logs/train.log | tail -n13 | head -n1 >> tmp.results
    # OR best valid bleu
    grep -oP "(?<=valid_best_bleu\": \")[0-9\.]+" $save_dir/logs/train.log | tail -n1 >> tmp.results

    # checkpoint
    echo ${checkpoint_prefix::-1} >> tmp.results

    # contrapro
    d=large_pronoun
    cat $save_dir/logs/$checkpoint_prefix$d.result | grep -A5 "statistics by ante distance" | grep -v "statistics by ante distance" | cut -d" " -f5 | cut -c1-6 | awk '{print $1*100}' >> tmp.results
    cat $save_dir/logs/$checkpoint_prefix$d.result | grep "total" | cut -d" " -f5 | awk '{print $1*100}' >> tmp.results

    # BLEU on test set
    d=test
    cat $save_dir/logs/$checkpoint_prefix$d.result | grep -oP "(?<=BLEU = )[0-9\.]+" >> tmp.results

    # transpose results
    cat tmp.results | tr "\n" "," > $save_dir/logs/all.results
    cat $save_dir/logs/all.results
    rm tmp.results
    echo -e "\n"

    # for d in lexical_choice; do
    #     echo "RESULTS FOR $d"
    #     cat $save_dir/logs/$checkpoint_prefix$d.result
    #     echo "-----------------------------------"
    #     echo ""
    # done
    # # for s in "" ".shuffled"; do
    # for s in ""; do
    #     d=test$s
    #     echo "RESULTS FOR $save_dir/logs/$checkpoint_prefix$d.score"
    #     echo ""
    #     cat $save_dir/logs/$checkpoint_prefix$d.result
    #     echo "-----------------------------------"
    #     echo ""
    # done
    # for s in ""; do
    #     d=large_pronoun$s
    #     echo "RESULTS FOR $save_dir/logs/$checkpoint_prefix$d.results"
    #     echo ""
    #     grep total $save_dir/logs/$checkpoint_prefix$d.result
    #     echo "-----------------------------------"
    #     echo ""
    # done
    # echo "####################################################################"
    # echo "RESULTS FOR $save_dir/logs/$checkpoint_prefix"

    # # best checkpoint epoch according to valid loss
    # grep "valid_best_loss" $save_dir/logs/train.log | tail -n1 | grep -oP "(?<=epoch\": )[0-9]+" >> tmp.results
    # # OR best checkpoint epoch according to valid bleu
    # grep "valid_best_bleu" $save_dir/logs/train.log | tail -n1 | grep -oP "(?<=epoch\": )[0-9]+" >> tmp.results

    # # discourse dev set
    # for d in deixis_dev lex_cohesion_dev
    # do
    #     cat $save_dir/logs/$checkpoint_prefix$d.result | grep -oP "(?<=Total accuracy:  )[0-9\.]+" | awk -v x=100 '{ print $1*x }' >> tmp.results
    # done

    # # best valid loss
    # grep -oP "(?<=valid_best_loss\": \")[0-9\.]+" $save_dir/logs/train.log | tail -n1 >> tmp.results
    # # OR best valid bleu
    # grep -oP "(?<=valid_best_bleu\": \")[0-9\.]+" $save_dir/logs/train.log | tail -n1 >> tmp.results

    # # checkpoint
    # echo ${checkpoint_prefix::-1} >> tmp.results

    # # discourse test set
    # for d in deixis_test lex_cohesion_test ellipsis_infl ellipsis_vp
    # do
    #     cat $save_dir/logs/$checkpoint_prefix$d.result | grep -oP "(?<=Total accuracy:  )[0-9\.]+" | awk -v x=100 '{ print $1*x }' >> tmp.results
    # done

    # # BLEU on test set
    # d=test
    # cat $save_dir/logs/$checkpoint_prefix$d.result | grep -oP "(?<=BLEU = )[0-9\.]+" >> tmp.results

    # # transpose results
    # cat tmp.results | tr "\n" ","
    # rm tmp.results
    # echo -e "\n"
###############################################################################
else
    echo "Argument t is not valid."
fi

