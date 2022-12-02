#!/bin/bash

# This script provides command-line statements for training and testing a
# Transformer model for the translation of a concatenated sequence of
# source and/or target sentences, via the "mode" and "val" arguments

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

# Script variables
if [ -n "$src" ]; then src=$src ; else src=en ; fi
if [ -n "$tgt" ]; then tgt=$tgt ; else tgt=ru ; fi
if [ -n "$corpus" ]; then corpus=$corpus ; else corpus=voita_opensubs/context_aware ; fi
if [ -n "$lang" ]; then lang=$lang; else lang=$src-$tgt ; fi
if [ -n "$this_script" ]; then this_script=$this_script; else this_script=sh/run/$lang/$corpus/concat.sh ; fi

# Common
if [ -n "$cuda" ] ; then export CUDA_VISIBLE_DEVICES=$cuda ; fi
if [ -n "$seed" ]; then seed=$seed ; else seed=0 ; fi
if [ -n "$num_workers" ]; then num_workers=$num_workers ; else num_workers=8 ; fi

# Data
if [ -n "$data_dir" ]; then data_dir=$data_dir; else data_dir=data/$lang/data-bin/$corpus/standard ; fi
if [ -n "$max_tokens" ]; then max_tokens=$max_tokens ; else max_tokens=16000; fi
if [ -n "$update_freq" ]; then update_freq=$update_freq ; else update_freq=1; fi

# Model
if [ -n "$arch" ]; then arch=$arch ; else arch=concat_voita_fairseq ; fi

# Loss
if [ -n "$criterion" ]; then criterion=$criterion; else criterion=label_smoothed_cross_entropy; fi
if [ -n "$label_smoothing" ]; then label_smoothing=$label_smoothing; else label_smoothing=0.1; fi
if [ -n "$weight_decay" ]; then weight_decay=$weight_decay; else weight_decay=0.0; fi

# Optimization
if [ -n "$lr_scheduler" ]; then lr_scheduler=$lr_scheduler; else lr_scheduler=inverse_sqrt; fi
if [ -n "$lr" ]; then lr=$lr; else lr=5e-04; fi
if [ -n "$min_lr" ]; then min_lr=$min_lr ; else min_lr=1e-09 ; fi # stop training when the lr reaches this minimum (default -1.0)
if [ -n "$total_num_update" ]; then total_num_update=$total_num_update; else total_num_update=16000; fi # 16000 ups corresponds to 15 epoch with max_tokens=128K
if [ -n "$warmup_updates" ]; then warmup_updates=$warmup_updates ; else warmup_updates=4000 ; fi
if [ -n "$warmup_init_lr" ]; then warmup_init_lr=$warmup_init_lr ; else warmup_init_lr=1e-07 ; fi
if [ -n "$end_learning_rate" ]; then end_learning_rate=$end_learning_rate ; else end_learning_rate=1e-09 ; fi

# Checkpoints
if [[ $save_dir != "checkpoints/"* ]]; then save_dir=checkpoints/$lang/$corpus/$save_dir; fi
if [ -n "$ncheckpoints" ]; then ncheckpoints=$ncheckpoints ; else ncheckpoints=5 ; fi
if [ -n "$patience" ]; then patience=$patience ; else patience=10 ; fi
if [ -n "$max_update"]; then max_update=$max_update ; else max_update=0 ; fi
if [ -n "$keep_last_epochs" ]; then keep_last_epochs=$keep_last_epochs ; else keep_last_epochs=$patience ; fi
if [ -n "$keep_best_checkpoints" ]; then keep_best_checkpoints=$keep_best_checkpoints ; else keep_best_checkpoints=$ncheckpoints ; fi
if [ -n "$save_interval_updates" ]; then save_interval_updates=$save_interval_updates ; else save_interval_updatesu=2048 ; fi
if [ -n "$keep_interval_updates" ]; then keep_interval_updates=$keep_interval_updates ; else keep_interval_updates=10 ; fi
if [ -n "$scored_checkpoint" ]; then scored_checkpoint=$scored_checkpoint ; else scored_checkpoint=best ; fi
if [ $scored_checkpoint = 'best' ]
then
    checkpoint_path=$save_dir/checkpoint_best.pt
    checkpoint_prefix=best.
elif [ $scored_checkpoint = 'avg_last5' ]
then
    checkpoint_path=$save_dir/checkpoint.avg_last$ncheckpoints.pt
    checkpoint_prefix=avg_last$ncheckpoints.
elif [ $scored_checkpoint = 'last' ]
then
    checkpoint_path=$save_dir/checkpoint_last.pt
    checkpoint_prefix=last.
else
    echo "Argument checkpoint_path is not valid."
fi

# Test
if [ -n "$mover" ]; then mover=$mover ; else mover="{}" ; fi
if [ -n "$lenpen" ]; then lenpen=$lenpen ; else lenpen=0.6 ; fi
if [ -n "$temperature" ]; then temperature=$temperature ; else temperature=1 ; fi
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
if [ -z "$mode" ]
then
    echo "You must specify --mode and --val (optional) for the doc2doc task."
    echo "Options are:"
    echo "1) --mode: none, block, slide_block "
    echo "   --val=<desired-num-of-tokens> (default=1000)"
    echo "2) --mode: n2n_block, slide_n2one, slide_n2n"
    echo "   --val=<desired-num-of-sentences> (default=5)"
    exit 0 
fi
if [ $mode = 'none' ] || [ $mode = 'block' ]  || [ $mode = 'slide_block' ]
then
    opt=num-tok
    if [ -n "$val" ]; then val=$val ; else val=1000 ; fi
elif [ $mode = 'n2n_block' ] || [ $mode = 'slide_n2one' ] || [ $mode = 'slide_n2n' ]
then
    opt=num-sent
    if [ -n "$val" ]; then val=$val; else val=4; fi
fi
if [ -n "$max_src_pos" ]; then max_src_pos=$max_src_pos; else max_src_pos=1024; fi
if [ -n "$max_tgt_pos" ]; then max_tgt_pos=$max_tgt_pos; else max_tgt_pos=1024; fi
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
    --num-workers $num_workers \
    --task $task \
    --mode $mode \
    --$opt $val \
    --context-discount $context_discount \
    --arch $arch \
    --criterion $criterion --label-smoothing $label_smoothing --weight-decay $weight_decay \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
    --lr-scheduler $lr_scheduler --lr $lr --warmup-updates $warmup_updates --warmup-init-lr $warmup_init_lr --min-lr $min_lr \
    --max-tokens $max_tokens \
    --update-freq $update_freq \
    --patience $patience \
    --keep-last-epochs $keep_last_epochs \
    --log-format $log_format \
    --log-interval $log_interval \
    --tensorboard-logdir $tensorboard_logdir \
    --fp16 \
    | tee -a $save_dir/logs/$trainlog.log
    # --lr-scheduler $lr_scheduler --lr $lr --warmup-updates $warmup_updates --warmup-init-lr=$warmup_init_lr --min-lr $min_lr \
    # --lr-scheduler $lr_scheduler --lr $lr --warmup-updates $warmup_updates --total-num-update $total_num_update --end-learning-rate $end_learning_rate \
    # --ddp-backend $ddp_backend \
    # --save-interval-updates $save_interval_updates \
    # --keep-interval-updates $keep_interval_updates \
    # --no-epoch-checkpoints \
    # --max-source-positions $max_src_pos \
    # --max-target-positions $max_tgt_pos \
    # --eval-bleu \
    # --eval-tokenized-bleu \
    # --eval-bleu-args '{"beam": 4, "lenpen": 0.6}' \
    # --eval-bleu-remove-bpe \
    # --best-checkpoint-metric bleu \
    # --maximize-best-checkpoint-metric \
###############################################################################
elif [ $t = "test" ]
then
    # echo $@ >> $save_dir/logs/$testlog.log
    # fairseq-generate $data_dir \
    # --gen-subset $gen_subset \
    # --path $checkpoint_path \
    # --task $task \
    # --source-lang $src \
    # --target-lang $tgt \
    # --mode $mode \
    # --$opt $val \
    # --batch-size 64 \
    # --remove-bpe \
    # --beam 4 \
    # --lenpen $lenpen \
    # --temperature $temperature \
    # --include-eos $include_eos \
    # --num-workers $num_workers \
    # --seed $seed \
    # | tee $save_dir/logs/$testlog.log
    # --max-source-positions $max_src_pos \
    # --max-target-positions $max_tgt_pos \
    # extract src, tgt (ref) and system hypothesis (sys) from output, keep 1 every 4 to match the context-agnostic test set
    grep ^S $save_dir/logs/$testlog.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- > $save_dir/logs/$testlog.out.src
    grep ^T $save_dir/logs/$testlog.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- > $save_dir/logs/$testlog.out.ref
    grep ^H $save_dir/logs/$testlog.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- > $save_dir/logs/$testlog.out.sys
    # score
    tools/mosesdecoder/scripts/generic/multi-bleu.perl -lc $save_dir/logs/$testlog.out.ref < $save_dir/logs/$testlog.out.sys | tee $save_dir/logs/$testlog.result
    cat $save_dir/logs/$testlog.result
###############################################################################
elif [ $t = "test-by-available-context" ]
then
    if [ ! -f "$save_dir/logs/$testlog.log" ]
    then
    bash $this_script --t=test --save_dir=$save_dir --data_dir=$data_dir --cuda=$cuda --lenpen=$lenpen --mover=$mover --mode=$mode --opt=$opt --val=$val --scored_checkpoint=$scored_checkpoint
    fi
    # extract src, tgt (ref) and system hypothesis (sys) from generation output
    grep ^S $save_dir/logs/$testlog.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- > $save_dir/logs/$testlog.all.out.src
    grep ^T $save_dir/logs/$testlog.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- > $save_dir/logs/$testlog.all.out.ref
    grep ^H $save_dir/logs/$testlog.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- > $save_dir/logs/$testlog.all.out.sys
    # extract sentences by available context
    rm $save_dir/logs/$testlog.bycontext.result
    available_context=0
    for n in 1 2 3 0
    do
    cat $save_dir/logs/$testlog.all.out.ref | awk -v n=$n 'NR % 4 == n' > $save_dir/logs/$testlog.out.ref 
    cat $save_dir/logs/$testlog.all.out.sys | awk -v n=$n 'NR % 4 == n' > $save_dir/logs/$testlog.out.sys
    # score
    # echo "------- available_context=$available_context ---------------------" >> $save_dir/logs/$testlog.bycontext.result
    tools/mosesdecoder/scripts/generic/multi-bleu.perl -lc $save_dir/logs/$testlog.out.ref  < $save_dir/logs/$testlog.out.sys | grep -oP "(?<=BLEU = )[0-9\.]+" | tr "\n" "," >> $save_dir/logs/$testlog.bycontext.result
    # available_context=$(($available_context+1))
    done
    # print results
    cat $save_dir/logs/$testlog.bycontext.result
###############################################################################
elif [ $t = "test-by-position" ]
then
    log_prefix=jumping.
    if [ ! -f "$save_dir/logs/$log_prefix$testlog.log" ]
    then
    bash $this_script --t=test --include_eos=1 --save_dir=$save_dir --data_dir=$data_dir --cuda=$cuda --lenpen=$lenpen --mover=$mover --mode=n2n_block --opt=$opt --val=$val --scored_checkpoint=$scored_checkpoint --log_prefix=$log_prefix
    fi
    testlog=$log_prefix$testlog
    # extract src, tgt (ref) and system hypothesis (sys) from generation output
    grep ^S $save_dir/logs/$testlog.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- > $save_dir/logs/$testlog.all.out.src
    grep ^T $save_dir/logs/$testlog.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- > $save_dir/logs/$testlog.all.out.ref
    grep ^H $save_dir/logs/$testlog.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- > $save_dir/logs/$testlog.all.out.sys
    python scripts/align_sys_ref.py --src $save_dir/logs/$testlog.all.out.src --hyp $save_dir/logs/$testlog.all.out.sys --ref $save_dir/logs/$testlog.all.out.ref
    # extract sentences by available context
    rm $save_dir/logs/$testlog.byposition.result
    position=0
    for n in 1 2 3 0
    do
    cat $save_dir/logs/$testlog.all.out.src.new | awk -v n=$n 'NR % 4 == n' > $save_dir/logs/$testlog.out.src
    cat $save_dir/logs/$testlog.all.out.ref.new | awk -v n=$n 'NR % 4 == n' > $save_dir/logs/$testlog.out.ref 
    cat $save_dir/logs/$testlog.all.out.sys.new | awk -v n=$n 'NR % 4 == n' > $save_dir/logs/$testlog.out.sys
    # score
    # echo "------- position=$position ---------------------" >> $save_dir/logs/$testlog.byposition.result
    tools/mosesdecoder/scripts/generic/multi-bleu.perl -lc $save_dir/logs/$testlog.out.ref  < $save_dir/logs/$testlog.out.sys | grep -oP "(?<=BLEU = )[0-9\.]+" | tr "\n" "," >> $save_dir/logs/$testlog.byposition.result
    # position=$(($position+1))
    done
    # print results
    cat $save_dir/logs/$testlog.byposition.result
###############################################################################
elif [ $t = "score-ref" ]
then
    fairseq-generate $data_dir \
    --task $task \
    --source-lang $src \
    --target-lang $tgt \
    --mode $mode \
    --$opt $val \
    --path $checkpoint_path \
    --score-reference \
    --batch-size 64 \
    --remove-bpe '@@ ' \
    --num-workers $num_workers \
    | tee $save_dir/logs/$testlog.log
###############################################################################
elif [ $t = "test-suites" ]
then
    # evaluate on test-set
    bash $this_script --t=test --save_dir=$save_dir --data_dir=$data_dir --cuda=$cuda --lenpen=$lenpen --mover=$mover --mode=$mode --opt=$opt --val=$val --scored_checkpoint=$scored_checkpoint
    # evaluate on consistency testset
    test_suite_data=data/en-ru/data-bin/voita_opensubs/testset_consistency
    test_suite_repo=data/en-ru/test_suites/good-translation-wrong-in-context
    for d in deixis_dev deixis_test lex_cohesion_dev lex_cohesion_test ellipsis_infl ellipsis_vp
    do
        # score reference
        bash $this_script --t=score-ref --save_dir=$save_dir --data_dir=$test_suite_data/$d --testlog=$d --cuda=$cuda --mover=$mover --mode=$mode --opt=$opt --val=$val --scored_checkpoint=$scored_checkpoint
        # evaluate
        echo "extract scores for $d..."
        grep ^H $save_dir/logs/$checkpoint_prefix$d.log | sed 's/^H-//g' | sort -nk 1 | cut -f2 > $save_dir/logs/$checkpoint_prefix$d.full_score 
        awk 'NR % 4 == 0' $save_dir/logs/$checkpoint_prefix$d.full_score | cut -c2- > $save_dir/logs/$checkpoint_prefix$d.score
        echo "evaluate model performance on $d by comparing scores..."
        python3 scripts/evaluate_consistency_voita.py --repo-dir $test_suite_repo --test $d --scores $save_dir/logs/$checkpoint_prefix$d.score --results-file $save_dir/logs/$checkpoint_prefix$d.results > $save_dir/logs/$checkpoint_prefix$d.result
        # Note: except for lexical cohesion, the correct example is always the first in each contrastive block
        echo "-----------------------------------"
    done
    bash $this_script --t=results --save_dir=$save_dir --mode=$mode --opt=$opt --val=$val
###############################################################################
elif [ $t = "dev-suites" ]
then
    # evaluate on consistency testset
    test_suite_data=data/en-ru/data-bin/voita_opensubs/testset_consistency
    test_suite_repo=data/en-ru/test_suites/good-translation-wrong-in-context
    for d in deixis_dev lex_cohesion_dev
    do
        # score reference
        bash $this_script --t=score-ref --save_dir=$save_dir --data_dir=$test_suite_data/$d --testlog=$d --cuda=$cuda --mover=$mover --mode=$mode --opt=$opt --val=$val --scored_checkpoint=$scored_checkpoint
        # evaluate
        echo "extract scores for $d..."
        grep ^H $save_dir/logs/$checkpoint_prefix$d.log | sed 's/^H-//g' | sort -nk 1 | cut -f2 > $save_dir/logs/$checkpoint_prefix$d.full_score 
        awk 'NR % 4 == 0' $save_dir/logs/$checkpoint_prefix$d.full_score | cut -c2- > $save_dir/logs/$checkpoint_prefix$d.score
        echo "evaluate model performance on $d by comparing scores..."
        python3 scripts/evaluate_consistency_voita.py --repo-dir $test_suite_repo --test $d --scores $save_dir/logs/$checkpoint_prefix$d.score --results-file $save_dir/logs/$checkpoint_prefix$d.results > $save_dir/logs/$checkpoint_prefix$d.result
        # Note: except for lexical cohesion, the correct example is always the first in each contrastive block
        echo "-----------------------------------"
    done
    bash $this_script --t=results --save_dir=$save_dir --mode=$mode --opt=$opt --val=$val
############################################################################### 
elif [ $t = "search-lenpen" ]
then
    logname=${checkpoint_prefix}valid_bleu_lenpen
    for l in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1 2
    do
    bash $this_script --t=test --save_dir=$save_dir --t=test --cuda=$cuda --lenpen=$l --gen_subset=valid --testlog=valid_bleu_lenpen$l --mode=$mode --opt=$opt --val=$val --scored_checkpoint=$scored_checkpoint
    # gather results
    echo "------- lenpen=$l ---------------------" >> $save_dir/logs/$logname.summary
    cat $save_dir/logs/$logname$l.result >> $save_dir/logs/$logname.summary
    # rm $save_dir/logs/$logname$l*
    done
    cat $save_dir/logs/$logname.summary
############################################################################### 
elif [ $t = "search-temperature" ]
then
    logname=${checkpoint_prefix}valid_bleu_temperature
    # 0.5 0.7 0.8 0.9 1 1.1 1.2 1.3 1.5
    for l in 1.1 1.2 1.3 1.5
    do
    # bash $this_script --t=test --save_dir=$save_dir --t=test --cuda=$cuda --temperature=$l --gen_subset=valid --testlog=valid_bleu_temperature$l --mode=$mode --opt=$opt --val=$val --scored_checkpoint=$scored_checkpoint
    # gather results
    echo "------- temperature=$l ---------------------" >> $save_dir/logs/$logname.summary
    cat $save_dir/logs/$logname$l.score >> $save_dir/logs/$logname.summary
    # rm $save_dir/logs/$logname$l*
    done
    cat $save_dir/logs/$logname.summary
###############################################################################
elif [ $t = "average" ]
then
    # python scripts/average_checkpoints.py \
    #     --inputs $save_dir/checkpoint.best_loss* \
    #     --output $save_dir/checkpoint.avg_best5.pt \
    # | tee $save_dir/logs/average.log
    python scripts/average_checkpoints.py \
        --inputs $save_dir/ \
        --num-epoch-checkpoints 5 \
        --output $save_dir/checkpoint.avg_last5.pt \
    # | tee $save_dir/logs/average.log
    # python scripts/average_checkpoints.py \
    # --inputs $save_dir/checkpoint366.pt $save_dir/checkpoint348.pt $save_dir/checkpoint376.pt $save_dir/checkpoint.best_loss_3.566.pt $save_dir/checkpoint.best_loss_3.569.pt \
    # --output $save_dir/checkpoint.avg_best5.pt
############################################################################### 
elif [ $t = "results" ]
then
    echo "####################################################################"
    echo "RESULTS FOR $save_dir/logs/$checkpoint_prefix"

    # best checkpoint epoch according to valid loss
    grep "valid_best_loss" $save_dir/logs/train.log | tail -n1 | grep -oP "(?<=epoch\": )[0-9]+" >> tmp.results
    # OR best checkpoint epoch according to valid bleu
    grep "valid_best_bleu" $save_dir/logs/train.log | tail -n1 | grep -oP "(?<=epoch\": )[0-9]+" >> tmp.results

    # discourse dev set
    for d in deixis_dev lex_cohesion_dev
    do
        cat $save_dir/logs/$checkpoint_prefix$d.result | grep -oP "(?<=Total accuracy:  )[0-9\.]+" | awk -v x=100 '{ print $1*x }' >> tmp.results
    done

    # best valid loss
    grep -oP "(?<=valid_best_loss\": \")[0-9\.]+" $save_dir/logs/train.log | tail -n1 >> tmp.results
    # OR best valid bleu
    grep -oP "(?<=valid_best_bleu\": \")[0-9\.]+" $save_dir/logs/train.log | tail -n1 >> tmp.results

    # checkpoint
    echo ${checkpoint_prefix::-1} >> tmp.results

    # discourse test set
    for d in deixis_test lex_cohesion_test ellipsis_infl ellipsis_vp
    do
        cat $save_dir/logs/$checkpoint_prefix$d.result | grep -oP "(?<=Total accuracy:  )[0-9\.]+" | awk -v x=100 '{ print $1*x }' >> tmp.results
    done

    # BLEU on test set
    d=test
    cat $save_dir/logs/$checkpoint_prefix$d.result | grep -oP "(?<=BLEU = )[0-9\.]+" >> tmp.results

    # transpose results
    cat tmp.results | tr "\n" ","
    rm tmp.results
    echo -e "\n"

    # echo "####################################################################"
    # echo "RESULTS FOR $save_dir/logs/$checkpoint_prefix"

    # echo ""
    # grep "valid_best_loss" $save_dir/logs/train.log | tail -n1 | grep -o "epoch......"
    # grep -o "valid_best_loss.........." $save_dir/logs/train.log | tail -n1

    # for d in deixis_dev lex_cohesion_dev
    # do
    #     echo ""
    #     cat $save_dir/logs/$checkpoint_prefix$d.result | head -n2
    #     echo ""
    # done

    # d=test
    # echo ""
    # cat $save_dir/logs/$checkpoint_prefix$d.result
    # for d in deixis_dev deixis_test lex_cohesion_dev lex_cohesion_test ellipsis_infl ellipsis_vp
    # do
    #     echo ""
    #     cat $save_dir/logs/$checkpoint_prefix$d.result | head -n2
    #     echo ""
    # done
else
    echo "Argument t is not valid."
fi
