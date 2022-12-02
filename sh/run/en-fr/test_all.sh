#!/bin/bash

cuda=1

# EVALUATE
for sdir in "standard/k1" "standard/fromnei/k1" "standard/k3" "standard/fromnei/k3"; do
    echo "EVALUATE $save_dir"
    bash sh/run/en-fr/iwslt17/han.sh --t=boom --cuda=$cuda --save_dir=$save_dir --lenpen=0.6
    echo "--------------------------------------------------------------------"
    echo ""
done
###
for sdir in "split/k1" "split/k3" "split/fromnei/k1" "split/fromnei/k3"; do
    echo "EVALUATE $save_dir"
    bash sh/run/en-fr/iwslt17/han.sh --t=boom --cuda=$cuda --save_dir=$save_dir --lenpen=1.3
    echo "--------------------------------------------------------------------"
    echo ""
done
###
for sdir in "fromsplit/k1" "fromsplit/k3" "fromsplit/fromnei/k1" "fromsplit/fromnei/k3"; do
    echo "EVALUATE $save_dir"
    bash sh/run/en-fr/iwslt17/han.sh --t=boom --cuda=$cuda --save_dir=$save_dir --lenpen=1.0
    echo "--------------------------------------------------------------------"
    echo ""
done

# PRINT RESULTS

for sdir in "standard/k1" "standard/fromnei/k1" "standard/k3" "standard/fromnei/k3"; do
    echo "RESULTS FOR $save_dir"
    bash sh/run/en-fr/iwslt17/han.sh --t=results --cuda=$cuda --save_dir=$save_dir
    echo "--------------------------------------------------------------------"
    echo ""
done
###
for sdir in "split/k1" "split/k3" "split/fromnei/k1" "split/fromnei/k3"; do
    echo "RESULTS FOR $save_dir"
    bash sh/run/en-fr/iwslt17/han.sh --t=results --cuda=$cuda --save_dir=$save_dir
    echo "--------------------------------------------------------------------"
    echo ""
done
###
for sdir in "fromsplit/k1" "fromsplit/k3" "fromsplit/fromnei/k1" "fromsplit/fromnei/k3"; do
    echo "RESULTS FOR $save_dir"
    bash sh/run/en-fr/iwslt17/han.sh --t=results --cuda=$cuda --save_dir=$save_dir
    echo "--------------------------------------------------------------------"
    echo ""
done