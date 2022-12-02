#!/bin/bash

echo $(grep "valid_loss" $1 | grep valid_loss....$(grep -oP "(?<=valid_best_loss\": \")[0-9\.]+"\" $1 | tail -n1) | head -n1 | grep -oP "(?<=epoch\": )[0-9]+")
# grep "valid_best_loss" $sd | grep valid_loss....$(grep -oP "(?<=valid_best_loss\": \")[0-9\.]+"\" $sd | tail -n1) | head -n1 |  grep -oP "(?<=epoch\": )[0-9]+"