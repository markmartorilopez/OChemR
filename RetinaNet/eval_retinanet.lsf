#!/usr/bin/env bash
#BSUB -q x86_1h
#BSUB -n 2
#BSUB -M 32768
#BSUB -gpu "num=1"
#BSUB -e "err/evalRetinaNet.sterr.%J"
#BSUB -o "logs/evalRetinaNet.stdout.%J"
#BSUB -J "RetinaNet_eval"


python eval_RetinaNet.py
