#!/bin/bash
# 2x2 proactivity ablation on the eval video: input gate x output gate.
cd ~/Mobile-O/experiments/proactive-inference
source ~/AsyncReasoning/.venv/bin/activate
mkdir -p results

echo "########## 1/4 NON-PROACTIVE (input=even, output=off) ##########"
python proactive_system.py --input-mode even     --output-gate off --out results/abl_nonproactive.json
echo "########## 2/4 OUTPUT-ONLY (input=even, output=on) ##########"
python proactive_system.py --input-mode even     --output-gate on  --out results/abl_output_only.json
echo "########## 3/4 INPUT-ONLY (input=adaptive, output=off) ##########"
python proactive_system.py --input-mode adaptive --output-gate off --out results/abl_input_only.json
echo "########## 4/4 BOTH (input=adaptive, output=on) ##########"
python proactive_system.py --input-mode adaptive --output-gate on  --out results/abl_both.json
echo "########## DONE ##########"
