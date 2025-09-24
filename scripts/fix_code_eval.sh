# when you meet some error in code eval, run this script to fix it

mkdir -p ./code_eval
wget https://raw.githubusercontent.com/huggingface/evaluate/main/metrics/code_eval/code_eval.py -O ./code_eval/code_eval.py
wget https://raw.githubusercontent.com/huggingface/evaluate/main/metrics/code_eval/execute.py -O ./code_eval/execute.py