@echo off
python -u main.py --learn-tracks^
 --params-file best-params.dat^
 --params-file lf-params.dat^
 --layer-functions layer_functions^
 --layer-functions layer_functions-dp^
 --beg 2000 --end 2249^
 --event-dir d:\kaggle-trackml-data\train_1\train_1^
 -p learn__true_coords=False^
 -p learn__analyze_pairs=True^
 --save-pairs layer_functions-new^
 > learn-tracks-analyze-pairs.log

echo "learned pairs"

del layer_functions-pair.csv
ren layer_functions-new-pair.csv layer_functions-pair.csv

call big_run.bat
