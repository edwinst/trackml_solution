@echo off
python -u main.py --learn-tracks^
 --layer-functions layer_functions^
 --params-file best-params.dat^
 --params-file lf-params.dat^
 --beg 2200 --end 2399^
 --save-displacements layer_functions-new^
 --event-dir d:\kaggle-trackml-data\train_1\train_1^
 > learn-displacements.log

echo "learned displacements"

python -u main.py --learn-tracks^
 --layer-functions layer_functions^
 --layer-functions layer_functions-new-dp^
 --params-file best-params.dat^
 --params-file lf-params.dat^
 --beg 2030 --end 2059^
 --save-displacements layer_functions-err^
 --event-dir d:\kaggle-trackml-data\train_1\train_1^
 --save-analytics d:\kaggle-trackml-tmp\ana.csv^
 > test-displacements.log

echo "analyzed errors"

del layer_functions-dp.csv
ren layer_functions-new-dp.csv layer_functions-dp.csv

call big_run.bat
