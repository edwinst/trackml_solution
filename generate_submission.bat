@echo off
set NAME=%1

set PROG_SHA1SUM=C:\cygwin\bin\sha1sum.exe
set PROG_7Z="C:\Program Files\7-Zip\7z.exe"

set LAYER_FUNCTIONS=layer_functions
set LAYER_FUNCTIONS_DP=layer_functions-dp
set LAYER_FUNCTIONS_PAIR=layer_functions-pair

set SUBMISSION_DIR=D:\kaggle-trackml-submissions
set SUBMISSION_FILE=%SUBMISSION_DIR%\sub-%NAME%.csv

set SUBMISSION_ZIPPED_LAYER_FUNCTIONS=%SUBMISSION_DIR%\layer_functions-%NAME%.csv.7z
set SUBMISSION_ZIPPED_LAYER_FUNCTIONS_DP=%SUBMISSION_DIR%\layer_functions-dp-%NAME%.csv.7z
set SUBMISSION_ZIPPED_LAYER_FUNCTIONS_PAIR=%SUBMISSION_DIR%\layer_functions-pair-%NAME%.csv.7z

echo SHA-1s of layer function files:
%PROG_SHA1SUM% %LAYER_FUNCTIONS%.csv
%PROG_SHA1SUM% %LAYER_FUNCTIONS_DP%.csv
%PROG_SHA1SUM% %LAYER_FUNCTIONS_PAIR%.csv

echo compressing layer function files...
%PROG_7Z% a -t7z -mx=9 %SUBMISSION_ZIPPED_LAYER_FUNCTIONS%      %LAYER_FUNCTIONS%.csv
%PROG_7Z% a -t7z -mx=9 %SUBMISSION_ZIPPED_LAYER_FUNCTIONS_DP%   %LAYER_FUNCTIONS_DP%.csv
%PROG_7Z% a -t7z -mx=9 %SUBMISSION_ZIPPED_LAYER_FUNCTIONS_PAIR% %LAYER_FUNCTIONS_PAIR%.csv

echo SHA-1s of decompressed layer function files:
%PROG_7Z% x -so %SUBMISSION_ZIPPED_LAYER_FUNCTIONS%      | %PROG_SHA1SUM%
%PROG_7Z% x -so %SUBMISSION_ZIPPED_LAYER_FUNCTIONS_DP%   | %PROG_SHA1SUM%
%PROG_7Z% x -so %SUBMISSION_ZIPPED_LAYER_FUNCTIONS_PAIR% | %PROG_SHA1SUM%

echo start time:
echo %time%

@echo on
python -u main.py^
 --with-cells^
 --params-file best-params.dat^
 --params-file lf-params.dat^
 --params-file iter-params.dat^
 --params-file crazy-fat-sunset-params.dat^
 -p post__nonphys_odd=True^
 --layer-functions %LAYER_FUNCTIONS%^
 --layer-functions %LAYER_FUNCTIONS_DP%^
 --layer-functions %LAYER_FUNCTIONS_PAIR%^
 --data-dir C:\Users\edwin\nobackup\kaggle-trackml-data^
 --event-dir D:\kaggle-trackml-data\test\test^
 --beg 0 --end 124^
 --submission-file %SUBMISSION_FILE%^
 --log 10

@echo off
echo end time:
call echo %time%

echo SHA-1 of submission file:
%PROG_SHA1SUM% %SUBMISSION_FILE%

