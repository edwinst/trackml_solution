@echo off
setlocal
call conda activate base
python -u main.py %*
endlocal