@echo off

set PROG_SHA1_SUM=C:\cygwin\bin\sha1sum.exe
set PROG_7Z="C:\Program Files\7-Zip\7z.exe"
set PROG_GZIP=C:\cygwin\bin\gzip.exe

set NAME=%1
set CSV_NAME=sub-%NAME%.csv
set LOG_NAME=sub-%NAME%.log
set ZIP_NAME=sub-%NAME%-lzma2.csv.7z

echo compressing log file
%PROG_GZIP% %LOG_NAME%

echo SHA-1 of uncompressed submission
%PROG_SHA1_SUM% %CSV_NAME%

echo compressing...
%PROG_7Z% a -t7z -mx=9 %ZIP_NAME% %CSV_NAME%

echo SHA-1 of decompressed submission
%PROG_7Z% x -so %ZIP_NAME% | %PROG_SHA1_SUM%

echo SHA-1 of compressed submission
%PROG_SHA1_SUM% %ZIP_NAME%

echo compressed submission is ready: %ZIP_NAME%
