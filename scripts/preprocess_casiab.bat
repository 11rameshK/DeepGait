@echo off
set ROOT=C:\Users\RAMESH\Desktop\Projects\Nikhil 9000
cd /d "%ROOT%\OpenGait-master"
python datasets\pretreatment.py -i "%ROOT%\CASIA-B" -o "%ROOT%\CASIA-B-pkl" -n 8 -r 64 -d CASIAB
