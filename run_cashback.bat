@echo off
REM Run CASHBACK_app.py using Anaconda Python
set ANACONDA=C:\Users\denis\Software\anaconda
set PYTHON=%ANACONDA%\python.exe
if not exist "%PYTHON%" set PYTHON=%ANACONDA%\Scripts\python.exe
if not exist "%PYTHON%" (
    echo Using conda run...
    "%ANACONDA%\Scripts\conda.exe" run -n base python "%~dp0CASHBACK_app.py"
) else (
    "%PYTHON%" "%~dp0CASHBACK_app.py"
)
pause
