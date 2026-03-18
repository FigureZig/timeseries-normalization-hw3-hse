@echo off
set target=%1

if "%target%"=="install" goto install
if "%target%"=="test" goto test
if "%target%"=="run" goto run
if "%target%"=="analyze" goto analyze
if "%target%"=="clean" goto clean
if "%target%"=="all" goto all

echo Usage: make [install ^| test ^| run ^| analyze ^| clean ^| all]
exit /b 1

:install
pip install -r requirements.txt
exit /b 0

:test
python run_experiment.py --n_series 5
python analysis_results.py
exit /b 0

:run
python run_experiment.py
exit /b 0

:analyze
python analysis_results.py
exit /b 0

:clean
if exist "results" rmdir /s /q results
if exist "data\*.pkl" del /q "data\*.pkl"
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc >nul 2>&1
exit /b 0

:all
call :install
python run_experiment.py
python analysis_results.py
exit /b 0