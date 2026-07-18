@echo off
setlocal
rem Double-click to launch Floor Plan Studio with an available project Python.
cd /d "%~dp0"
if defined INTERIOR_PLAN_PYTHON if exist "%INTERIOR_PLAN_PYTHON%" (
  "%INTERIOR_PLAN_PYTHON%" plan2.py
  goto done
)
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" plan2.py
  goto done
)
if exist "hf_cache\venv\Scripts\python.exe" (
  "hf_cache\venv\Scripts\python.exe" plan2.py
  goto done
)
if exist "C:\SIA\Interior_design\hf_cache\venv\Scripts\python.exe" (
  "C:\SIA\Interior_design\hf_cache\venv\Scripts\python.exe" plan2.py
  goto done
)
python plan2.py
:done
pause
