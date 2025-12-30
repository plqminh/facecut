@echo off
echo Starting FaceCut AI...
if exist venv\Scripts\python.exe (
    echo Using Virtual Environment...
    venv\Scripts\python.exe gui.py
) else (
    echo Using Global Python...
    python gui.py
)
pause
