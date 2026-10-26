@echo off
cd /d "%~dp0"
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
echo Installing requirements (first time takes a few minutes)...
pip install -r requirements.txt
streamlit run app.py
pause
