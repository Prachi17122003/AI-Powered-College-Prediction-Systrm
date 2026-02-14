@echo off
echo Activating virtual environment...
call env\Scripts\activate.bat
echo Virtual environment activated!
echo.
echo To run the Django server, use: python manage.py runserver
echo To deactivate, use: deactivate
echo.
cmd /k
