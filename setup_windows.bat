@echo off
rem Effect Factory - one-time setup for Windows.
rem Double-click this file. It prepares a private Python environment in .venv
rem and puts an "Effect Factory" icon on the desktop and in the Start menu.
rem Run it again after moving this folder. "setup_windows.bat --remove" removes the icons.
setlocal
cd /d "%~dp0"
title Effect Factory setup
echo.
echo   Effect Factory setup
echo   ====================
echo.

if /i "%~1"=="--remove" goto remove

rem --- find Python 3.10+ with Tkinter ---------------------------------------
:findpython
set "PY="
call :trypy py -3
if not defined PY call :trypy python
if not defined PY call :trypy python3
if not defined PY for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do if not defined PY call :trypy "%%D\python.exe"
if defined PY goto havepython
if defined TRIED_WINGET goto installed

echo Python 3.10 or later was not found.
where winget >nul 2>nul
if errorlevel 1 goto nopython
choice /C YN /M "Install Python 3.12 now with winget"
if errorlevel 2 goto nopython
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
set "TRIED_WINGET=1"
goto findpython

:installed
echo.
echo Python has been installed. Please double-click setup_windows.bat once more.
pause
exit /b 0

:nopython
echo Please install Python from https://www.python.org/downloads/windows/
echo (keep "tcl/tk and IDLE" selected), then double-click setup_windows.bat again.
start "" "https://www.python.org/downloads/windows/"
pause
exit /b 1

:havepython
rem --- private environment ------------------------------------------------------
if not exist ".venv\Scripts\python.exe" goto makevenv
".venv\Scripts\python.exe" -c "import sys, tkinter" >nul 2>nul
if not errorlevel 1 goto venvready
:makevenv
echo Creating a private Python environment (.venv) ...
%PY% -m venv --clear .venv
if errorlevel 1 goto fail
:venvready
set "VPY=.venv\Scripts\python.exe"
echo Installing numpy and Pillow ...
"%VPY%" -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 goto fail

rem --- desktop and Start menu shortcuts ---------------------------------------
echo.
"%VPY%" tools\install_desktop.py
if errorlevel 1 goto fail

rem --- ffmpeg (optional: MP4 / MOV export) ------------------------------------
"%VPY%" -c "import sys; sys.path.insert(0, '.'); from efx.export import find_ffmpeg; sys.exit(0 if find_ffmpeg() else 1)"
if not errorlevel 1 goto launch
echo.
echo ffmpeg was not found. It is needed for MP4 / MOV export (PNG sequences work without it).
where winget >nul 2>nul
if errorlevel 1 goto launch
choice /C YN /M "Install ffmpeg now with winget"
if errorlevel 2 goto launch
winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements

:launch
echo.
choice /C YN /M "Start Effect Factory now"
if errorlevel 2 goto end
start "" ".venv\Scripts\pythonw.exe" "%~dp0effect_factory.py"
goto end

:remove
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" tools\install_desktop.py --remove
) else (
  echo Nothing to remove: setup has not been run in this folder.
)
goto end

:fail
echo.
echo Setup failed. Please check the messages above.
pause
exit /b 1

:end
echo.
pause
exit /b 0

rem --- helper: use %* as Python when it is 3.10+ and has Tkinter ----------------
:trypy
%* -c "import sys, tkinter; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=%*"
exit /b 0
