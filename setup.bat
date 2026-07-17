@echo off
setlocal

set ENV_NAME=pymodaq_env
set PYTHON_VERSION=3.12

echo ============================================================
echo Installing environment %ENV_NAME%
echo ============================================================

echo.
echo Cleaning up leftover mamba/conda locks...
call mamba clean --locks

echo.
echo Creating environment %ENV_NAME% (Python %PYTHON_VERSION%)...
call mamba create -y -n %ENV_NAME% python=%PYTHON_VERSION%
if errorlevel 1 goto :error_create

echo.
echo Activating environment...
call conda activate %ENV_NAME%
if errorlevel 1 goto :error_activate

echo.
echo Installing PyQt from conda-forge...
call mamba install -y -c conda-forge pyqt
if errorlevel 1 goto :error_pyqt

echo.
echo Installing the plugin in editable mode (without pip dependencies)...
pip install -e . --no-deps
if errorlevel 1 goto :error_plugin

echo.
echo Installing remaining dependencies (bitstring, msl-loadlib, pipython)...
pip install bitstring msl-loadlib pipython
if errorlevel 1 goto :error_deps

echo.
echo ============================================================
echo Checking VISA communication
echo ============================================================
python -c "import pyvisa; rm = pyvisa.ResourceManager(); print('Detected instruments:', rm.list_resources())"
if errorlevel 1 goto :error_visa

echo.
echo ============================================================
echo Installation completed successfully.
echo To reuse this environment later:
echo     conda activate %ENV_NAME%
echo ============================================================
goto :eof

:error_create
echo.
echo ERROR while creating the environment.
echo If the message mentions a lockfile, rerun this script as administrator,
echo or manually delete C:\ProgramData\miniforge3\pkgs\pkgs.lock
goto :error_end

:error_activate
echo.
echo ERROR: could not activate the environment.
echo Run this script from a "Miniforge Prompt" (not a regular cmd.exe).
goto :error_end

:error_pyqt
echo.
echo ERROR while installing PyQt.
goto :error_end

:error_plugin
echo.
echo ERROR while installing the plugin (pip install -e . --no-deps).
echo Make sure this script is run from the plugin folder
echo (the one containing pyproject.toml).
goto :error_end

:error_deps
echo.
echo ERROR while installing the remaining dependencies.
goto :error_end

:error_visa
echo.
echo ERROR: PyVISA could not list any instruments.
echo Check that the NI-VISA drivers are properly installed (see README.md).
goto :error_end

:error_end
echo.
echo Installation aborted.
exit /b 1