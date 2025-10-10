@echo off
REM Install Models Script for Windows
REM Installs COLMAP and other required models

echo ============================================================
echo 3D MEASUREMENT SYSTEM - MODEL INSTALLATION
echo ============================================================
echo.

echo This script will install:
echo   1. pycolmap (COLMAP Python bindings)
echo   2. timm (additional depth models)
echo   3. einops (performance enhancement)
echo.

pause

echo.
echo ============================================================
echo Installing pycolmap...
echo ============================================================
python -m pip install pycolmap
if %errorlevel% neq 0 (
    echo [WARNING] pycolmap installation failed
    echo You may need to install COLMAP binary manually
    echo Download from: https://github.com/colmap/colmap/releases
) else (
    echo [OK] pycolmap installed successfully
)

echo.
echo ============================================================
echo Installing timm (optional)...
echo ============================================================
python -m pip install timm
if %errorlevel% neq 0 (
    echo [WARNING] timm installation failed
) else (
    echo [OK] timm installed successfully
)

echo.
echo ============================================================
echo Installing einops (optional)...
echo ============================================================
python -m pip install einops
if %errorlevel% neq 0 (
    echo [WARNING] einops installation failed
) else (
    echo [OK] einops installed successfully
)

echo.
echo ============================================================
echo Verifying Installation...
echo ============================================================
python -c "import pycolmap; print('[OK] pycolmap version:', pycolmap.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] pycolmap not found
) else (
    echo [OK] pycolmap verified
)

python -c "import timm; print('[OK] timm version:', timm.__version__)" 2>nul
python -c "import einops; print('[OK] einops installed')" 2>nul

echo.
echo ============================================================
echo Running Model Check...
echo ============================================================
python check_models.py

echo.
echo ============================================================
echo INSTALLATION COMPLETE
echo ============================================================
echo.
echo Next steps:
echo   1. Run: python validate_system.py
echo   2. Run: python main.py info
echo   3. Test: python main.py measure img1.jpg img2.jpg img3.jpg
echo.

pause

