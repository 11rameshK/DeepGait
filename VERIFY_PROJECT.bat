@echo off
echo ==========================================
echo DeepGait Project Verification
echo ==========================================
python -m py_compile src\deepgait_model.py src\dataset.py src\losses.py
if errorlevel 1 goto :fail
python scripts\quick_verify_processed.py
if errorlevel 1 goto :fail
python scripts\test_model_shapes.py
if errorlevel 1 goto :fail
echo.
echo ALL CORE VERIFICATION CHECKS PASSED
goto :end
:fail
echo.
echo VERIFICATION FAILED
exit /b 1
:end
pause
