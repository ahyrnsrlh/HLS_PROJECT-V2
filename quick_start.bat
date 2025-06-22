@echo off
echo ===========================================
echo   SISTEM KLASIFIKASI SDG - QUICK START
echo ===========================================

echo Menjalankan setup otomatis...
python setup_and_run.py

echo.
echo Menjalankan demo prediksi SDG...
python predict_enhanced.py

echo.
echo ===========================================
echo   SELESAI! Tekan tombol apapun untuk keluar.
echo ===========================================
pause
