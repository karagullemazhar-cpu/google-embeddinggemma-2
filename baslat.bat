@echo off
chcp 65001 >nul
title EmbeddingGemma 2 - Cok Modlu Arama (Port 5102)
call :main %*
set EC=%ERRORLEVEL%
if %EC% equ 0 exit /b 0
echo.
echo [Cikis kodu %EC% - pencereyi kapatmak icin bir tusa basin...]
pause >nul
exit /b %EC%

:main
echo ==========================================================
echo    EMBEDDINGGEMMA 2 - COK MODLU ARAMA
echo    Windows native calisir - WSL gerekmez
echo ==========================================================
echo.
cd /d "%~dp0"
set LOG=%~dp0baslat.log
echo [%date% %time%] baslat.bat basladi, klasor=%CD% > "%LOG%"

REM --- 1) Eski 5102 sureci varsa OTOMATIK KAPAT ---
REM     (eski kod acik kalirsa arayuz yeni, backend eski olur)
netstat -ano | findstr ":5102" | findstr "LISTENING" >nul 2>nul
if errorlevel 1 goto :noold
echo [*] 5102 portunda eski sunucu bulundu, kapatiliyor...
echo [%date% %time%] eski surec kapatiliyor >> "%LOG%"
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":5102" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%p >nul 2>nul
)
timeout /t 1 /nobreak >nul
netstat -ano | findstr ":5102" | findstr "LISTENING" >nul 2>nul
if not errorlevel 1 (
    echo [!] Port 5102 hala dolu - elle kapatin: taskkill /F /IM python.exe
    exit /b 1
)
echo [*] Eski sunucu kapatildi, guncel surum baslatilacak.
:noold

REM --- 2) Bu klasorde calisan YETIM app.py sureclerini de kapat ---
REM     (eski kodlu surec capraz istek alirsa garip hatalar uretir)
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*google-embeddinggemma-2*app.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>nul
timeout /t 1 /nobreak >nul

REM --- 2) Windows python bul ---
set PY=
where py >nul 2>nul
if not errorlevel 1 set PY=py -3
if not "%PY%"=="" goto :havepy
where python >nul 2>nul
if not errorlevel 1 set PY=python
if "%PY%"=="" goto :nopython
:havepy
echo [*] Python bulundu (%PY%), Windows native baslatiliyor. >> "%LOG%"
goto :pyok
:nopython
echo [!] Windows python bulunamadi. >> "%LOG%"
echo [!] Windows python bulunamadi.
echo [!] https://www.python.org/downloads/ adresinden Python 3.11+ kurun.
echo [!] Kurulumda "Add python.exe to PATH" secenegini isaretleyin.
where wsl >nul 2>nul
if errorlevel 1 exit /b 1
echo [*] WSL ile deneniyor... >> "%LOG%"
for /f "delims=" %%p in ('wsl wslpath -a "%~dp0" 2^>nul') do set WSLDIR=%%p
if "%WSLDIR%"=="" set WSLDIR=/mnt/c/Users/Mazhar/Documents/google-embeddinggemma-2
start "" cmd /c "timeout /t 6 >nul & start http://localhost:5102"
wsl bash -c "cd '%WSLDIR%' && { [ -x .venv/bin/python ] && .venv/bin/python app.py || python3 app.py; }"
set EXITCODE=%ERRORLEVEL%
echo [%date% %time%] wsl sunucu kapandi, kod=%EXITCODE% >> "%LOG%"
if %EXITCODE% equ 130 exit /b 0
if %EXITCODE% neq 0 (
    echo.
    echo [!] Sunucu beklenmedik sekilde kapandi.
)
exit /b %EXITCODE%
:pyok

REM --- 3) Sanal ortam ---
if not exist "winvenv\Scripts\python.exe" (
    echo [*] winvenv kuruluyor, ilk sefer biraz surer... >> "%LOG%"
    echo [*] winvenv sanal ortami kuruluyor, ilk sefer biraz surer.
    %PY% -m venv winvenv
    if errorlevel 1 (
        echo [!] venv kurulamadi. >> "%LOG%"
        echo [!] venv kurulamadi.
        exit /b 1
    )
)

REM --- 4) Bagimliliklar (eksik veya surum kilidi degismis ise) ---
winvenv\Scripts\python.exe -c "import flask, sentence_transformers, torchvision" >nul 2>nul
if errorlevel 1 goto :pipinstall
if not exist "winvenv\.deps_ok_v4" goto :pipinstall
goto :depsok
:pipinstall
echo [*] Bagimliliklar kuruluyor, ilk sefer buyuk indirme torch+model... >> "%LOG%"
echo [*] Bagimliliklar kuruluyor... ilk sefer uzun surer, torch yaklasik 2.5 GB.
winvenv\Scripts\python.exe -m pip install --upgrade pip
winvenv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo [!] pip kurulumu basarisiz. >> "%LOG%"
    echo [!] pip kurulumu basarisiz. Interneti ve requirements.txt dosyasini kontrol edin.
    exit /b 1
)
winvenv\Scripts\python.exe -c "import flask, sentence_transformers" >nul 2>nul
if errorlevel 1 (
    echo [!] Gerekli paketler yine de ice aktarilamadi. >> "%LOG%"
    echo [!] Gerekli paketler yine de ice aktarilamadi.
    exit /b 1
)
echo ok > "winvenv\.deps_ok_v4"
:depsok

REM --- 5) Arayuz dosyalari ---
if not exist "index.html" (
    echo [!] index.html bulunamadi. Klasoru kontrol edin: %CD% >> "%LOG%"
    echo [!] index.html bulunamadi. Klasoru kontrol edin: %CD%
    exit /b 1
)
if not exist "app.py" (
    echo [!] app.py bulunamadi. >> "%LOG%"
    echo [!] app.py bulunamadi.
    exit /b 1
)

echo [*] Klasor : %CD%
echo [*] Sunucu : http://localhost:5102
echo [*] Tarayici 6 saniye sonra otomatik acilacak.
echo [*] Kapatmak icin bu pencerede CTRL + C.
echo.
start "" cmd /c "timeout /t 6 >nul & start http://localhost:5102"

echo [%date% %time%] sunucu baslatiliyor (Windows native, surum v3)... >> "%LOG%"
winvenv\Scripts\python.exe app.py
set EXITCODE=%ERRORLEVEL%
echo [%date% %time%] sunucu kapandi, kod=%EXITCODE% >> "%LOG%"
if %EXITCODE% equ 130 exit /b 0
if %EXITCODE% neq 0 (
    echo.
    echo [!] Sunucu beklenmedik sekilde kapandi.
    echo [!] Ayrinti icin: %LOG%
)
exit /b %EXITCODE%
