# bgremove-native first-time setup
# 1. uv sync (Python deps)
# 2. ffmpeg.exe + ffprobe.exe → bin/
# 3. RVM ONNX models → models/

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Write-Step($num, $total, $msg) {
    Write-Host ""
    Write-Host "[$num/$total] $msg" -ForegroundColor Cyan
}

function Write-Ok($msg) { Write-Host "  OK $msg" -ForegroundColor Green }
function Write-Skip($msg) { Write-Host "  ~  $msg (already present, skipping)" -ForegroundColor DarkGray }
function Write-Fail($msg) { Write-Host "  X  $msg" -ForegroundColor Red }

Write-Host "=== bgremove-native setup ===" -ForegroundColor White

# ---------- 1. uv sync ----------
Write-Step 1 3 "Installing Python dependencies (uv sync)..."
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    Write-Fail "uv not found in PATH."
    Write-Host "  Install with: winget install astral-sh.uv" -ForegroundColor Yellow
    exit 1
}
uv sync
if ($LASTEXITCODE -ne 0) {
    Write-Fail "uv sync failed."
    exit 1
}
Write-Ok "Python deps installed."

# ---------- 2. ffmpeg ----------
Write-Step 2 3 "Setting up ffmpeg / ffprobe..."
New-Item -ItemType Directory -Force -Path "bin" | Out-Null
$ffmpegExe  = Join-Path $PSScriptRoot "bin\ffmpeg.exe"
$ffprobeExe = Join-Path $PSScriptRoot "bin\ffprobe.exe"

if ((Test-Path $ffmpegExe) -and (Test-Path $ffprobeExe)) {
    Write-Skip "ffmpeg.exe + ffprobe.exe"
} else {
    $zipUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    $zipPath = Join-Path $env:TEMP "bgremove_ffmpeg.zip"
    $extractPath = Join-Path $env:TEMP "bgremove_ffmpeg_extract"

    Write-Host "  Downloading $zipUrl ..."
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing

    Write-Host "  Extracting..."
    if (Test-Path $extractPath) { Remove-Item -Recurse -Force $extractPath }
    Expand-Archive -Path $zipPath -DestinationPath $extractPath

    $foundBin = Get-ChildItem -Path $extractPath -Recurse -Directory -Filter "bin" | Select-Object -First 1
    if (-not $foundBin) {
        Write-Fail "ffmpeg bin/ folder not found inside the zip."
        exit 1
    }
    Copy-Item -Path (Join-Path $foundBin.FullName "ffmpeg.exe")  -Destination $ffmpegExe  -Force
    Copy-Item -Path (Join-Path $foundBin.FullName "ffprobe.exe") -Destination $ffprobeExe -Force

    Remove-Item -Recurse -Force $extractPath
    Remove-Item -Force $zipPath
    Write-Ok "ffmpeg.exe + ffprobe.exe placed in bin/"
}

# ---------- 3. RVM models ----------
Write-Step 3 3 "Downloading RVM ONNX models..."
New-Item -ItemType Directory -Force -Path "models" | Out-Null
$rvmBase = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0"
$rvmModels = @(
    @{ Name = "rvm_resnet50_fp32.onnx";     Size = 108MB },
    @{ Name = "rvm_mobilenetv3_fp32.onnx";  Size = 15MB }
)

foreach ($m in $rvmModels) {
    $dest = Join-Path $PSScriptRoot ("models\" + $m.Name)
    if (Test-Path $dest) {
        Write-Skip $m.Name
        continue
    }
    Write-Host ("  Downloading {0} (~{1:N0} MB)..." -f $m.Name, ($m.Size / 1MB))
    $url = "$rvmBase/$($m.Name)"
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
    Write-Ok $m.Name
}

# rembg models are downloaded on demand to %USERPROFILE%\.u2net\

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host "Run launch.bat to start the app." -ForegroundColor White
