#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory = $false, HelpMessage = "The action to perform. Can be 'start', 'stop', or 'status'.")]
    [ValidateSet("start", "stop", "status")]
    [string]$Action = "start"
)

$ErrorActionPreference = 'Stop'
$InformationPreference = 'Continue'

# --- Configuration ---
$ProjectRoot = $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"

$PythonExe = Join-Path (Join-Path $VenvPath "Scripts") "python.exe"

$BackendPort = 8000
$FrontendPort = 8501
$LogsDir = Join-Path $ProjectRoot "logs"

# --- Service Definitions ---
$Services = @(
    @{
        Name      = "Backend API"
        Port      = $BackendPort
        Command   = "& `"$PythonExe`" -m uvicorn src.api.main:app --host 0.0.0.0 --port $BackendPort"
        Url       = "http://localhost:$BackendPort"
        LogPrefix = "backend"
    },
    @{
        Name      = "Frontend UI"
        Port      = $FrontendPort
        Command   = "& `"$PythonExe`" -m streamlit run src/ui/app.py --server.port $FrontendPort"
        Url       = "http://localhost:$FrontendPort"
        LogPrefix = "frontend"
    }
)

# --- Functions ---

function Get-ProcessByPort {
    param([int]$Port)
    try {
        $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if ($connection) {
            return Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
        }
    }
    catch {
        Write-Warning "Could not get process for port $Port. Error: $_"
    }
    return $null
}

function Start-Services {
    Write-Host "--- Starting all services ---" -ForegroundColor White
    if (-not (Test-Path $LogsDir)) {
        New-Item -Path $LogsDir -ItemType Directory | Out-Null
    }

    foreach ($service in $Services) {
        $process = Get-ProcessByPort -Port $service.Port
        if ($process) {
            Write-Host "[$($service.Name)] is already RUNNING (Port: $($service.Port), PID: $($process.Id))" -ForegroundColor Cyan
        }
        else {
            Write-Host "Starting [$($service.Name)] on port $($service.Port)..."
            $logFile = Join-Path $LogsDir "$($service.LogPrefix).log"
            
            $job = Start-Job -Name "Service_$($service.LogPrefix)" -ScriptBlock {
                param($command, $logPath, $workingDir)

                # Chuyển hướng vào thư mục dự án trước khi chạy lệnh
                Set-Location -Path $workingDir
                
                # Ghi log output và error vào file
                $scriptblock = [scriptblock]::Create("$command *> `"$logPath`"")
                & $scriptblock
            } -ArgumentList $service.Command, $logFile, $ProjectRoot

            # Đợi 60 giây để các model AI (PaddleOCR, VietOCR) load xong vào RAM
            Start-Sleep -Seconds 60
            $newProcess = Get-ProcessByPort -Port $service.Port
            if ($newProcess) {
                 Write-Host "[$($service.Name)] STARTED successfully (Port: $($service.Port), PID: $($newProcess.Id))" -ForegroundColor Green
            } else {
                 Write-Error "[$($service.Name)] FAILED to start. Check logs for details: $logFile"
            }
        }
    }
}

function Stop-Services {
    Write-Host "--- Stopping all services ---" -ForegroundColor White
    foreach ($service in $Services) {
        $process = Get-ProcessByPort -Port $service.Port
        if ($process) {
            Write-Host "Stopping [$($service.Name)] (Port: $($service.Port), PID: $($process.Id))..."
            Stop-Process -Id $process.Id -Force
            Write-Host "[$($service.Name)] STOPPED." -ForegroundColor Green
        }
        else {
            Write-Host "[$($service.Name)] is not running." -ForegroundColor Yellow
        }
    }

    Get-Job -Name "Service_*" -ErrorAction SilentlyContinue | Stop-Job
    Get-Job -Name "Service_*" -ErrorAction SilentlyContinue | Remove-Job -Force
}

function Show-ServiceStatus {
    Write-Host "--- Service Status ---" -ForegroundColor White
    foreach ($service in $Services) {
        $process = Get-ProcessByPort -Port $service.Port
        if ($process) {
            Write-Host "[RUNNING] $($service.Name) - Port: $($service.Port), PID: $($process.Id), URL: $($service.Url)" -ForegroundColor Green
        }
        else {
            Write-Host "[STOPPED] $($service.Name) - Port: $($service.Port)" -ForegroundColor Yellow
        }
    }
}


# --- Main Logic ---

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python virtual environment not found at '$PythonExe'. Please run the setup script."
    exit 1
}

switch ($Action) {
    "start" {
        # Tự động dừng các service đang chạy trước khi bật mới
        Write-Host "Auto-stopping existing services before start..." -ForegroundColor Gray
        Stop-Services
        
        # Đợi 2 giây để đảm bảo hệ điều hành đã giải phóng hoàn toàn cổng (Port)
        Start-Sleep -Seconds 2
        
        Start-Services
    }
    "stop" {
        Stop-Services
    }
    "status" {
        Show-ServiceStatus
    }
}

Write-Host "-------------------------"
Show-ServiceStatus
Write-Host "Script finished."