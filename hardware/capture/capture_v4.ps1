# capture.ps1 -- record the measurement campaign from the NUCLEO board.
#
# Usage (from a folder with no square brackets in its path):
#     powershell -ExecutionPolicy Bypass -File capture.ps1
#     powershell -ExecutionPolicy Bypass -File capture.ps1 COM5 traces
#
# Writes one CSV per benchmark block into the output folder, named after the
# "#task," line the board emits, and stops when the board reports
# "#campaign_complete".
#
# Start this BEFORE pressing RESET on the board.

param(
    [string]$PortName = "",
    [string]$OutDir   = "traces"
)

$ErrorActionPreference = "Stop"

if ($PortName -eq "") {
    $ports = [System.IO.Ports.SerialPort]::GetPortNames()
    if ($ports.Count -eq 0) { Write-Host "No COM ports found."; exit 1 }
    $PortName = $ports[-1]
}

if (-not (Test-Path -LiteralPath $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}

$raw = Join-Path $OutDir "raw_capture.txt"

Write-Host "Port      : $PortName"
Write-Host "Output    : $OutDir"
Write-Host "Raw log   : $raw"
Write-Host ""
Write-Host "Opening port and releasing the board..."
Write-Host "Ctrl+C aborts. Stops by itself at #campaign_complete."
Write-Host ""

$p = New-Object System.IO.Ports.SerialPort $PortName,921600,'None',8,'One'
$p.ReadTimeout = 500
# At 115200 baud roughly 1150 bytes arrive per 100 ms. The default 4 KB
# receive buffer overflows whenever the read loop is delayed by console
# output or CPU contention, which silently truncates a campaign.
$p.ReadBufferSize = 1048576
$p.Open()
Start-Sleep -Milliseconds 300
$p.DiscardInBuffer()          # drop anything stale in the ST-LINK buffer
$p.Write("g")                 # release the board's start handshake
Write-Host "Sent start signal."


$rawWriter = New-Object System.IO.StreamWriter($raw, $false)
$rawWriter.AutoFlush = $false
$cur       = $null          # current per-task StreamWriter
$curName   = ""
$count     = 0
$buffer    = ""
$started   = Get-Date

try {
    while ($true) {
        $chunk = $p.ReadExisting()
        if (-not $chunk) { Start-Sleep -Milliseconds 20; continue }

        $rawWriter.Write($chunk)
        $buffer += $chunk

        # Process only whole lines; keep any partial tail in the buffer.
        while ($buffer.Contains("`n")) {
            $idx  = $buffer.IndexOf("`n")
            $line = $buffer.Substring(0, $idx).Trim("`r", " ")
            $buffer = $buffer.Substring($idx + 1)

            if ($line -eq "") { continue }

            if ($line.StartsWith("#task,")) {
                if ($cur) { $cur.Close() }
                $curName = $line.Substring(6).Trim()
                $path = Join-Path $OutDir "$curName.csv"
                $cur = New-Object System.IO.StreamWriter($path, $false)
                $cur.AutoFlush = $false
                $cur.WriteLine($line)
                $count = 0
                Write-Host ("[{0}] starting {1}" -f (Get-Date -Format HH:mm:ss), $curName)
                continue
            }

            if ($line -eq "#campaign_complete") {
                if ($cur) { $cur.Close(); $cur = $null }
                $el = [int]((Get-Date) - $started).TotalSeconds
                Write-Host ""
                Write-Host ("Campaign complete in {0}s. Files in {1}" -f $el, $OutDir)
                Get-ChildItem -LiteralPath $OutDir -Filter *.csv |
                    ForEach-Object { Write-Host ("  {0}  {1:N0} bytes" -f $_.Name, $_.Length) }
                return
            }

            if ($cur) {
                $cur.WriteLine($line)
                if (-not $line.StartsWith("#")) {
                    $count++
                    if ($count % 10000 -eq 0) {
                        if ($cur) { $cur.Flush() }
                        Write-Host ("    {0}: {1} samples" -f $curName, $count)
                    }
                }
            } else {
                Write-Host $line      # header lines before the first block
            }
        }
    }
}
finally {
    if ($cur) { $cur.Close() }
    $rawWriter.Close()
    if ($p.IsOpen) { $p.Close() }
}
