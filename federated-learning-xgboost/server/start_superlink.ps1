$ErrorActionPreference = "Stop"

$CERTDIR = ".\certificates"
$CA  = Join-Path $CERTDIR "ca.crt"
$CRT = Join-Path $CERTDIR "server.pem"
$KEY = Join-Path $CERTDIR "server.key"

foreach ($p in @($CA, $CRT, $KEY)) {
  if (!(Test-Path $p)) { throw "Missing file: $p" }
}

# Ensure venv Scripts is in PATH so SuperLink can spawn its internal executables (e.g., flower-superexec)
$VenvScripts = (Resolve-Path ".\.venv\Scripts").Path
$env:Path = "$VenvScripts;$env:Path"

# (Optional) sanity check
if (!(Test-Path (Join-Path $VenvScripts "flower-superexec.exe"))) {
  Write-Host "WARNING: flower-superexec.exe not found in .venv\Scripts. SuperLink may fail in subprocess mode."
}

Write-Host "Starting SuperLink (server IP: 10.160.8.2)"
Write-Host " Fleet      : 0.0.0.0:9192"
Write-Host " ServerAppIO: 0.0.0.0:9191"
Write-Host " Control    : 0.0.0.0:9193"

.\.venv\Scripts\flower-superlink.exe `
  --ssl-ca-certfile $CA `
  --ssl-certfile    $CRT `
  --ssl-keyfile     $KEY `
  --enable-supernode-auth `
  --fleet-api-address 0.0.0.0:9192 `
  --serverappio-api-address 0.0.0.0:9191 `
  --control-api-address 0.0.0.0:9193
