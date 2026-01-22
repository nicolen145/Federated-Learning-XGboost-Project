$ErrorActionPreference = "Stop"

$KEYSDIR = "keys"
$APP_PATH = "."
$FEDERATION = "remote-federation"

if (!(Test-Path $KEYSDIR)) { throw "Missing keys directory: $KEYSDIR" }

$pubKeys = Get-ChildItem -Path $KEYSDIR -Filter "*.pub" -File | Sort-Object Name
if ($pubKeys.Count -eq 0) { throw "No .pub files found in $KEYSDIR" }

foreach ($k in $pubKeys) {
  Write-Host "Approving $($k.Name)"
  flwr supernode register $k.FullName $APP_PATH $FEDERATION
  if ($LASTEXITCODE -ne 0) { throw "Register failed for $($k.Name)" }
}

flwr supernode list $APP_PATH $FEDERATION
