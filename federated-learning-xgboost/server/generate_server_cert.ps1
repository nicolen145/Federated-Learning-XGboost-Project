$ErrorActionPreference = "Stop"

# ---- PATHS ----
$CONF = ".\certificate.conf"
$CERTDIR = ".\certificates"

$SERVER_KEY = Join-Path $CERTDIR "server.key"
$SERVER_CSR = Join-Path $CERTDIR "server.csr"
$SERVER_PEM = Join-Path $CERTDIR "server.pem"
$CA_CRT = Join-Path $CERTDIR "ca.crt"
$CA_KEY = Join-Path $CERTDIR "ca.key"
# ----------------

foreach ($p in @($CONF, $SERVER_KEY, $CA_CRT, $CA_KEY)) {
  if (!(Test-Path $p)) { throw "Missing file: $p" }
}

Write-Host "Generating server.csr ..."
openssl req -new -key $SERVER_KEY -out $SERVER_CSR -config $CONF
if ($LASTEXITCODE -ne 0) { throw "openssl req failed" }

Write-Host "Signing server.pem ..."
openssl x509 -req -in $SERVER_CSR `
  -CA $CA_CRT -CAkey $CA_KEY -CAcreateserial `
  -out $SERVER_PEM -days 365 -sha256 `
  -extfile $CONF -extensions req_ext
if ($LASTEXITCODE -ne 0) { throw "openssl x509 failed" }

Write-Host "Done. Updated server.csr and server.pem"
