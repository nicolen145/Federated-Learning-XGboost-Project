#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Usage
# -------------------------
if [[ $# -ne 1 ]]; then
  echo "Usage: ./run_supernode.sh <CLIENT_ID>"
  echo "Example: ./run_supernode.sh 2"
  exit 1
fi

CLIENT_ID="$1"

# -------------------------
# Fixed configuration
# -------------------------
# Your Server IP here
SUPERLINK_IP="x.x.x.x"
SUPERLINK_PORT="9192"
SUPERLINK="${SUPERLINK_IP}:${SUPERLINK_PORT}"

CERT_DIR="./certificates"
KEY_DIR="./keys"

CA_CERT="${CERT_DIR}/ca.crt"
PRIVATE_KEY="${KEY_DIR}/client_credentials_${CLIENT_ID}"

DATA_BASE="/home/archadmin/projects/my-app-xgboost/my_app_xgboost/datasets"

NUM_PARTITIONS=3
BASE_CLIENTAPPIO_PORT=9094

# -------------------------
# Per-client mapping
# -------------------------
case "$CLIENT_ID" in
  1)
    CSV_FILE="${DATA_BASE}/client0_20.csv"
    PARTITION_ID=0
    ;;
  2)
    CSV_FILE="${DATA_BASE}/client1_10.csv"
    PARTITION_ID=1
    ;;
  3)
    CSV_FILE="${DATA_BASE}/client2_70.csv"
    PARTITION_ID=2
    ;;
  *)
    echo "❌ Invalid CLIENT_ID: $CLIENT_ID (allowed: 1,2,3)"
    exit 1
    ;;
esac

CLIENTAPPIO_PORT=$((BASE_CLIENTAPPIO_PORT + PARTITION_ID))

# -------------------------
# Sanity checks
# -------------------------
[[ -f "$CA_CERT" ]] || { echo "❌ Missing CA cert: $CA_CERT"; exit 1; }
[[ -f "$PRIVATE_KEY" ]] || { echo "❌ Missing private key: $PRIVATE_KEY"; exit 1; }
[[ -f "$CSV_FILE" ]] || { echo "❌ Missing dataset: $CSV_FILE"; exit 1; }

# -------------------------
# Run SuperNode
# -------------------------
echo "🚀 Starting Flower SuperNode"
echo "  Client ID     : $CLIENT_ID"
echo "  SuperLink     : $SUPERLINK"
echo "  Partition     : $PARTITION_ID / $NUM_PARTITIONS"
echo "  ClientAppIO   : 0.0.0.0:${CLIENTAPPIO_PORT}"
echo "  Dataset       : $CSV_FILE"
echo

DATASET_PATH="$CSV_FILE" flower-supernode \
  --superlink "$SUPERLINK" \
  --root-certificates "$CA_CERT" \
  --node-config "partition-id=${PARTITION_ID} num-partitions=${NUM_PARTITIONS}" \
  --clientappio-api-address "0.0.0.0:${CLIENTAPPIO_PORT}" \
  --auth-supernode-private-key "$PRIVATE_KEY"
