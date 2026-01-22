# Federated Learning with XGBoost - Internship Project

This project implements a secure Federated Learning (FL) pipeline using Flower and XGBoost.
It simulates a real-world distributed environment with a central server and multiple clients
running on separate machines, communicating over TLS and authenticated keys.

The project was developed as part of an internship at Altera Digital Health, conducted within the framework of the Starship academic internship program at Ben-Gurion University of the Negev, and focuses on secure communication in distributed systems, federated training with heterogeneous client data sizes, and a practical comparison between centralized and federated learning approaches in a real-world, healthcare-oriented setting.


---

## Project Structure
```text
FEDERATED-LEARNING-XGBOOST/
|
|-- client/
|   |-- certificates/                 TLS certificates used by the client (generated locally, not tracked)
|   |-- keys/                         Client authentication keys (generated locally, not tracked)
|   |-- my-app-xgboost/
|   |   |-- datasets/                 
|   |   |   |-- client0_20.csv        Local dataset used by the client 1
|   |   |   |-- client1_10.csv        Local dataset used by the client 2
|   |   |   |-- client2_70.csv        Local dataset used by the client 3
|   |   |   |-- diabetes_binary_5050split_health_indicators_BRFSS2015.csv     full dataset
|   |   |-- __init__.py               Marks the directory as a Python package
|   |   |-- client_app.py             Flower client implementation
|   |   |-- task.py                   Data loading, preprocessing, and XGBoost training logic
|   |   |-- preprocessing.py          Feature preprocessing utilities
|   |-- pyproject.toml                Client configuration (server address, TLS settings)
|   |-- run_supernodes.sh             Script to start a federated client node
|
|-- server/
|   |-- certificates/                 Server TLS certificates (generated locally, not tracked)
|   |-- keys/                         Public authentication keys of approved clients
|   |-- my-app-xgboost/
|   |   |-- client_app.py             Shared Flower application definitions
|   |   |-- server_app.py             Flower SuperLink server implementation
|   |   |-- task.py                   Server-side task and aggregation logic
|   |-- approve_supernodes.ps1        Script to approve federated client connections
|   |-- certificate.conf              OpenSSL configuration file (includes server IP in SAN)
|   |-- generate_auth_keys.sh         Generates authentication keys for clients
|   |-- generate_server_cert.ps1      Generates and signs TLS certificates for the server
|   |-- preprocessing.py              Shared preprocessing utilities
|   |-- pyproject.toml                Server configuration (remote federation, TLS)
|   |-- start_superlink.ps1           Starts the Flower SuperLink server on Windows
|
|-- .gitignore                         Excludes certificates, keys, and sensitive artifacts
```
---

## Dataset Description

The dataset used in this project is:
BRFSS 2015 – Diabetes Health Indicators  
File: `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`

This dataset was selected because:
- It represents a realistic healthcare-related binary classification task
- It contains structured, tabular features suitable for tree-based models such as XGBoost
- It is commonly used for benchmarking machine learning models
- It allows controlled splitting and distribution across federated clients

Dataset source:
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?resource=download

---

## Dataset Splitting and Federated Setup

The full dataset was intentionally split into three parts with different sizes:
- 10%
- 20%
- 70%

Each split represents a different federated client.

## Important Privacy Guarantee

- The raw data never leaves the organization or the client machine
- Each client trains locally on its own private dataset
- Only model updates are sent to the central server
- No raw samples, features, or labels are transmitted at any stage

---

## ClearML Integration

The server component is integrated with ClearML for experiment tracking and logging.

- All training metrics, logs, and artifacts are sent from the server to ClearML
- ClearML must be configured on the server machine
- Client machines do not interact directly with ClearML

---

## Prerequisites

- Python 3.x
- OpenSSL
- Flower
- XGBoost
- ClearML (server machine only)
- Linux (clients) / Windows (server)

---

## Configuration Notes (Server IP)

Before running the system, the server IP address must be updated manually in all of
the following locations:

1) `certificate.conf` - Update IP.4 before generating server certificates

2) `pyproject.toml` (server and clients) - Update the remote federation address (host:port)

3) `run_supernodes.sh` (clients) - Update the server hostname or IP used for connection

---

## Certificate and Key Management
### TLS Certificates
TLS Certificates are generated on the server machine using `generate_server_cert.ps1`.

After running this script, the following files are created in `server/certificates/`:
- server.key          (server private key – pre-existing or reused)
- server.csr          (certificate signing request – generated)
- server.pem          (signed server certificate – generated)
- ca.crt              (CA public certificate – pre-existing)
- ca.key              (CA private key – pre-existing)

### Authentication Keys
Authentication keys are generated using `generate_auth_keys.sh`.

After running this script, the following files are created:
- Client private authentication keys (one per client)
- Client public authentication keys (one per client)

### Manual distribution is required:
- Client private keys ( must be copied manually to `client/keys/` on the corresponding client machine
- Client public keys (*.pub) must be copied manually to `server/keys/` on the server machine
- Each client machine must contain a copy of the `ca.crt` file in its certificates directory in order to verify the server’s TLS certificate.

IMPORTANT:
- This distribution step is manual and not automated
- None of the generated certificates or keys are included in the repository
- All certificate and key files are excluded via `.gitignore`

---

## How to Run the Project

Step 1 - Generate Server Certificates (Server Machine)  
Run `generate_server_cert.ps1`

Update the server IP address in `certificate.conf` before running

Step 2 - Generate Authentication Keys  
Run `generate_auth_keys.sh`  

Manually distribute the generated keys to the server and client machines

Step 3 - Start the Server (Windows)  
Run `start_superlink.ps1`  

Ensure ClearML is configured on the server

Step 4 - Approve Client Connections  
Run `approve_supernodes.ps1` on the server machine

Step 5 - Start Clients  
Run `run_supernodes.sh <client id>` on each client machine 

Each client runs using its local dataset only

Step 6 - Start the federated learning process

Once all clients are connected and approved, start the federated learning process from the server by running: `flwr run . remote-federation`

After the run completes, open the ClearML UI and inspect the experiment results,
metrics, and saved artifacts.

---

## Experimental Design Notes

The project intentionally uses non-uniform data splits across clients to simulate
heterogeneous real-world environments and analyze the effect of data size on
federated convergence and performance.

---

## Notes

- No certificates, keys, or sensitive data are included in this repository
- All security-related artifacts are generated and distributed manually
- ClearML integration is handled exclusively on the server side
- The repository reflects the actual execution structure used during development

---
## Project Presentation

This repository also includes a project presentation that provides a high-level overview of the work.

The presentation explains:
- The motivation for using Federated Learning in this project
- The overall system architecture (server–client setup)
- How Flower and XGBoost are integrated in a federated setting
- The data distribution strategy across clients (non-uniform splits)
- Key design decisions related to privacy, security, and deployment
- A comparison between centralized and federated learning approaches
