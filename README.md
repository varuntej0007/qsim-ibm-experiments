# QSimEdge IBM Real Hardware Experiments

Real quantum circuit experiments from ARM edge devices (Raspberry Pi 5 + NVIDIA Jetson Orin Nano) connecting to IBM Quantum real superconducting processors.

## Platforms
- Raspberry Pi 5 8GB — home network, Hyderabad
- NVIDIA Jetson Orin Nano 8GB — college lab, Hyderabad

## IBM Quantum Backends
- ibm_torino (133Q Heron r1) — Pi 5
- ibm_marrakesh (156Q Heron r2) — Jetson
- ibm_kingston (156Q Heron r2) — Jetson validation

## Experiments
| Experiment | Noise % | Job ID |
|---|---|---|
| Bell State — ibm_torino | 11.62% | d74egjgqhmps73b41bf0 |
| GHZ State — ibm_torino | 5.08% | d74ej2dkoquc73e2m7og |
| Grover's — ibm_torino | 16.31% | d74ejngqhmps73b41edg |
| Bell State — ibm_marrakesh | 3.03% | d75oh468faus73f0bth0 |
| GHZ State — ibm_marrakesh | 5.86% | d75oh9e8faus73f0bt00 |
| Grover's — ibm_marrakesh | 96.2% target | d75ohfvq1anc738cm5k0 |
| Bell State — ibm_kingston | 2.34% | d75one23qcgc73fs1vn0 |
| GHZ State — ibm_kingston | 4.20% | d75onjlbjrds73ecn1jg |

All job IDs verifiable at quantum.ibm.com

## PQC Benchmarks
CRYSTALS-Dilithium (FIPS 204) vs SPHINCS+ (FIPS 205) vs RSA on ARM IoT edge hardware.

## Author
Miriyala Varun Tej — varuntej3126@gmail.com
B.Tech CSE (IoT), Malla Reddy University, 2026
