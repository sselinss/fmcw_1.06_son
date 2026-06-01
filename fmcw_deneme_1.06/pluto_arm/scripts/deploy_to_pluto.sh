#!/bin/bash
# Pluto'ya kaynak dosyaları kopyala (SSH üzerinden)
# Kullanım: bash deploy_to_pluto.sh [pluto_ip]

PLUTO_IP="${1:-192.168.2.1}"
PLUTO_USER="root"
REMOTE_DIR="/root/fmcw_radar"

echo "Pluto'ya kopyalanıyor: ${PLUTO_USER}@${PLUTO_IP}:${REMOTE_DIR}"

ssh "${PLUTO_USER}@${PLUTO_IP}" "mkdir -p ${REMOTE_DIR}"

scp ../src/radar_config.py \
    ../src/acquisition.py  \
    ../src/dsp.py          \
    ../src/result_sender.py \
    ../src/main.py          \
    ../../configs/radar_params.yaml \
    "${PLUTO_USER}@${PLUTO_IP}:${REMOTE_DIR}/"

echo "Tamamlandı."
echo "Çalıştırmak için:"
echo "  ssh ${PLUTO_USER}@${PLUTO_IP} 'cd ${REMOTE_DIR} && python3 main.py'"
