#!/bin/bash

while true; do
        echo "[*] Starting detection..."
        python3 motion_ident.py
        echo "[!] Detection stopped, restarting..."
done
