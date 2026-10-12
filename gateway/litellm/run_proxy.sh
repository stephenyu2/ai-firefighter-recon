#!/bin/bash
# Starts the gateway on 127.0.0.1:4000. Needs config.yaml (from config.example.yaml) and .env.local (from .env.example).
set -e
cd "$(dirname "$0")"
set -a; source .env.local; set +a
exec litellm --config config.yaml --host 127.0.0.1 --port 4000 --num_workers 1
