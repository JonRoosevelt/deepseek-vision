#!/bin/bash
# Start the DeepSeek Vision Bridge server
# Runs moondream2 (~1.8B params) locally on Apple Silicon MPS.
# Exposes OpenAI-compatible /v1/chat/completions on port 8901
# so pi can use it as a custom provider for image→text preprocessing.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/.venv/bin/python3" "${SCRIPT_DIR}/vision_server.py"
