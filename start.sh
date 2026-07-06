#!/bin/bash
# Start the DeepSeek Vision Bridge server
# This runs moondream2 locally on Apple Silicon MPS to describe images
# so DeepSeek V4 Pro (which lacks vision API) can understand pasted images.

cd "$(dirname "$0")"
exec .venv/bin/python3 vision_server.py
