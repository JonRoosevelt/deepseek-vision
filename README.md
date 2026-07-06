# DeepSeek Vision Bridge

Lightweight VLM service that adds image understanding to [pi](https://github.com/earendil-works/pi-coding-agent) when using DeepSeek (which lacks a vision API).

Runs [moondream2](https://github.com/vikhyatk/moondream) (0.5B params) locally on Apple Silicon MPS. Given a pasted image, it describes the contents so DeepSeek can "see" what you're showing it.

## How it works

```
┌──────────┐     image      ┌─────────────────┐    description     ┌──────────┐
│  pi TUI  │ ────────────── │  Vision Bridge   │ ────────────────  │ DeepSeek │
│ (paste)  │                │  (moondream2)    │                   │   V4 Pro │
└──────────┘                └─────────────────┘                   └──────────┘
                                  127.0.0.1:8901
```

1. You paste/read an image in pi
2. The pi extension intercepts the image, sends it to the local vision server
3. moondream2 describes the image in English
4. The description is injected into DeepSeek's context as text

## Install

### 1. Clone the repo

```bash
mkdir -p ~/.pi/agent
cd ~/.pi/agent
git clone https://github.com/jonathanmartins/deepseek-vision.git
```

### 2. Set up the Python environment

```bash
cd ~/.pi/agent/deepseek-vision
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Requires Python 3.10+ and Apple Silicon (MPS). On Intel Macs, it falls back to CPU.

### 3. Install the pi extension

Copy the extension file into pi's extensions folder:

```bash
cp ~/.pi/agent/deepseek-vision/pi-extension.ts ~/.pi/agent/extensions/deepseek-vision-bridge.ts
```

> The extension auto-starts the server on first image and kills it when pi exits. No manual server management needed.

### 4. Restart pi

Start a new pi session — the bridge is now active.

## Manual usage

If you want to run the server standalone:

```bash
cd ~/.pi/agent/deepseek-vision
./start.sh
```

Check health:

```bash
curl http://127.0.0.1:8901/health
# {"status":"ok","model":"vikhyatk/moondream2","device":"mps"}
```

Send an image for description:

```bash
curl -X POST http://127.0.0.1:8901/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moondream2",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image in detail."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }]
  }'
```

## Commands

In pi, type `/vision-status` to check if the server is running.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~1GB disk space (model download)
- ~2GB RAM during inference
