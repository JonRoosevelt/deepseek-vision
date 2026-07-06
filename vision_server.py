"""
Lightweight VLM service for pi image → text preprocessing.
Runs moondream2 (0.5B params) on Apple Silicon MPS.
Exposes an OpenAI-compatible /v1/chat/completions endpoint
so pi can be configured as a custom provider.

Usage:
    .venv/bin/python vision_server.py
"""

import base64
import io
import logging
import time
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vision-server")

MODEL_ID = "vikhyatk/moondream2"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

model: Optional[AutoModelForCausalLM] = None
tokenizer: Optional[AutoTokenizer] = None


def load_model():
    global model, tokenizer
    logger.info(f"Loading {MODEL_ID} on {DEVICE}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    logger.info("Model loaded.")


def decode_image_from_base64_uri(uri: str) -> Image.Image:
    """Extract base64 data from data:image/...;base64,... URI."""
    if "," not in uri:
        raise ValueError("Invalid data URI, no comma found")
    b64 = uri.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def describe_image(image: Image.Image, question: str) -> str:
    """Run moondream2 caption/query on the image."""
    enc = model.encode_image(image)
    return model.answer_question(enc, question, tokenizer)


# --- OpenAI-compatible schemas ---

class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[dict] = None


class Message(BaseModel):
    role: str
    content: str | list[ContentPart]


class ChatRequest(BaseModel):
    model: str = "moondream2"
    messages: list[Message]
    max_tokens: int = 512
    temperature: float = 0.0


class Choice(BaseModel):
    index: int
    message: dict
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str = "vision-bridge"
    object: str = "chat.completion"
    created: int = 0
    model: str = "moondream2"
    choices: list[Choice]
    usage: Usage = Usage()


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="DeepSeek Vision Bridge", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    # Find the last user message
    last_msg = None
    for msg in reversed(req.messages):
        if msg.role == "user":
            last_msg = msg
            break

    if last_msg is None:
        raise HTTPException(status_code=400, detail="No user message found")

    # Parse content for image + text
    text_parts = []
    images = []

    content = last_msg.content
    if isinstance(content, str):
        text_parts.append(content)
    else:
        for part in content:
            if part.type == "text" and part.text:
                text_parts.append(part.text)
            elif part.type == "image_url" and part.image_url:
                try:
                    img = decode_image_from_base64_uri(part.image_url["url"])
                    images.append(img)
                except Exception as e:
                    logger.error(f"Failed to decode image: {e}")

    if not images:
        raise HTTPException(status_code=400, detail="No image found in message")

    prompt = " ".join(text_parts).strip() if text_parts else "Describe this image in detail."

    try:
        start = time.time()
        description = describe_image(images[0], prompt)
        elapsed = time.time() - start
        logger.info(f"Inference took {elapsed:.2f}s")

        return ChatResponse(
            id=f"vision-{int(start)}",
            created=int(start),
            choices=[Choice(index=0, message={"role": "assistant", "content": description})],
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8901, log_level="info")
