"""
Lightweight VLM service for pi image → text preprocessing.
Runs moondream2 (~1.8B params) on Apple Silicon MPS.
Exposes an OpenAI-compatible /v1/chat/completions endpoint
so pi can be configured as a custom provider.

Features:
  - Embedding cache: reuses encode_image() results for same image across
    multi-turn conversations (keyed by SHA256 hash). Cuts follow-up
    latency from ~2s to ~300ms.
  - Structured VQA: prefix "[structured]" in the prompt to get JSON
    output with bounding boxes, raw OCR text, and spatial layout.
  - Multi-image: processes all images in a message, returning combined
    results (max 10).
  - Stats: /health returns cache hit rate and avg inference time.
  - Memory management: auto-eviction under pressure, MPS cache clearing,
    aggressive GC between images.

Usage:
    .venv/bin/python vision_server.py
"""

import base64
import gc
import hashlib
import io
import logging
import resource
import time
from threading import Lock
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
# Small cache: each embedding tensor is 2-4 MB. 8 entries = ~32 MB max.
# Tight cap prevents cache from eating RAM on multi-image sessions.
MAX_CACHE_SIZE = 8
MAX_IMAGES_PER_REQUEST = 10

model: Optional[AutoModelForCausalLM] = None
tokenizer: Optional[AutoTokenizer] = None

# --- Embedding cache ---
# Keyed by SHA256 of the raw base64 bytes. Only caches encode_image()
# results (the image embeddings), not the final text answer.
_embedding_cache: dict[str, torch.Tensor] = {}
_cache_lock = Lock()
_cache_hits = 0
_cache_misses = 0
_total_inferences = 0
_total_inference_time = 0.0


def _get_memory_mb() -> int:
    """Get current process RSS in MB (macOS ru_maxrss is in bytes)."""
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 * 1024)
    except Exception:
        return 0


def _cleanup_memory():
    """Force garbage collection and clear MPS cache.

    MPS (Metal Performance Shaders) on Apple Silicon does NOT auto-release
    tensor memory promptly. After processing even 2-3 images, stale
    allocations accumulate and the process balloons to 7+ GB.

    This must be called after every inference batch.
    """
    gc.collect()
    if DEVICE == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _hash_image(img_bytes: bytes) -> str:
    return hashlib.sha256(img_bytes).hexdigest()


def _cache_stats() -> dict:
    with _cache_lock:
        total = _cache_hits + _cache_misses
        hit_rate = _cache_hits / total if total > 0 else 0.0
        avg_time = _total_inference_time / _total_inferences if _total_inferences > 0 else 0.0
        return {
            "cache_hits": _cache_hits,
            "cache_misses": _cache_misses,
            "cache_size": len(_embedding_cache),
            "cache_max": MAX_CACHE_SIZE,
            "hit_rate": round(hit_rate, 3),
            "total_inferences": _total_inferences,
            "avg_inference_seconds": round(avg_time, 3),
            "memory_mb": _get_memory_mb(),
        }


def _check_memory_pressure():
    """If process RSS exceeds 15 GB, evict oldest cache entries.

    With 24 GB total system RAM and moondream2 using ~8 GB for the model,
    we need to keep overhead under ~15 GB to avoid swapping.
    """
    with _cache_lock:
        while len(_embedding_cache) > 1:  # never evict to zero
            if _get_memory_mb() < 15_000:
                break
            oldest = next(iter(_embedding_cache))
            del _embedding_cache[oldest]
            logger.warning(f"Memory pressure — evicted cache entry {oldest[:12]}...")
            gc.collect()
            if DEVICE == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass


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
    _cleanup_memory()
    logger.info(f"Model loaded. Memory: {_get_memory_mb()} MB.")


STRUCTURED_PREFIX = "[structured]"
STRUCTURED_SYSTEM_PROMPT = (
    "You are a precise visual analysis tool. Describe this image in a structured format.\n"
    "Output a JSON object with:\n"
    '  "summary": 2-3 sentence overall description,\n'
    '  "elements": list of distinct UI elements or objects you see (name only, no duplicates),\n'
    '  "text": visible text strings you can read (each item once, no duplicates),\n'
    '  "layout": spatial arrangement in 1-2 sentences (e.g. nav bar top, main content center),\n'
    '  "colors": dominant colors present.\n'
    "For code screenshots, include a \"code\" field with the transcribed code.\n"
    "IMPORTANT: Each text item must appear only ONCE. Do not repeat the same string.\n"
    "Output ONLY valid JSON, no markdown fences, no trailing commas."
)


def decode_image_from_base64_uri(uri: str) -> tuple[Image.Image, bytes]:
    """Extract base64 data from data:image/...;base64,... URI.
    Returns (PIL Image, raw bytes) for caching purposes."""
    if "," not in uri:
        raise ValueError("Invalid data URI, no comma found")
    b64 = uri.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)), raw


def _get_or_encode(image: Image.Image, image_hash: Optional[str]) -> tuple:
    """Get cached embeddings or encode the image. Returns (enc, encode_time, cache_hit)."""
    global _cache_hits, _cache_misses

    # Try cache first
    if image_hash:
        with _cache_lock:
            cached = _embedding_cache.get(image_hash)
        if cached is not None:
            _cache_hits += 1
            logger.debug(f"Cache hit for {image_hash[:12]}...")
            return cached, 0.0, True

    # Encode
    t0 = time.time()
    enc = model.encode_image(image)
    encode_time = time.time() - t0
    _cache_misses += 1

    if image_hash:
        with _cache_lock:
            if len(_embedding_cache) >= MAX_CACHE_SIZE:
                oldest = next(iter(_embedding_cache))
                del _embedding_cache[oldest]
                logger.debug(f"Cache full — evicted {oldest[:12]}...")
            _embedding_cache[image_hash] = enc
        logger.debug(f"Cache miss for {image_hash[:12]}... (encode: {encode_time:.2f}s)")

    return enc, encode_time, False


def describe_image(image: Image.Image, question: str, image_hash: Optional[str] = None) -> tuple[str, float, bool]:
    """Run moondream2 answer_question on the image.

    Uses embedding cache if image_hash is provided. Returns (answer, elapsed, cache_hit).
    """
    global _total_inferences, _total_inference_time

    enc, encode_time, cache_hit = _get_or_encode(image, image_hash)

    t1 = time.time()
    answer = model.answer_question(enc, question, tokenizer)
    answer_time = time.time() - t1

    total_time = encode_time + answer_time
    _total_inferences += 1
    _total_inference_time += total_time

    logger.info(f"Inference (VQA): {'cache hit' if cache_hit else 'cache miss'} -> "
                f"{total_time:.2f}s (encode: {encode_time:.2f}s, answer: {answer_time:.2f}s)")

    return answer, total_time, cache_hit


def caption_image(image: Image.Image, length: str = "normal", image_hash: Optional[str] = None) -> tuple[str, float, bool]:
    """Run moondream2 caption (no question, just describe).

    Uses embedding cache. Returns (caption, elapsed, cache_hit).
    Caption is cleaner than answer_question for general descriptions.
    """
    global _total_inferences, _total_inference_time

    enc, encode_time, cache_hit = _get_or_encode(image, image_hash)

    t1 = time.time()
    raw = model.caption(enc, length=length)  # type: ignore[arg-type]
    caption_time = time.time() - t1

    # caption() returns a dict like {"caption": "..."}
    if isinstance(raw, dict):
        caption = raw.get("caption", str(raw))
    else:
        caption = str(raw)

    total_time = encode_time + caption_time
    _total_inferences += 1
    _total_inference_time += total_time

    logger.info(f"Inference (caption/{length}): {'cache hit' if cache_hit else 'cache miss'} -> "
                f"{total_time:.2f}s (encode: {encode_time:.2f}s, caption: {caption_time:.2f}s)")

    return caption, total_time, cache_hit


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
    image_id: Optional[str] = None  # optional: client-side image dedup key


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
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "stats": _cache_stats(),
    }


class DetectRequest(BaseModel):
    """Request for object detection. Accepts either base64 image or image_id (cached)."""
    image_b64: Optional[str] = None  # base64-encoded image (data:image/...;base64,...)
    image_id: Optional[str] = None   # client-side cache key (if image was previously sent)
    objects: list[str]               # object names to detect, e.g. ["button", "text field"]


class DetectResult(BaseModel):
    """Detection result for a single object."""
    label: str
    boxes: Optional[list] = None  # list of [x1, y1, x2, y2] in normalized coords (0-1)
    points: Optional[list] = None  # list of [x, y] center points


class DetectResponse(BaseModel):
    objects: list[DetectResult]
    cache_hit: bool
    elapsed_seconds: float


@app.post("/v1/detect")
async def detect(req: DetectRequest):
    """Detect objects in an image and return bounding boxes.

    Uses detect() for object detection and point() for center coordinates.
    Results use normalized coordinates (0-1) relative to image dimensions.
    """
    if not req.image_b64 and not req.image_id:
        raise HTTPException(status_code=400, detail="Either image_b64 or image_id required")

    if not req.objects:
        raise HTTPException(status_code=400, detail="objects list required")

    # Decode image
    try:
        if req.image_b64:
            img, raw_bytes = decode_image_from_base64_uri(req.image_b64)
            img_hash = _hash_image(raw_bytes)
        else:
            # Look up cached image by id
            img_hash = req.image_id
            with _cache_lock:
                if img_hash not in _embedding_cache:
                    raise HTTPException(status_code=404, detail=f"Image '{req.image_id}' not in cache")
            img = None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    try:
        start = time.time()
        results: list[DetectResult] = []
        cache_hit_count = 0

        enc, encode_time, was_cache_hit = _get_or_encode(img, img_hash)
        if was_cache_hit:
            cache_hit_count += 1

        for obj_name in req.objects:
            t1 = time.time()
            boxes = None
            points = None

            try:
                raw = model.detect(enc, obj_name)  # type: ignore[arg-type]
                if raw and "objects" in raw:
                    boxes = []
                    for obj in raw["objects"]:
                        boxes.append([obj.get("x_min", 0), obj.get("y_min", 0),
                                       obj.get("x_max", 0), obj.get("y_max", 0)])
            except Exception as e:
                logger.warning(f"detect '{obj_name}' failed: {e}")

            try:
                raw_pt = model.point(enc, obj_name)  # type: ignore[arg-type]
                if raw_pt and "points" in raw_pt:
                    points = []
                    for pt in raw_pt["points"]:
                        points.append([pt.get("x", 0), pt.get("y", 0)])
            except Exception as e:
                logger.debug(f"point '{obj_name}' failed: {e}")

            results.append(DetectResult(label=obj_name, boxes=boxes, points=points))
            global _total_inferences, _total_inference_time
            _total_inferences += 1
            _total_inference_time += time.time() - t1

        elapsed = time.time() - start
        logger.info(f"Detection: {len(req.objects)} objects in {elapsed:.2f}s (cache: {was_cache_hit})")

        _cleanup_memory()
        return DetectResponse(objects=results, cache_hit=was_cache_hit, elapsed_seconds=round(elapsed, 3))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/cache/clear")
async def clear_cache():
    """Clear the embedding cache (useful after memory pressure)."""
    global _cache_hits, _cache_misses
    with _cache_lock:
        size = len(_embedding_cache)
        _embedding_cache.clear()
    _cache_hits = 0
    _cache_misses = 0
    _cleanup_memory()
    logger.info(f"Cache cleared ({size} entries). Memory: {_get_memory_mb()} MB")
    return {"cleared": size, "memory_mb": _get_memory_mb()}


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
    images: list[Image.Image] = []
    image_hashes: list[str] = []

    content = last_msg.content
    if isinstance(content, str):
        text_parts.append(content)
    else:
        for part in content:
            if part.type == "text" and part.text:
                text_parts.append(part.text)
            elif part.type == "image_url" and part.image_url:
                try:
                    img, raw_bytes = decode_image_from_base64_uri(part.image_url["url"])
                    images.append(img)
                    image_hashes.append(_hash_image(raw_bytes))
                except Exception as e:
                    logger.error(f"Failed to decode image: {e}")

    if not images:
        raise HTTPException(status_code=400, detail="No image found in message")

    if len(images) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many images: {len(images)} (max {MAX_IMAGES_PER_REQUEST})"
        )

    # Build prompt
    raw_prompt = " ".join(text_parts).strip() if text_parts else "Describe this image in detail."

    # Check for structured mode: uses caption() for clean description + VQA for details
    use_structured = raw_prompt.startswith(STRUCTURED_PREFIX)
    user_prompt = raw_prompt[len(STRUCTURED_PREFIX):].strip() if use_structured else raw_prompt
    if not user_prompt:
        user_prompt = "Describe this image in detail."

    try:
        start = time.time()
        results: list[str] = []
        cache_hits = 0

        for i, img in enumerate(images):
            img_hash = image_hashes[i] if i < len(image_hashes) else None
            label = f"[Image {i + 1} of {len(images)}]" if len(images) > 1 else ""

            # Before processing, check if we're under memory pressure
            _check_memory_pressure()

            if use_structured:
                # Structured mode: caption (clean) + VQA (detailed) for richer output
                cap, _, hit1 = caption_image(img, "long", img_hash)
                if hit1:
                    cache_hits += 1

                vqa_prompt = f"{STRUCTURED_SYSTEM_PROMPT}\n\nUser request: {user_prompt}"
                vqa, _, hit2 = describe_image(img, vqa_prompt, img_hash)
                if hit2:
                    cache_hits += 1

                result = f"Caption: {cap}\n\nStructured details:\n{vqa}"
            else:
                # Simple mode: just answer the question
                result, _, hit = describe_image(img, user_prompt, img_hash)
                if hit:
                    cache_hits += 1

            results.append(f"{label} {result}".strip() if label else result)

            # CRITICAL: Clean up after each image to prevent MPS memory bloat.
            # Without this, 4 images × stale tensor allocations → 7+ GB.
            # Don't clean after the last image — the function is about to
            # return and final cleanup happens below.
            if i < len(images) - 1:
                _cleanup_memory()
                del img  # Release PIL Image reference

        elapsed = time.time() - start
        mem_mb = _get_memory_mb()
        logger.info(
            f"Request complete: {len(images)} image(s), "
            f"{cache_hits} cache hit(s), "
            f"{elapsed:.2f}s wall clock, "
            f"mem: {mem_mb}MB"
        )

        # Final cleanup to release any lingering tensors
        _cleanup_memory()

        content = "\n\n---\n\n".join(results) if len(results) > 1 else results[0]

        return ChatResponse(
            id=f"vision-{int(start)}",
            created=int(start),
            choices=[Choice(index=0, message={"role": "assistant", "content": content})],
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=len(content.split()),
                total_tokens=len(content.split()),
            ),
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8901, log_level="info")
