/**
 * DeepSeek Vision Bridge Extension
 *
 * Intercepts read tool calls that return images, sends them to a local
 * moondream2 VLM for description, and injects the text description so
 * DeepSeek (which doesn't support vision) can understand image content.
 *
 * Features:
 *   - Auto-manages the vision server (start on first use, kill on exit)
 *   - All images in a single read are sent together (multi-image support)
 *   - Structured mode: prompts for code/UI screenshots get JSON with
 *     bounding boxes, OCR text, and spatial layout
 *   - Embedding cache on server side: follow-up questions about same
 *     image skip re-encoding (~300ms vs ~2s)
 *
 * No manual setup needed beyond first install.
 */

import { spawn, type ChildProcess } from "node:child_process";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const __dirname = dirname(fileURLToPath(import.meta.url));

const VISION_SERVER_URL = "http://127.0.0.1:8901";
const SERVER_DIR = join(__dirname, "../deepseek-vision");
const START_SCRIPT = join(SERVER_DIR, "start.sh");

interface VisionResponse {
  choices: Array<{
    message: { role: string; content: string };
  }>;
}

let serverProcess: ChildProcess | null = null;
let serverAvailable = false;

async function checkServer(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000);
    const res = await fetch(`${VISION_SERVER_URL}/health`, { signal: controller.signal });
    clearTimeout(timeout);
    return res.ok;
  } catch {
    return false;
  }
}

async function ensureServerRunning(): Promise<boolean> {
  if (await checkServer()) {
    serverAvailable = true;
    return true;
  }

  // Server not running - start it
  console.log("[vision-bridge] Starting vision server...");

  return new Promise((resolve) => {
    serverProcess = spawn("bash", [START_SCRIPT], {
      cwd: SERVER_DIR,
      stdio: "pipe",
      detached: false,
    });

    let started = false;

    serverProcess.stdout?.on("data", (data: Buffer) => {
      const text = data.toString();
      // Model loaded signal
      if (text.includes("Application startup complete") || text.includes("Uvicorn running")) {
        started = true;
        serverAvailable = true;
        console.log("[vision-bridge] Vision server ready");
        resolve(true);
      }
    });

    serverProcess.stderr?.on("data", (data: Buffer) => {
      const text = data.toString();
      // Also check stderr for uvicorn startup messages
      if (text.includes("Application startup complete") || text.includes("Uvicorn running")) {
        started = true;
        serverAvailable = true;
        console.log("[vision-bridge] Vision server ready");
        resolve(true);
      }
    });

    serverProcess.on("error", (err) => {
      console.error("[vision-bridge] Failed to start server:", err.message);
      resolve(false);
    });

    serverProcess.on("exit", (code) => {
      if (!started) {
        console.error(`[vision-bridge] Server exited with code ${code}`);
        resolve(false);
      }
      serverProcess = null;
      serverAvailable = false;
    });

    // Timeout after 30 seconds
    setTimeout(() => {
      if (!started) {
        console.error("[vision-bridge] Server start timed out");
        resolve(false);
      }
    }, 30000);
  });
}

function stopServer() {
  if (serverProcess) {
    console.log("[vision-bridge] Stopping vision server...");
    serverProcess.kill("SIGTERM");
    // Force kill after 3 seconds
    setTimeout(() => {
      if (serverProcess) {
        serverProcess.kill("SIGKILL");
        serverProcess = null;
      }
    }, 3000);
  }
}

async function describeImages(
  images: Array<{ data: string; mimeType: string }>,
  contextPrompt?: string,
): Promise<string | null> {
  // Build content array with all images plus a description prompt
  const contentParts: Array<Record<string, unknown>> = [];

  // Use structured mode for better spatial/OCR output unless user had a
  // specific prompt in mind. Structured mode gives JSON with bounding
  // boxes, transcribed text, layout info, etc.
  const prompt = contextPrompt
    ? `[structured] ${contextPrompt}`
    : "[structured]";

  contentParts.push({ type: "text", text: prompt });

  for (const img of images) {
    contentParts.push({
      type: "image_url",
      image_url: { url: `data:${img.mimeType};base64,${img.data}` },
    });
  }

  try {
    const res = await fetch(`${VISION_SERVER_URL}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "moondream2",
        messages: [{ role: "user", content: contentParts }],
        max_tokens: 512,
      }),
    });

    if (!res.ok) return null;
    const data = (await res.json()) as VisionResponse;
    return data.choices?.[0]?.message?.content || null;
  } catch {
    return null;
  }
}

export default function (pi: ExtensionAPI) {
  // Clean up on shutdown
  pi.on("session_shutdown", () => {
    stopServer();
  });

  // Intercept image reads from the read tool
  pi.on("tool_result", async (event, _ctx) => {
    if (event.toolName !== "read") return;

    const content = event.content;
    if (!Array.isArray(content)) return;

    const imageBlocks = content.filter(
      (c): c is { type: "image"; data: string; mimeType: string } => c.type === "image",
    );

    if (imageBlocks.length === 0) return;

    // Ensure server is running
    const running = await ensureServerRunning();
    if (!running) {
      // Server failed to start - pass through with original content
      // (DeepSeek will reject the image_url, but at least the user sees the error)
      return {
        content: [
          ...content.filter((c: any) => c.type === "text"),
          {
            type: "text",
            text: "[Vision server failed to start. Run manually: cd ~/.pi/agent/deepseek-vision && ./start.sh]",
          },
        ],
      };
    }

    // Send all images in one request (server handles multi-image)
    const desc = await describeImages(
      imageBlocks.map((img) => ({ data: img.data, mimeType: img.mimeType })),
    );

    if (!desc) {
      // All descriptions failed — keep text content only
      const textOnly = content
        .filter((c: any) => c.type === "text")
        .map((c: any) => c as { type: "text"; text: string });

      textOnly.push({
        type: "text",
        text: `[Vision description failed for ${imageBlocks.length} image(s)]`,
      });

      return { content: textOnly };
    }

    // Replace image blocks with text descriptions
    const newContent = content
      .filter((c: any) => c.type === "text")
      .map((c: any) => c as { type: "text"; text: string });

    newContent.push({
      type: "text",
      text: `[Vision description of ${imageBlocks.length} image(s)]:\n${desc}`,
    });

    return { content: newContent };
  });

  pi.registerCommand("vision-status", {
    description: "Check vision server status",
    handler: async (_args, ctx) => {
      const ok = await checkServer();
      if (ok) {
        try {
          const res = await fetch(`${VISION_SERVER_URL}/health`);
          const data = (await res.json()) as Record<string, unknown>;
          ctx.ui.notify(
            `Vision server: ${data.status} | model: ${data.model} | device: ${data.device}`,
            "info",
          );
        } catch {
          ctx.ui.notify("Vision server running but health check failed", "warning");
        }
      } else if (serverProcess) {
        ctx.ui.notify("Vision server starting up...", "warning");
      } else {
        ctx.ui.notify("Vision server not running. Will auto-start on next image.", "info");
      }
      serverAvailable = ok;
    },
  });
}
