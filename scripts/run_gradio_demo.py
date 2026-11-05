import argparse
import atexit
import json
import os
import signal
import sys
import threading
import time

# Ensure repository root is in sys.path when running as a standalone script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cv2
import gradio as gr
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    from scripts.session_manager import SessionManager
    from scripts.ssl_utils import ensure_ssl_certificates, get_lan_ip
except ModuleNotFoundError:
    from session_manager import SessionManager
    from ssl_utils import ensure_ssl_certificates, get_lan_ip

from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle

default_prompt = "The person is wearing the clothing item from the reference image, perfect realistic fit, high quality photorealistic, preserve person's face, hair and background."

session_manager = SessionManager(timeout_seconds=60.0)

stream_processor = None
input_tensor = None
output_tensor = None
resolution = None
use_int8 = False

processor_lock = threading.Lock()
current_video_id = 0
current_video_id_lock = threading.Lock()

local_current_frame = None
local_processed_frame = None
local_frame_lock = threading.Lock()


def get_processor():
    global stream_processor, input_tensor, output_tensor, resolution

    if stream_processor is None:
        # stream_processor = StreamProcessor("configs/stream_processor_config.json") # uncomment if you dont need reference image
        stream_processor = StreamProcessor("configs/config_with_reference.json")
        if use_int8:
            stream_processor.enable_quantization()
        stream_processor.start()
        stream_processor.set_prompt(default_prompt)

        input_tensor = stream_processor.get_input_tensor()
        output_tensor = stream_processor.get_output_tensor()
        resolution = stream_processor.get_resolution()

    return stream_processor, input_tensor, output_tensor, resolution


def to_bgr(frame):
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def to_rgb(frame):
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def process_frame(frame):
    sp, input_tensor, output_tensor, resolution = get_processor()
    frame = crop_maximal_rectangle(frame, resolution["height"], resolution["width"])

    with processor_lock:
        input_tensor.copy_from(frame)
        if sp.is_ready():
            processed = output_tensor.to_numpy()
        else:
            processed = frame.copy()
            cv2.putText(
                processed,
                "GPU Warming Up... Please wait",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return frame, processed


webcam_frame_count = 0


def set_prompt(prompt: str):
    sp, _, _, _ = get_processor()
    print(f"[FluxRT] Prompt updated to: '{prompt[:60]}...'")
    sp.set_prompt(prompt)


def set_reference_image_ui(image):
    if image is not None:
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        print(f"[FluxRT] Reference garment updated: shape {image.shape}", flush=True)
    sp, _, _, _ = get_processor()
    sp.set_reference_image(image)


def _video_loop(video_path: str, video_id: int):
    """Background thread: process video frames and store in shared state."""
    global local_current_frame, local_processed_frame
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_time = 1.0 / fps
    try:
        while True:
            with current_video_id_lock:
                if current_video_id != video_id:
                    break
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            start = time.time()
            input_frame, processed = process_frame(frame)
            with local_frame_lock:
                local_current_frame = to_rgb(input_frame)
                local_processed_frame = to_rgb(processed)
            time.sleep(max(0, frame_time - (time.time() - start)))
    finally:
        cap.release()


def start_local_video(video_path: str | None):
    """Start video processing in a background thread, cancelling any previous one."""
    global current_video_id, local_current_frame, local_processed_frame
    with current_video_id_lock:
        current_video_id += 1
        my_id = current_video_id
    with local_frame_lock:
        local_current_frame = None
        local_processed_frame = None
    if not video_path:
        return
    t = threading.Thread(target=_video_loop, args=(video_path, my_id), daemon=True)
    t.start()


def poll_local_video():
    """Return the latest processed frames for gr.Timer display."""
    with local_frame_lock:
        return local_current_frame, local_processed_frame


def switch_mode(mode: str, request: gr.Request | None):
    global current_video_id, local_current_frame, local_processed_frame
    if mode == "webcam":
        with current_video_id_lock:
            current_video_id += 1
        with local_frame_lock:
            local_current_frame = None
            local_processed_frame = None

    webcam_visible = mode == "webcam"
    local_visible = mode == "local"
    return (
        gr.update(visible=webcam_visible),  # webcam_output_col
        gr.update(visible=local_visible),  # local_output_col
        gr.update(visible=webcam_visible),  # webcam_input_col
        gr.update(visible=local_visible),  # local_input_col
        gr.update(active=local_visible),  # local_timer
    )


def process_webcam(frame):
    global webcam_frame_count
    if frame is None:
        return None

    webcam_frame_count += 1
    if webcam_frame_count == 1:
        print("[FluxRT] First webcam frame received from browser! Streaming is active.")
    elif webcam_frame_count % 50 == 0:
        print(f"[FluxRT] Streaming active: processed {webcam_frame_count} frames from browser.")

    _, processed = process_frame(to_bgr(frame))
    return to_rgb(processed)


def main():
    global use_int8
    parser = argparse.ArgumentParser(description="Run FluxRT Gradio demo with LAN and multi-device support.")
    parser.add_argument("--int8", action="store_true", help="Enable int8 quantization")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument(
        "--ssl",
        action="store_true",
        default=True,
        help="Enable HTTPS with SSL certificates for camera access (default: True)",
    )
    parser.add_argument(
        "--no-ssl",
        action="store_false",
        dest="ssl",
        help="Disable SSL and run plain HTTP",
    )
    parser.add_argument(
        "--cert",
        type=str,
        default="certs/cert.pem",
        help="Path to SSL certificate (default: certs/cert.pem)",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="certs/key.pem",
        help="Path to SSL private key (default: certs/key.pem)",
    )
    parser.add_argument(
        "--session-timeout",
        type=float,
        default=60.0,
        help="Seconds of webcam inactivity before auto-releasing the GPU session (default: 60)",
    )
    args = parser.parse_args()
    use_int8 = args.int8
    session_manager.timeout_seconds = args.session_timeout

    get_processor()
    use_reference_image = stream_processor.config.get("use_reference_image", False)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    garments_dir = os.path.join(repo_root, "garments")
    catalog_path = os.path.join(garments_dir, "catalog.json")

    garments_catalog = []
    if os.path.exists(catalog_path):
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                garments_catalog = json.load(f)
        except Exception as e:
            print(f"[FluxRT] Error loading garments catalog: {e}", flush=True)

    default_garment_id = "mens_denim_jacket"
    default_garment = next((g for g in garments_catalog if g.get("id") == default_garment_id), None)
    if default_garment is None and garments_catalog:
        default_garment = garments_catalog[0]
        default_garment_id = default_garment["id"]

    # Preload default reference garment into the GPU inference subprocess
    if default_garment and use_reference_image:
        init_file = os.path.join(repo_root, default_garment["file"])
        if os.path.exists(init_file):
            init_img = cv2.imread(init_file)
            if init_img is not None:
                init_img_rgb = cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB)
                set_reference_image_ui(init_img_rgb)
                print(f"[FluxRT] Preloaded default reference garment: {default_garment['name']}", flush=True)

    ws_client_js = """
    <script>
    let tryonWs = null;
    let tryonStream = null;
    let tryonStreaming = false;
    let tryonSending = false;
    let tryonFrames = 0;
    let tryonLastFpsTime = performance.now();
    let tryonAnimId = null;
    let tryonLastSendTime = 0;

    const tryonCapCanvas = document.createElement('canvas');
    tryonCapCanvas.width = 576;
    tryonCapCanvas.height = 320;
    const tryonCapCtx = tryonCapCanvas.getContext('2d');

    function tryonCaptureLoop() {
        if (!tryonStreaming) return;

        // Watchdog timeout: if a frame takes > 1000ms, unlock sending
        if (tryonSending && (performance.now() - tryonLastSendTime > 1000)) {
            tryonSending = false;
        }

        if (tryonWs && tryonWs.readyState === WebSocket.OPEN && !tryonSending) {
            const video = document.getElementById('tryon-webcam');
            if (video && video.readyState >= 2) {
                tryonSending = true;
                tryonLastSendTime = performance.now();
                tryonCapCtx.drawImage(video, 0, 0, tryonCapCanvas.width, tryonCapCanvas.height);
                tryonCapCanvas.toBlob((blob) => {
                    if (blob && tryonWs && tryonWs.readyState === WebSocket.OPEN) {
                        blob.arrayBuffer().then((buf) => {
                            if (tryonWs && tryonWs.readyState === WebSocket.OPEN) {
                                tryonWs.send(buf);
                            } else {
                                tryonSending = false;
                            }
                        }).catch(() => { tryonSending = false; });
                    } else {
                        tryonSending = false;
                    }
                }, 'image/jpeg', 0.80);
            }
        }

        tryonAnimId = requestAnimationFrame(tryonCaptureLoop);
    }

    async function acquireCameraStream() {
        if (!window.isSecureContext && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
            throw new Error("INSECURE_CONTEXT");
        }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("NO_MEDIA_DEVICES");
        }

        // Try standard ideal 640x480 first (desktop safe, NO facingMode)
        try {
            return await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 } },
                audio: false
            });
        } catch (e1) {
            console.warn("Ideal resolution request failed, trying basic video: true", e1);
            return await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        }
    }

    function handleCameraError(err) {
        const status = document.getElementById('tryon-status');
        console.error("Camera error details:", err);
        
        let msg = "";
        if (err.message === "INSECURE_CONTEXT") {
            msg = `❌ <b>Insecure HTTP:</b> You accessed via <code>http://${window.location.host}</code>.<br>` +
                  `Webcams on LAN devices require HTTPS for browser security.<br>` +
                  `👉 <b>Click here to open HTTPS:</b> <a href="https://${window.location.host}" style="color:#38bdf8; font-weight:bold; text-decoration:underline;">https://${window.location.host}</a><br>` +
                  `<span style="font-size:11.5px; color:#9ca3af;">(Click <i>'Advanced' → 'Proceed to ${window.location.hostname}'</i> when prompted)</span>`;
        } else if (err.message === "CAMERA_TIMEOUT") {
            msg = `⚠️ <b>Camera Request Timed Out (15s):</b><br>` +
                  `Your device did not respond to the camera request.<br>` +
                  `1. Check if browser has camera access in System Settings.<br>` +
                  `2. Look for the camera icon 📷 in your address bar and click <b>Always allow</b>.<br>` +
                  `3. Make sure FaceTime, Zoom, or other apps aren't locking the camera.`;
        } else if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
            msg = `❌ <b>Camera Permission Denied:</b><br>` +
                  `1. Click the camera/tune icon 📷 in your browser's address bar and set Camera to <b>Allow</b>.<br>` +
                  `2. Check OS Privacy settings for camera access.<br>` +
                  `3. Refresh the page and try again.`;
        } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
            msg = `❌ <b>Camera In Use / Hardware Locked:</b><br>` +
                  `Another app is currently using your webcam.<br>` +
                  `Please close other camera apps and click Start again.`;
        } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
            msg = `❌ <b>No Camera Found:</b> No webcam was detected on this device.`;
        } else {
            msg = `❌ <b>Camera Error:</b> ${err.name || "Error"}: ${err.message || "Failed to start camera."}`;
        }
        
        status.innerHTML = `<span style="color:#f87171; line-height:1.5;">${msg}</span>`;
        stopLiveTryon();
    }

    async function testCameraOnly() {
        const status = document.getElementById('tryon-status');
        try {
            status.innerHTML = "Testing camera access... <span style='color:#9ca3af;'>(Look for prompt)</span>";
            status.style.color = "#fbbf24";

            const stream = await acquireCameraStream();
            const video = document.getElementById('tryon-webcam');
            video.srcObject = stream;
            await video.play();
            status.innerHTML = "🟢 <b>Camera works!</b> Now click <b>▶️ Start Live Try-On</b>.";
            status.style.color = "#34d399";
        } catch (err) {
            handleCameraError(err);
        }
    }

    async function startLiveTryon() {
        const status = document.getElementById('tryon-status');
        const startBtn = document.getElementById('btn-tryon-start');
        const stopBtn = document.getElementById('btn-tryon-stop');
        
        try {
            // Check if GPU is already occupied by another LAN device before opening camera
            try {
                const preCheck = await fetch('/api/session_status');
                if (preCheck.ok) {
                    const stat = await preCheck.json();
                    if (stat.in_use) {
                        status.innerHTML = `<span style="color:#f59e0b; font-weight:600;">⏳ GPU Busy:</span> <span style="font-size:12px; color:#d1d5db;">In use by ${stat.active_ip} (${stat.elapsed_seconds}s).</span>`;
                        showToast(`⚠️ GPU is currently in use by ${stat.active_ip}. Please wait for their session to finish.`);
                        return;
                    }
                }
            } catch (e) {}

            status.innerHTML = "Requesting webcam access... <span style='color:#9ca3af;'>(Allow camera in browser if prompted)</span>";
            status.style.color = "#fbbf24";

            const tipTimer = setTimeout(() => {
                if (!tryonStream) {
                    status.innerHTML = "<span style='color:#f59e0b; font-weight:bold;'>⚠️ Waiting for camera approval!</span><br>" +
                        "<span style='font-size:12px; color:#d1d5db;'>Look at the camera icon 📷 in your address bar or permission popup.</span>";
                }
            }, 3000);

            const camPromise = acquireCameraStream();
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error("CAMERA_TIMEOUT")), 15000);
            });

            tryonStream = await Promise.race([camPromise, timeoutPromise]);
            clearTimeout(tipTimer);

            const video = document.getElementById('tryon-webcam');
            video.srcObject = tryonStream;
            await new Promise((resolve) => {
                video.onloadedmetadata = () => video.play().then(resolve).catch(resolve);
                if (video.readyState >= 1) video.play().then(resolve).catch(resolve);
                setTimeout(resolve, 800);
            });
            
            status.innerHTML = '<span style="color:#38bdf8;">● Connecting to GPU...</span>';

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            tryonWs = new WebSocket(`${protocol}//${window.location.host}/ws_tryon`);
            tryonWs.binaryType = 'arraybuffer';
            
            const canvas = document.getElementById('tryon-canvas');
            const ctx = canvas.getContext('2d');
            
            tryonWs.onopen = () => {
                tryonStreaming = true;
                tryonSending = false;
                status.innerHTML = '<span style="color:#10b981; font-weight:600;">● Live Try-On Active</span>';
                startBtn.disabled = true;
                startBtn.style.opacity = "0.5";
                stopBtn.disabled = false;
                stopBtn.style.opacity = "1.0";
                
                if (tryonAnimId) cancelAnimationFrame(tryonAnimId);
                tryonAnimId = requestAnimationFrame(tryonCaptureLoop);
            };
            
            tryonWs.onmessage = async (event) => {
                try {
                    // Check if server sent a text control/error message (e.g. GPU_BUSY)
                    if (typeof event.data === 'string') {
                        try {
                            const info = JSON.parse(event.data);
                            if (info.type === 'error' && info.code === 'GPU_BUSY') {
                                status.innerHTML = `<span style="color:#f59e0b; font-weight:600;">⏳ GPU Busy:</span> <span style="font-size:12px; color:#d1d5db;">In use by ${info.active_ip} (${info.elapsed_seconds}s).</span>`;
                                showToast(`⚠️ GPU Busy: Device ${info.active_ip} is currently streaming.`);
                                stopLiveTryon();
                                return;
                            }
                        } catch (pe) {
                            console.warn("WS text message parse error:", pe);
                        }
                        return;
                    }

                    const blob = new Blob([event.data], { type: 'image/jpeg' });
                    const bitmap = await createImageBitmap(blob);
                    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
                    bitmap.close();
                    
                    tryonFrames++;
                    const now = performance.now();
                    if (now - tryonLastFpsTime >= 1000) {
                        const fps = Math.round((tryonFrames * 1000) / (now - tryonLastFpsTime));
                        document.getElementById('tryon-fps').innerText = `${fps} FPS`;
                        tryonFrames = 0;
                        tryonLastFpsTime = now;
                    }
                } catch (e) {
                    console.error("Frame draw error:", e);
                } finally {
                    tryonSending = false;
                }
            };
            
            tryonWs.onerror = (e) => {
                console.error("WebSocket error:", e);
                status.innerHTML = '<span style="color:#f87171;">● Connection Error</span>';
                stopLiveTryon();
            };
            
            tryonWs.onclose = (event) => {
                if (event.code === 1008) {
                    console.log("WebSocket closed by server: GPU busy with another device.");
                }
                if (tryonStreaming) {
                    stopLiveTryon();
                }
            };
            
        } catch (err) {
            handleCameraError(err);
        }
    }

    function stopLiveTryon() {
        tryonStreaming = false;
        tryonSending = false;
        if (tryonAnimId) {
            cancelAnimationFrame(tryonAnimId);
            tryonAnimId = null;
        }
        if (tryonWs) {
            try { tryonWs.close(); } catch(e) {}
            tryonWs = null;
        }
        if (tryonStream) {
            tryonStream.getTracks().forEach(t => t.stop());
            tryonStream = null;
        }
        const video = document.getElementById('tryon-webcam');
        if (video) video.srcObject = null;
        
        const startBtn = document.getElementById('btn-tryon-start');
        const stopBtn = document.getElementById('btn-tryon-stop');
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.style.opacity = "1.0";
        }
        if (stopBtn) {
            stopBtn.disabled = true;
            stopBtn.style.opacity = "0.5";
        }
        const status = document.getElementById('tryon-status');
        if (status) {
            status.innerHTML = '<span style="color:#94a3b8;">● Ready</span>';
        }
        const fps = document.getElementById('tryon-fps');
        if (fps) fps.innerText = "0 FPS";
    }

    function toggleFullscreen() {
        const target = document.getElementById('tryon-canvas-container') || document.getElementById('tryon-canvas');
        if (!target) return;

        const isFullscreen = !!(document.fullscreenElement || document.webkitFullscreenElement || document.mozFullScreenElement || document.msFullscreenElement);

        if (!isFullscreen) {
            if (target.requestFullscreen) {
                target.requestFullscreen().catch(e => console.warn(e));
            } else if (target.webkitRequestFullscreen) {
                target.webkitRequestFullscreen();
            } else if (target.mozRequestFullScreen) {
                target.mozRequestFullScreen();
            } else if (target.msRequestFullscreen) {
                target.msRequestFullscreen();
            }
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen().catch(e => console.warn(e));
            } else if (document.webkitExitFullscreen) {
                document.webkitExitFullscreen();
            } else if (document.mozCancelFullScreen) {
                document.mozCancelFullScreen();
            } else if (document.msExitFullscreen) {
                document.msExitFullscreen();
            }
        }
    }

    function updateFullscreenUI() {
        const isFullscreen = !!(document.fullscreenElement || document.webkitFullscreenElement || document.mozFullScreenElement || document.msFullscreenElement);
        const btnMain = document.getElementById('btn-tryon-fullscreen');
        const btnHeader = document.getElementById('btn-header-fs');
        const btnOverlay = document.getElementById('btn-overlay-fullscreen');
        const label = isFullscreen ? '⤓ Exit Fullscreen' : '⛶ Fullscreen';
        if (btnMain) btnMain.innerHTML = label;
        if (btnHeader) btnHeader.innerHTML = label;
        if (btnOverlay) btnOverlay.innerHTML = label;
    }

    ['fullscreenchange', 'webkitfullscreenchange', 'mozfullscreenchange', 'MSFullscreenChange'].forEach(evt => {
        document.addEventListener(evt, updateFullscreenUI);
    });

    window.acquireCameraStream = acquireCameraStream;
    window.testCameraOnly = testCameraOnly;
    window.startLiveTryon = startLiveTryon;
    window.stopLiveTryon = stopLiveTryon;
    window.toggleFullscreen = toggleFullscreen;

    // --- Preset Wardrobe & Interactive Garment Selector ---
    const GARMENTS = __CATALOG_JSON__;
    let activeGarmentId = __DEFAULT_GARMENT_ID__;
    let currentGarmentFilter = 'all';
    let garmentSearchQuery = '';

    function showToast(message) {
        let toast = document.getElementById('flux-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'flux-toast';
            toast.className = 'flux-toast';
            document.body.appendChild(toast);
        }
        toast.innerHTML = message;
        toast.classList.add('show');
        clearTimeout(window._toastTimeout);
        window._toastTimeout = setTimeout(() => {
            toast.classList.remove('show');
        }, 2500);
    }

    function updateEquippedBanner(item) {
        const banner = document.getElementById('flux-equipped-banner');
        if (!banner) return;
        if (!item) {
            banner.innerHTML = '<span style="color:#64748b; font-size:12px;">No garment equipped</span>';
            return;
        }
        banner.innerHTML = `
            <div class="flux-equipped-item">
                <img class="equipped-thumb" src="/${item.file}" alt="${item.name}" />
                <div class="equipped-details">
                    <span class="equipped-label">Equipped Garment</span>
                    <span class="equipped-name" title="${item.name}">${item.name}</span>
                </div>
                <span class="equipped-badge">✓ Active</span>
            </div>
        `;
    }

    async function selectPresetGarmentById(id) {
        const item = GARMENTS.find(g => g.id === id);
        if (!item) return;
        activeGarmentId = id;
        updateEquippedBanner(item);
        renderGarmentGallery();

        try {
            const resp = await fetch('/api/select_garment', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: item.file })
            });
            const data = await resp.json();
            if (data.success) {
                showToast(`✨ Equipped: <b>${item.name}</b>`);
            } else {
                console.error("Garment select error:", data.error);
            }
        } catch (e) {
            console.error("Garment select request failed:", e);
        }
    }

    function filterGarments(filter) {
        currentGarmentFilter = filter;
        document.querySelectorAll('.flux-filter-pill').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.filter === filter);
        });
        renderGarmentGallery();
    }

    function searchGarments(query) {
        garmentSearchQuery = (query || '').toLowerCase().trim();
        renderGarmentGallery();
    }

    function garmentMatchesFilter(item, filter) {
        if (filter === 'all') return true;
        if (filter === 'men') return item.gender === 'Men';
        if (filter === 'women') return item.gender === 'Women';
        if (filter === 'dresses') return item.category === 'Dresses' || item.category === 'Gowns';
        if (filter === 'lehengas') return item.category === 'Lehengas';
        if (filter === 'suits') return item.category === 'Suits' || item.category === 'Full Sets';
        if (filter === 'tops') return ['Shirts', 'T-Shirts', 'Jackets'].includes(item.category);
        if (filter === 'bottoms') return ['Pants', 'Half-Pants', 'Jeans'].includes(item.category);
        return true;
    }

    function renderGarmentGallery() {
        const grid = document.getElementById('garment-gallery-grid');
        const countEl = document.getElementById('garment-visible-count');
        if (!grid) return;

        const filtered = GARMENTS.filter(item => {
            if (!garmentMatchesFilter(item, currentGarmentFilter)) return false;
            if (garmentSearchQuery) {
                const text = `${item.name} ${item.category} ${item.gender} ${item.type}`.toLowerCase();
                return text.includes(garmentSearchQuery);
            }
            return true;
        });

        if (countEl) {
            countEl.innerText = `${filtered.length} styles`;
        }

        if (filtered.length === 0) {
            grid.innerHTML = '<div class="flux-garment-empty">No garments match your search/filter.</div>';
            return;
        }

        grid.innerHTML = filtered.map(item => {
            const isActive = item.id === activeGarmentId;
            const typeLabel = (item.type || '').replace('_', ' ');
            return `
                <div class="flux-garment-card ${isActive ? 'active-card' : ''}" onclick="window.selectPresetGarmentById('${item.id}')" title="Click to try on ${item.name}">
                    <div class="flux-garment-img-wrapper">
                        <img src="/${item.file}" alt="${item.name}" loading="lazy" />
                        ${isActive ? '<span class="flux-active-pill">✓ In Use</span>' : ''}
                        <span class="flux-gender-badge ${item.gender.toLowerCase()}">${item.gender}</span>
                    </div>
                    <div class="flux-garment-meta">
                        <div class="flux-garment-title" title="${item.name}">${item.name}</div>
                        <div class="flux-garment-tags">
                            <span class="flux-tag">${item.category}</span>
                            <span class="flux-tag type">${typeLabel}</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    function initWardrobe() {
        const activeItem = GARMENTS.find(g => g.id === activeGarmentId) || GARMENTS[0];
        if (activeItem) {
            activeGarmentId = activeItem.id;
            updateEquippedBanner(activeItem);
        }
        renderGarmentGallery();
    }

    function initWhenReady() {
        if (document.getElementById('garment-gallery-grid')) {
            initWardrobe();
        } else {
            setTimeout(initWhenReady, 80);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initWhenReady);
    } else {
        initWhenReady();
    }

    window.selectPresetGarmentById = selectPresetGarmentById;
    window.filterGarments = filterGarments;
    window.searchGarments = searchGarments;

    // --- LAN GPU Session Coordinator Status Poller ---
    async function updateLanGpuStatus() {
        if (tryonStreaming) return;
        try {
            const resp = await fetch('/api/session_status');
            if (!resp.ok) return;
            const data = await resp.json();
            const pill = document.getElementById('lan-session-pill');
            if (!pill) return;
            if (data.in_use) {
                pill.className = 'flux-badge-lan busy';
                pill.innerHTML = `<span class="status-dot amber"></span> GPU Busy (${data.active_ip || 'LAN'})`;
                pill.title = `Device ${data.active_ip} is currently streaming (${data.elapsed_seconds}s)`;
            } else {
                pill.className = 'flux-badge-lan ready';
                pill.innerHTML = `<span class="status-dot green"></span> GPU Ready`;
                pill.title = 'GPU pipeline is free for any LAN device';
            }
        } catch (e) {}
    }

    setInterval(updateLanGpuStatus, 3000);
    setTimeout(updateLanGpuStatus, 300);
    </script>
    """

    ws_client_js = ws_client_js.replace(
        "__CATALOG_JSON__", json.dumps(garments_catalog)
    ).replace(
        "__DEFAULT_GARMENT_ID__", json.dumps(default_garment_id)
    )

    ui_html = """
    <style>
    .flux-container {
        background: #0f172a;
        border-radius: 14px;
        padding: 16px;
        border: 1px solid #1e293b;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        margin-bottom: 16px;
    }
    .flux-toolbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        padding-bottom: 14px;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 14px;
    }
    .flux-btn-group {
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
    }
    .flux-btn {
        border: none;
        border-radius: 8px;
        font-size: 13.5px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.15s ease;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 9px 18px;
    }
    .flux-btn:active {
        transform: scale(0.97);
    }
    .flux-btn-primary {
        background: #10b981;
        color: #ffffff;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    .flux-btn-primary:hover:not(:disabled) {
        background: #059669;
    }
    .flux-btn-danger {
        background: #ef4444;
        color: #ffffff;
    }
    .flux-btn-danger:hover:not(:disabled) {
        background: #dc2626;
    }
    .flux-btn-secondary {
        background: #1e293b;
        color: #94a3b8;
        border: 1px solid #334155;
    }
    .flux-btn-secondary:hover:not(:disabled) {
        background: #334155;
        color: #f8fafc;
    }
    .flux-btn-indigo {
        background: #4f46e5;
        color: #ffffff;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.25);
    }
    .flux-btn-indigo:hover:not(:disabled) {
        background: #4338ca;
    }
    .flux-info-group {
        display: flex;
        gap: 12px;
        align-items: center;
        font-size: 13px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .flux-badge {
        background: #1e293b;
        border: 1px solid #334155;
        padding: 5px 12px;
        border-radius: 6px;
        color: #38bdf8;
        font-weight: 700;
        font-size: 13px;
    }
    .flux-badge-lan {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
        border: 1px solid #334155;
        background: #1e293b;
        color: #94a3b8;
        transition: all 0.2s ease;
    }
    .flux-badge-lan.ready {
        border-color: rgba(16, 185, 129, 0.4);
        color: #34d399;
    }
    .flux-badge-lan.busy {
        border-color: rgba(245, 158, 11, 0.4);
        color: #fbbf24;
        background: rgba(245, 158, 11, 0.12);
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .status-dot.green {
        background: #10b981;
        box-shadow: 0 0 6px #10b981;
    }
    .status-dot.amber {
        background: #f59e0b;
        box-shadow: 0 0 6px #f59e0b;
    }
    .flux-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
    }
    @media (max-width: 800px) {
        .flux-grid {
            grid-template-columns: 1fr;
        }
    }
    .flux-feed-card {
        background: #020617;
        border: 1px solid #1e293b;
        border-radius: 10px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
    }
    .flux-feed-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 14px;
        background: #0f172a;
        border-bottom: 1px solid #1e293b;
        font-size: 12.5px;
        font-weight: 600;
        color: #94a3b8;
        letter-spacing: 0.03em;
    }
    .flux-video-wrapper {
        position: relative;
        width: 100%;
        height: 360px;
        background: #000000;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }
    .flux-video-wrapper video,
    .flux-video-wrapper canvas {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    #tryon-canvas-container:fullscreen,
    #tryon-canvas-container:-webkit-full-screen {
        width: 100vw !important;
        height: 100vh !important;
        background: #000000 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        border-radius: 0 !important;
    }
    #tryon-canvas-container:fullscreen #tryon-canvas,
    #tryon-canvas-container:-webkit-full-screen #tryon-canvas {
        width: 100vw !important;
        height: 100vh !important;
        max-width: 100vw !important;
        max-height: 100vh !important;
        object-fit: contain !important;
    }
    .flux-fs-overlay-btn {
        position: absolute;
        top: 10px;
        right: 10px;
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #f8fafc;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
        z-index: 50;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .flux-fs-overlay-btn:hover {
        background: rgba(30, 41, 59, 0.95);
        border-color: rgba(255, 255, 255, 0.4);
        transform: scale(1.02);
    }
    .flux-wardrobe-section {
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid #1e293b;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .flux-wardrobe-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .flux-wardrobe-heading {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .flux-wardrobe-icon {
        font-size: 18px;
    }
    .flux-wardrobe-title {
        font-size: 15px;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.01em;
    }
    .flux-wardrobe-count {
        font-size: 11.5px;
        color: #64748b;
        background: #1e293b;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
    }
    .flux-equipped-banner {
        display: flex;
        align-items: center;
    }
    .flux-equipped-item {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #1e293b;
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 4px 10px 4px 5px;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
    }
    .equipped-thumb {
        width: 34px;
        height: 34px;
        border-radius: 6px;
        object-fit: cover;
        background: #0b1120;
        border: 1px solid #334155;
    }
    .equipped-details {
        display: flex;
        flex-direction: column;
        line-height: 1.2;
    }
    .equipped-label {
        font-size: 9.5px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
    }
    .equipped-name {
        font-size: 12px;
        color: #f8fafc;
        font-weight: 600;
        max-width: 220px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .equipped-badge {
        background: #10b981;
        color: #ffffff;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 10px;
    }
    .flux-wardrobe-controls {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }
    .flux-filter-pills {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
    }
    .flux-filter-pill {
        background: #1e293b;
        color: #94a3b8;
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    .flux-filter-pill:hover {
        background: #334155;
        color: #f8fafc;
    }
    .flux-filter-pill.active {
        background: #3b82f6;
        color: #ffffff;
        border-color: #60a5fa;
        box-shadow: 0 2px 6px rgba(59, 130, 246, 0.35);
    }
    .flux-search-input {
        background: #0f172a;
        border: 1px solid #334155;
        color: #f8fafc;
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 12px;
        outline: none;
        min-width: 220px;
        transition: border-color 0.2s ease;
    }
    .flux-search-input:focus {
        border-color: #38bdf8;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
    }
    .flux-garment-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
        gap: 12px;
        max-height: 400px;
        overflow-y: auto;
        padding: 4px 2px;
    }
    .flux-garment-scroll::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    .flux-garment-scroll::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 3px;
    }
    .flux-garment-scroll::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
    .flux-garment-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        overflow: hidden;
        cursor: pointer;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        flex-direction: column;
        position: relative;
        user-select: none;
    }
    .flux-garment-card:hover {
        transform: translateY(-3px);
        border-color: #60a5fa;
        box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.5);
    }
    .flux-garment-card.active-card {
        border: 2px solid #10b981 !important;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.45) !important;
        transform: translateY(-2px);
        background: #132338;
    }
    .flux-garment-img-wrapper {
        position: relative;
        width: 100%;
        aspect-ratio: 1 / 1;
        background: #0b1120;
        overflow: hidden;
    }
    .flux-garment-img-wrapper img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.3s ease;
    }
    .flux-garment-card:hover .flux-garment-img-wrapper img {
        transform: scale(1.05);
    }
    .flux-active-pill {
        position: absolute;
        top: 6px;
        right: 6px;
        background: #10b981;
        color: #ffffff;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
        letter-spacing: 0.02em;
    }
    .flux-gender-badge {
        position: absolute;
        bottom: 6px;
        left: 6px;
        font-size: 9.5px;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
        backdrop-filter: blur(4px);
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .flux-gender-badge.men {
        background: rgba(30, 58, 138, 0.85);
        color: #93c5fd;
        border: 1px solid rgba(147, 197, 253, 0.3);
    }
    .flux-gender-badge.women {
        background: rgba(131, 24, 67, 0.85);
        color: #fbcfe8;
        border: 1px solid rgba(251, 207, 232, 0.3);
    }
    .flux-garment-meta {
        padding: 8px 10px;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .flux-garment-title {
        font-size: 11.5px;
        font-weight: 600;
        color: #f1f5f9;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        line-height: 1.3;
    }
    .flux-garment-tags {
        display: flex;
        gap: 4px;
        flex-wrap: wrap;
    }
    .flux-tag {
        font-size: 9.5px;
        padding: 1px 5px;
        border-radius: 4px;
        background: #0f172a;
        color: #94a3b8;
        border: 1px solid #334155;
    }
    .flux-tag.type {
        color: #38bdf8;
        border-color: rgba(56, 189, 248, 0.25);
    }
    .flux-garment-empty {
        grid-column: 1 / -1;
        text-align: center;
        padding: 30px;
        color: #64748b;
        font-size: 13px;
    }
    .flux-toast {
        position: fixed;
        bottom: 24px;
        right: 24px;
        background: #0f172a;
        border: 1px solid #10b981;
        color: #f8fafc;
        padding: 10px 18px;
        border-radius: 8px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        font-size: 13px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
        z-index: 99999;
        opacity: 0;
        transform: translateY(10px);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        pointer-events: none;
    }
    .flux-toast.show {
        opacity: 1;
        transform: translateY(0);
    }
    </style>

    <div class="flux-container">
        <div class="flux-toolbar">
            <div class="flux-btn-group">
                <button id="btn-tryon-start" class="flux-btn flux-btn-primary" onclick="window.startLiveTryon()">
                    ▶ Start Live Try-On
                </button>
                <button id="btn-tryon-stop" class="flux-btn flux-btn-danger" onclick="window.stopLiveTryon()" style="opacity: 0.5;" disabled>
                    ⏹ Stop
                </button>
                <button id="btn-tryon-test" class="flux-btn flux-btn-secondary" onclick="window.testCameraOnly()">
                    📷 Test Camera
                </button>
                <button id="btn-tryon-fullscreen" class="flux-btn flux-btn-indigo" onclick="window.toggleFullscreen()" title="Toggle Fullscreen Output">
                    ⛶ Fullscreen
                </button>
            </div>
            <div class="flux-info-group">
                <div id="lan-session-pill" class="flux-badge-lan ready" title="GPU pipeline status across company LAN">
                    <span class="status-dot green"></span> GPU Ready
                </div>
                <div id="tryon-status"><span style="color:#94a3b8;">● Ready</span></div>
                <span id="tryon-fps" class="flux-badge">0 FPS</span>
            </div>
        </div>
        
        <div class="flux-grid">
            <div class="flux-feed-card">
                <div class="flux-feed-header">
                    <span>📹 Webcam Input</span>
                </div>
                <div class="flux-video-wrapper">
                    <video id="tryon-webcam" autoplay muted playsinline></video>
                </div>
            </div>
            <div class="flux-feed-card">
                <div class="flux-feed-header">
                    <span>✨ Virtual Try-On Output</span>
                    <button id="btn-header-fs" onclick="window.toggleFullscreen()" style="background:none; border:none; color:#94a3b8; font-size:12px; cursor:pointer; display:flex; align-items:center; gap:4px;" title="View Fullscreen">
                        ⛶ Fullscreen
                    </button>
                </div>
                <div id="tryon-canvas-container" class="flux-video-wrapper">
                    <canvas id="tryon-canvas" width="576" height="320" ondblclick="window.toggleFullscreen()" title="Double-click to toggle fullscreen"></canvas>
                    <button id="btn-overlay-fullscreen" class="flux-fs-overlay-btn" onclick="window.toggleFullscreen()" title="Toggle full screen">
                        ⛶ Fullscreen
                    </button>
                </div>
            </div>
        </div>

        <div class="flux-wardrobe-section">
            <div class="flux-wardrobe-header">
                <div class="flux-wardrobe-title-row">
                    <div class="flux-wardrobe-heading">
                        <span class="flux-wardrobe-icon">👗</span>
                        <span class="flux-wardrobe-title">Preset Garments Wardrobe</span>
                        <span class="flux-wardrobe-count" id="garment-visible-count">33 styles</span>
                    </div>
                    <div id="flux-equipped-banner" class="flux-equipped-banner">
                        <!-- Populated by JS -->
                    </div>
                </div>
                <div class="flux-wardrobe-controls">
                    <div class="flux-filter-pills" id="flux-filter-pills">
                        <button class="flux-filter-pill active" data-filter="all" onclick="window.filterGarments('all')">All (33)</button>
                        <button class="flux-filter-pill" data-filter="men" onclick="window.filterGarments('men')">Men</button>
                        <button class="flux-filter-pill" data-filter="women" onclick="window.filterGarments('women')">Women</button>
                        <button class="flux-filter-pill" data-filter="dresses" onclick="window.filterGarments('dresses')">Dresses & Gowns</button>
                        <button class="flux-filter-pill" data-filter="lehengas" onclick="window.filterGarments('lehengas')">Lehengas</button>
                        <button class="flux-filter-pill" data-filter="suits" onclick="window.filterGarments('suits')">Suits & Sets</button>
                        <button class="flux-filter-pill" data-filter="tops" onclick="window.filterGarments('tops')">Shirts & Tops</button>
                        <button class="flux-filter-pill" data-filter="bottoms" onclick="window.filterGarments('bottoms')">Pants & Shorts</button>
                    </div>
                    <input type="text" id="flux-garment-search" class="flux-search-input" placeholder="🔍 Search clothes (e.g., suit, dress, jacket)..." oninput="window.searchGarments(this.value)" />
                </div>
            </div>
            <div id="garment-gallery-grid" class="flux-garment-grid flux-garment-scroll">
                <!-- Rendered dynamically by JS -->
            </div>
        </div>
    </div>
    """

    with gr.Blocks(title="FluxRT Live Virtual Try-On") as demo:
        gr.Markdown("### ⚡ FluxRT Live Virtual Try-On")
        gr.HTML(ui_html)

        with gr.Accordion("⚙️ Prompt & Custom Garment Upload (Optional)", open=False):
            with gr.Row():
                prompt = gr.Textbox(
                    value=default_prompt,
                    label="Prompt",
                    lines=2,
                    scale=3,
                )

                if use_reference_image:
                    ref_image_input = gr.Image(
                        label="Upload Custom Garment (Overrides preset)",
                        type="numpy",
                        sources=["upload"],
                        image_mode="RGB",
                        scale=2,
                    )

        prompt.change(set_prompt, inputs=prompt, outputs=None)

        if use_reference_image:
            ref_image_input.change(
                set_reference_image_ui,
                inputs=ref_image_input,
                outputs=None,
            )

    # --- Create a custom FastAPI app and register routes on it ---
    server_app = FastAPI()

    # Enable CORS for all LAN devices
    server_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static garments directory
    if os.path.exists(garments_dir):
        server_app.mount("/garments", StaticFiles(directory=garments_dir), name="garments")

    @server_app.get("/api/session_status")
    async def api_session_status():
        """Returns the current GPU try-on session status for multi-device coordination."""
        return session_manager.get_status()

    @server_app.get("/api/garments")
    async def api_get_garments():
        return garments_catalog

    @server_app.post("/api/select_garment")
    async def api_select_garment(request: Request):
        try:
            body = await request.json()
            path = body.get("path")
            if not path:
                return {"success": False, "error": "No garment path provided"}

            full_path = os.path.abspath(os.path.join(repo_root, path))
            garments_dir_abs = os.path.abspath(garments_dir)
            if not full_path.startswith(garments_dir_abs) or not os.path.exists(full_path):
                return {"success": False, "error": "Garment image not found"}

            img_bgr = cv2.imread(full_path)
            if img_bgr is None:
                return {"success": False, "error": "Failed to decode garment image"}

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            set_reference_image_ui(img_rgb)

            garment_name = next((g["name"] for g in garments_catalog if g["file"] == path), os.path.basename(path))
            print(f"[FluxRT] Reference garment updated to: '{garment_name}'", flush=True)
            return {"success": True, "path": path, "name": garment_name}
        except Exception as e:
            print(f"[FluxRT] Error updating reference garment: {e}", flush=True)
            return {"success": False, "error": str(e)}

    @server_app.websocket("/ws_tryon")
    async def websocket_tryon(websocket: WebSocket):
        await websocket.accept()
        client_ip = websocket.client.host if websocket.client else "unknown"
        session_id = f"{client_ip}_{time.time()}"

        # Acquire exclusive GPU try-on lock
        if not session_manager.acquire(session_id, client_ip):
            status = session_manager.get_status()
            print(
                f"[FluxRT WebSocket] Connection rejected for {client_ip}: GPU busy with {status.get('active_ip')}",
                flush=True,
            )
            err_payload = json.dumps({
                "type": "error",
                "code": "GPU_BUSY",
                "message": f"GPU is currently in use by {status.get('active_ip')} ({status.get('elapsed_seconds', 0)}s). Please wait for their session to finish.",
                "active_ip": status.get("active_ip"),
                "elapsed_seconds": status.get("elapsed_seconds", 0),
            })
            await websocket.send_text(err_payload)
            await websocket.close(code=1008, reason="GPU is busy")
            return

        sp, in_tensor, out_tensor, res = get_processor()
        target_h, target_w = res["height"], res["width"]
        print(f"[FluxRT WebSocket] Client {client_ip} granted GPU session! Activating pipeline...", flush=True)
        sp.set_active(True)
        frame_idx = 0
        try:
            while True:
                data = await websocket.receive_bytes()
                if not data:
                    continue
                session_manager.touch(session_id)
                np_arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                frame = crop_maximal_rectangle(frame, target_h, target_w)
                with processor_lock:
                    in_tensor.copy_from(frame)
                    frame_idx += 1
                    if frame_idx == 1:
                        print(f"[FluxRT WebSocket] First webcam frame received from {client_ip}! Processing try-on.", flush=True)
                    elif frame_idx % 100 == 0:
                        print(f"[FluxRT WebSocket] Active streaming for {client_ip}: {frame_idx} frames processed.", flush=True)

                    if not sp.is_alive():
                        exit_code = sp.model_inference_subprocess.process.exitcode if sp.model_inference_subprocess.process else "None"
                        out_frame = frame.copy()
                        cv2.putText(
                            out_frame,
                            f"GPU Inference Crashed! Exit: {exit_code}",
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            out_frame,
                            "Check terminal for error log",
                            (20, 85),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    elif not sp.is_ready():
                        out_frame = frame.copy()
                        cv2.putText(
                            out_frame,
                            "Fitting Garment... Please wait",
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                    else:
                        out_frame = out_tensor.to_numpy()
                _, out_jpeg = cv2.imencode(".jpg", out_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                await websocket.send_bytes(out_jpeg.tobytes())
        except WebSocketDisconnect:
            print(f"[FluxRT WebSocket] Client {client_ip} disconnected. Pausing GPU pipeline.", flush=True)
        except Exception as e:
            print(f"[FluxRT WebSocket Error from {client_ip}]: {e}", flush=True)
        finally:
            session_manager.release(session_id)
            sp.set_active(False)

    def cleanup():
        global stream_processor
        session_manager.force_release()
        if stream_processor is not None:
            print("[FluxRT] Cleaning up and stopping background processes...", flush=True)
            try:
                stream_processor.stop()
            except Exception as e:
                print(f"[FluxRT] Cleanup warning: {e}", flush=True)
            stream_processor = None

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda sig, frame: (cleanup(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda sig, frame: (cleanup(), sys.exit(0)))

    # Mount Gradio onto the FastAPI app (preserves our /ws_tryon route)
    demo.queue(default_concurrency_limit=1)
    server_app = gr.mount_gradio_app(server_app, demo, path="/", head=ws_client_js)

    lan_ip = get_lan_ip()
    try:
        if args.ssl:
            cert_file, key_file = ensure_ssl_certificates(
                cert_path=args.cert,
                key_path=args.key,
                lan_ip=lan_ip,
            )
            print("=" * 66, flush=True)
            print("⚡ FluxRT Live Virtual Try-On LAN Service", flush=True)
            print("🔒 HTTPS Enabled (Secure Context for LAN Webcams)", flush=True)
            print(f"🌐 LAN URL:   https://{lan_ip}:{args.port}", flush=True)
            print(f"💻 Local URL: https://localhost:{args.port}", flush=True)
            print(f"ℹ️  LAN users: Open https://{lan_ip}:{args.port} in Chrome/Safari/Edge.", flush=True)
            print(f"   When prompted with 'Your connection is not private', click 'Advanced'", flush=True)
            print(f"   -> 'Proceed to {lan_ip}' (required for self-signed certificate).", flush=True)
            print("=" * 66, flush=True)

            uvicorn.run(
                server_app,
                host="0.0.0.0",
                port=args.port,
                ssl_certfile=cert_file,
                ssl_keyfile=key_file,
            )
        else:
            print("=" * 66, flush=True)
            print("⚡ FluxRT Live Virtual Try-On LAN Service (Plain HTTP)", flush=True)
            print(f"🌐 LAN URL:  http://{lan_ip}:{args.port}", flush=True)
            print("=" * 66, flush=True)
            uvicorn.run(server_app, host="0.0.0.0", port=args.port)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
