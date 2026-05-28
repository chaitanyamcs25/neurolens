"""
camera_stream.py — Run this on your Raspberry Pi
Streams Camera Module 3 (IMX708) feed at MAXIMUM QUALITY.

Every sensor capability is pushed to the limit:
  • Full 12MP (4608×2592) still capture endpoint
  • 1080p streaming with max JPEG quality
  • Continuous PDAF Autofocus (fast mode)
  • HDR mode (for /capture_hdr endpoint)
  • Saturation, contrast, sharpness all boosted
  • High-quality noise reduction
  • Auto white balance + auto exposure

Install deps on Pi:
  sudo apt update && sudo apt install -y python3-picamera2 python3-libcamera python3-opencv
  pip install flask flask-cors numpy

Run:
  python camera_stream.py

Endpoints:
  /stream          — Live MJPEG stream (1080p, max quality)
  /capture         — Single full 12MP JPEG (4608×2592)
  /capture_hdr     — Single HDR capture (lower res, high dynamic range)
  /health          — Camera status + live sensor metadata
"""

import io
import time
import logging
import threading
import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_file
from flask_cors import CORS

# ─── MAX QUALITY Configuration ───────────────────────────────────────────────
# Stream resolution — 1080p for live feed (Pi encodes this in real-time)
STREAM_RESOLUTION = (1920, 1080)

# Still capture resolution — FULL 12MP sensor output
STILL_RESOLUTION = (4608, 2592)

# JPEG quality — near-lossless (95 = excellent quality, minimal artifacts)
STREAM_JPEG_QUALITY = 95
STILL_JPEG_QUALITY = 98

# Image enhancement — push the sensor to its visual best
SHARPNESS = 2.0           # Max useful sharpness (default 1.0, range 0-16)
SATURATION = 1.2           # Slightly boosted color vibrancy (default 1.0)
CONTRAST = 1.1             # Slightly boosted contrast (default 1.0)
BRIGHTNESS = 0.05          # Tiny brightness lift (default 0.0, range -1 to 1)
ANALOGUE_GAIN_MAX = 12.0   # Allow higher ISO for low light (default ~8)

# Server
PORT = 5001
HOST = '0.0.0.0'
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger('camera_stream')

app = Flask(__name__)
CORS(app)

# ─── Initialize Picamera2 (NO HDR for stream — HDR caps resolution to 3MP) ──
from picamera2 import Picamera2

picam2 = Picamera2()

# Use still_configuration for maximum quality per frame
# still_configuration uses the full sensor readout path (higher quality than video)
stream_config = picam2.create_still_configuration(
    main={"size": STREAM_RESOLUTION, "format": "RGB888"},
    buffer_count=3,
)
picam2.configure(stream_config)
picam2.start()
log.info(f"📷 Camera started at {STREAM_RESOLUTION[0]}×{STREAM_RESOLUTION[1]} (still pipeline)")

# Sensor stabilization
time.sleep(2)

# ─── Autofocus — Continuous PDAF (Camera Module 3 exclusive) ─────────────────
AF_ENABLED = False
try:
    from libcamera import controls

    picam2.set_controls({
        "AfMode": controls.AfModeEnum.Continuous,
        "AfSpeed": controls.AfSpeedEnum.Fast,
        "AfMetering": controls.AfMeteringEnum.Windows,  # Focus on center region
    })
    AF_ENABLED = True
    log.info("✅ Continuous PDAF autofocus enabled (fast, center-weighted)")
except ImportError:
    log.warning("⚠️  libcamera controls not available — autofocus skipped")
except Exception as e:
    log.warning(f"⚠️  Autofocus init failed: {e}")

# ─── MAX Image Quality Controls ──────────────────────────────────────────────
try:
    from libcamera import controls

    # Core image processing
    picam2.set_controls({
        "AwbEnable": True,                  # Auto white balance ON
        "AeEnable": True,                   # Auto exposure ON
        "Sharpness": SHARPNESS,             # Boosted sharpness
        "Saturation": SATURATION,           # Boosted color
        "Contrast": CONTRAST,              # Boosted contrast
        "Brightness": BRIGHTNESS,           # Slight brightness lift
        "AnalogueGain": 1.0,               # Start at base ISO (cleanest)
        "ExposureValue": 0.0,              # Neutral EV compensation
    })
    log.info(f"✅ Image quality maxed — Sharpness:{SHARPNESS} Saturation:{SATURATION} Contrast:{CONTRAST}")

    # High-quality noise reduction
    try:
        picam2.set_controls({
            "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
        })
        log.info("✅ Noise reduction: HighQuality")
    except (AttributeError, Exception):
        log.info("ℹ️  NoiseReductionMode not available on this libcamera version")

except Exception as e:
    log.warning(f"⚠️  Image quality tuning skipped: {e}")

# JPEG encode params (reusable)
STREAM_JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
STILL_JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, STILL_JPEG_QUALITY]


# ─── Streaming ───────────────────────────────────────────────────────────────
def generate_frames():
    """Yield MJPEG frames at maximum quality."""
    while True:
        # capture_array — direct NumPy array, no disk I/O
        array = picam2.capture_array()

        # RGB → BGR for OpenCV encoding
        bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

        # Encode to JPEG at 95% quality
        ret, jpeg = cv2.imencode('.jpg', bgr, STREAM_JPEG_PARAMS)
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
        )


@app.route('/stream')
def video_feed():
    """Live MJPEG video stream at max quality 1080p."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ─── Full 12MP Still Capture ─────────────────────────────────────────────────
capture_lock = threading.Lock()


@app.route('/capture')
def capture_still():
    """
    Capture a single FULL 12MP (4608×2592) JPEG.
    Temporarily switches to full-res mode, captures, then switches back.
    """
    with capture_lock:
        try:
            # Stop current stream config
            picam2.stop()

            # Switch to full 12MP still configuration
            still_config = picam2.create_still_configuration(
                main={"size": STILL_RESOLUTION, "format": "RGB888"},
            )
            picam2.configure(still_config)
            picam2.start()
            time.sleep(1)  # Let AE/AF settle at new resolution

            # Re-apply AF
            if AF_ENABLED:
                try:
                    picam2.set_controls({
                        "AfMode": controls.AfModeEnum.Auto,
                        "AfTrigger": controls.AfTriggerEnum.Start,
                    })
                    time.sleep(2)  # Wait for AF to lock
                except Exception:
                    pass

            # Capture full-res frame
            array = picam2.capture_array()
            bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            ret, jpeg = cv2.imencode('.jpg', bgr, STILL_JPEG_PARAMS)

            # Switch back to stream config
            picam2.stop()
            picam2.configure(stream_config)
            picam2.start()
            time.sleep(1)

            # Re-apply continuous AF for streaming
            if AF_ENABLED:
                try:
                    picam2.set_controls({
                        "AfMode": controls.AfModeEnum.Continuous,
                        "AfSpeed": controls.AfSpeedEnum.Fast,
                    })
                except Exception:
                    pass

            # Re-apply image quality controls
            try:
                picam2.set_controls({
                    "Sharpness": SHARPNESS,
                    "Saturation": SATURATION,
                    "Contrast": CONTRAST,
                    "Brightness": BRIGHTNESS,
                })
            except Exception:
                pass

            if ret:
                buf = io.BytesIO(jpeg.tobytes())
                buf.seek(0)
                return send_file(
                    buf,
                    mimetype='image/jpeg',
                    as_attachment=True,
                    download_name=f'picam3_12mp_{int(time.time())}.jpg'
                )
            else:
                return jsonify({'error': 'JPEG encoding failed'}), 500

        except Exception as e:
            # Ensure we recover to stream mode on error
            try:
                picam2.stop()
                picam2.configure(stream_config)
                picam2.start()
            except Exception:
                pass
            return jsonify({'error': str(e)}), 500


# ─── HDR Capture (separate endpoint — HDR caps resolution) ───────────────────
@app.route('/capture_hdr')
def capture_hdr():
    """
    Capture a single HDR image.
    HDR merges multiple exposures for higher dynamic range.
    Note: HDR limits output to ~3MP due to IMX708 processing pipeline.
    """
    with capture_lock:
        try:
            picam2.stop()

            # Enable HDR mode
            hdr_enabled = False
            try:
                from picamera2.devices.imx708 import IMX708
                with IMX708(0) as imx:
                    imx.set_sensor_hdr_mode(True)
                hdr_enabled = True
                log.info("HDR mode activated for capture")
            except Exception as e:
                log.warning(f"HDR not available: {e}")

            # HDR caps at roughly 2304×1296
            hdr_config = picam2.create_still_configuration(
                main={"size": (2304, 1296), "format": "RGB888"},
            )
            picam2.configure(hdr_config)
            picam2.start()
            time.sleep(2)  # HDR needs longer to settle

            # Capture HDR frame
            array = picam2.capture_array()
            bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            ret, jpeg = cv2.imencode('.jpg', bgr, STILL_JPEG_PARAMS)

            # Disable HDR and switch back to stream mode
            picam2.stop()
            if hdr_enabled:
                try:
                    from picamera2.devices.imx708 import IMX708
                    with IMX708(0) as imx:
                        imx.set_sensor_hdr_mode(False)
                except Exception:
                    pass

            picam2.configure(stream_config)
            picam2.start()
            time.sleep(1)

            # Re-apply controls
            if AF_ENABLED:
                try:
                    picam2.set_controls({
                        "AfMode": controls.AfModeEnum.Continuous,
                        "AfSpeed": controls.AfSpeedEnum.Fast,
                    })
                except Exception:
                    pass
            try:
                picam2.set_controls({
                    "Sharpness": SHARPNESS,
                    "Saturation": SATURATION,
                    "Contrast": CONTRAST,
                    "Brightness": BRIGHTNESS,
                })
            except Exception:
                pass

            if ret:
                buf = io.BytesIO(jpeg.tobytes())
                buf.seek(0)
                return send_file(
                    buf,
                    mimetype='image/jpeg',
                    as_attachment=True,
                    download_name=f'picam3_hdr_{int(time.time())}.jpg'
                )
            else:
                return jsonify({'error': 'HDR JPEG encoding failed'}), 500

        except Exception as e:
            try:
                picam2.stop()
                picam2.configure(stream_config)
                picam2.start()
            except Exception:
                pass
            return jsonify({'error': str(e)}), 500


# ─── Health + Live Metadata ──────────────────────────────────────────────────
@app.route('/health')
def health():
    """Full camera status with live sensor metadata."""
    info = {
        'status': 'ok',
        'camera': 'Camera Module 3 (IMX708) — MAX QUALITY',
        'stream_resolution': f"{STREAM_RESOLUTION[0]}×{STREAM_RESOLUTION[1]}",
        'still_resolution': f"{STILL_RESOLUTION[0]}×{STILL_RESOLUTION[1]}",
        'autofocus': 'Continuous PDAF (fast)' if AF_ENABLED else 'Disabled',
        'stream_jpeg_quality': f"{STREAM_JPEG_QUALITY}%",
        'still_jpeg_quality': f"{STILL_JPEG_QUALITY}%",
        'sharpness': SHARPNESS,
        'saturation': SATURATION,
        'contrast': CONTRAST,
        'brightness': BRIGHTNESS,
        'endpoints': {
            'stream': f'http://{HOST}:{PORT}/stream',
            'capture_12mp': f'http://{HOST}:{PORT}/capture',
            'capture_hdr': f'http://{HOST}:{PORT}/capture_hdr',
            'health': f'http://{HOST}:{PORT}/health',
        }
    }

    # Live sensor metadata
    try:
        metadata = picam2.capture_metadata()
        info['live_sensor'] = {
            'lens_position': metadata.get('LensPosition', 'N/A'),
            'af_state': metadata.get('AfState', 'N/A'),
            'exposure_time_us': metadata.get('ExposureTime', 'N/A'),
            'analogue_gain': round(metadata.get('AnalogueGain', 0), 2),
            'digital_gain': round(metadata.get('DigitalGain', 0), 2),
            'colour_temperature_K': metadata.get('ColourTemperature', 'N/A'),
            'lux': metadata.get('Lux', 'N/A'),
            'focus_fom': metadata.get('FocusFoM', 'N/A'),
            'sensor_temperature': metadata.get('SensorTemperature', 'N/A'),
        }
    except Exception:
        pass

    return jsonify(info)


# ─── Startup ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    log.info("=" * 65)
    log.info("  🔥 Pi Camera Module 3 — MAX QUALITY MJPEG Server")
    log.info(f"  Stream     : {STREAM_RESOLUTION[0]}×{STREAM_RESOLUTION[1]} @ JPEG {STREAM_JPEG_QUALITY}%")
    log.info(f"  Stills     : {STILL_RESOLUTION[0]}×{STILL_RESOLUTION[1]} @ JPEG {STILL_JPEG_QUALITY}%")
    log.info(f"  Autofocus  : {'Continuous PDAF (fast)' if AF_ENABLED else 'Disabled'}")
    log.info(f"  Sharpness  : {SHARPNESS}  Saturation: {SATURATION}  Contrast: {CONTRAST}")
    log.info(f"  Endpoints:")
    log.info(f"    Stream   → http://{HOST}:{PORT}/stream")
    log.info(f"    12MP     → http://{HOST}:{PORT}/capture")
    log.info(f"    HDR      → http://{HOST}:{PORT}/capture_hdr")
    log.info(f"    Health   → http://{HOST}:{PORT}/health")
    log.info("=" * 65)
    app.run(host=HOST, port=PORT, threaded=True)
