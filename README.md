# Fall and Stroke Warning System

**Real-time, privacy-first camera monitoring that warns families when an elderly person falls, collapses, convulses, or shows visible signs of acute illness.**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime%20(DirectML)-005CED.svg)](https://onnxruntime.ai/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6.svg)](#installation)
[![Version](https://img.shields.io/badge/version-0.3.4-informational.svg)](src/version.py)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Plug in a webcam or point it at an IP camera. The app tracks every person in view, reads their body pose, analyses motion over time, and sends a Telegram alert (with a snapshot and a short video clip) when it sees something that needs a human to check. Everything runs on the user's own computer. No video leaves the machine.

> [!IMPORTANT]
> **Safety alerting, not medical diagnosis.** The system detects *observable warning signs* (a fall, a collapse, rhythmic convulsion-like shaking, staggering, clutching the head or chest, prolonged stillness after a fall). It never claims that someone has had a stroke, a seizure or a fracture. Every alert asks a human to verify: *"Possible medical emergency / abnormal behavior detected. Please check the person."* It is an extra pair of eyes and **must not replace** professional care or emergency services.

---

## Table of contents

- [Why this project](#why-this-project)
- [Key features](#key-features)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Using the dashboard](#using-the-dashboard)
- [Configuration](#configuration)
- [HTTP / WebSocket API](#http--websocket-api)
- [Security and privacy](#security-and-privacy)
- [Training the models](#training-the-models)
- [Model performance and known limitations](#model-performance-and-known-limitations)
- [Testing](#testing)
- [Project structure](#project-structure)
- [Release, code signing and auto-update](#release-code-signing-and-auto-update)
- [Documentation](#documentation)
- [License and third-party notices](#license-and-third-party-notices)
- [Hướng dẫn nhanh bằng tiếng Việt](#hướng-dẫn-nhanh-bằng-tiếng-việt)

---

## Why this project

Falls are one of the leading causes of injury among older adults, and the real danger is often the **time spent lying on the floor unnoticed**. Many elderly people live alone or are home by themselves during the day, while their families are at work. Wearable alarm buttons only help if the person is conscious and remembers to press them, and cloud camera services send private home video to a third party.

This project takes a different approach:

- **Camera-based and hands-free.** Nothing to wear, nothing to press.
- **Local-first.** Video is processed on the user's PC. Only a short text alert (and optionally a snapshot or clip) goes to the Telegram bot that the user configures.
- **Built to avoid false alarms.** A family that gets false alerts every day will stop trusting the system, so the risk logic is deliberately conservative and debounced (see [How it works](#how-it-works)).
- **Easy enough for non-technical families.** One-click Windows installer, no Python and no command line for end users.

---

## Key features

**Perception**
- **Multiple video sources:** USB / built-in webcams, RTSP / RTSPS / RTMP / HTTP(S) IP cameras (MJPEG, HLS), and local MP4 / AVI files for testing.
- **Self-healing camera streams:** automatic reconnect with exponential backoff (RTSP up to 30 s between tries, retries forever for 24/7 monitoring). If another app such as Zalo, Teams or Phone Link grabs the webcam, the stream is reopened automatically.
- **Multi-camera:** add extra cameras from the web UI or `configs/camera.yaml`.
- **Camera discovery and diagnostics:** scan the user's own LAN for RTSP cameras and get readable diagnostics when a camera cannot be opened.
- **Single-pass pose and tracking:** RTMO-s pose model (17 COCO keypoints, ONNX) with ByteTrack-style multi-person tracking so each person keeps a stable ID.

**Behaviour understanding**
- **Spatial-Temporal Transformer** over a 30-frame sliding window (about 2 s at 15 Hz) that classifies 10 actions: `walking`, `standing`, `sitting`, `lying`, `bending`, `falling`, `getting_up`, `stumbling`, `abnormal_movement`, `immobile`.
- **Unsupervised pose autoencoder** that scores how unusual a movement sequence is.
- **Kinematic signals:** downward velocity and acceleration, torso angle and angular velocity, bounding-box aspect ratio, distance of the hips to the floor.
- **Convulsion-like shaking detector:** rhythmic movement (1.8 to 6 Hz) in at least 3 body parts at the same tempo, sustained for at least 5 s. It is designed to ignore walking, clapping, waving and phone use.
- **Lying-context analysis:** tells apart lying down on purpose (sit first, then lie down slowly, already lying when monitoring started) from a collapse (fast drop from standing, unsteady before going down).
- **Supported-posture filter:** bending to pick something up, squatting, kneeling and crawling are not treated as falls because the legs are still holding the body.

**Decision and alerting**
- **Multi-signal risk fusion** (action model 0.35, kinematics 0.35, anomaly 0.15, immobility 0.15) feeding a **debounced finite state machine** with hysteresis: `NORMAL → SUSPICIOUS → ABNORMAL → HIGH_RISK → ALERT_SENT`.
- **Minimum 3.5 s of sustained high risk before an alarm**, so bends, sitting down quickly or turning over in bed cannot trigger one.
- **Telegram alerts** with snapshot and recorded event clip, set up from the UI (including auto-detect of the chat ID and a "send test message" button).
- **Event history** with snapshots and video clips, viewable and deletable from the dashboard.

**Efficiency**
- **Tier-0 motion gating** (MOG2 background subtraction): when the room is static, the system throttles to about 12 FPS and skips the heavy AI work, then jumps to full speed when something moves.
- **ONNX Runtime with DirectML:** GPU acceleration on any DirectX 12 GPU (NVIDIA, AMD or Intel), with automatic CPU fallback. No CUDA install and no PyTorch needed at runtime (the runtime stack is about 700 MB instead of several GB).
- **Decoupled pipeline:** capture and display run separately from inference so the video stays smooth.

**Product**
- **Web dashboard** at `http://localhost:8000` with live MJPEG video, live alerts over WebSocket, camera management and settings.
- **Native desktop window** in the packaged app, which keeps running in the background when the window is closed so alerts keep flowing.
- **Auto-start with Windows**, toggled from the settings screen.
- **Built-in updater** that checks GitHub Releases and installs new versions silently.
- **Authentication and hardening** for access from other devices on the network (see [Security and privacy](#security-and-privacy)).

---

## How it works

```text
Camera (webcam / RTSP / video file)
   │
   ▼
Frame reader with auto-reconnect and zero-lag buffer
   │
   ▼
Tier-0 motion gating (MOG2)  ── room static → throttle to ~12 FPS, skip AI
   │ motion
   ▼
RTMO-s single-pass detector + pose  → person boxes + 17 keypoints
   │
   ▼
ByteTrack tracker → stable track ID per person
   │
   ▼
Keypoint normalisation + One-Euro smoothing filter
   │
   ▼
Ring buffer (T = 30 frames @ 15 Hz)
   │
   ▼
┌───────────────────────────────────────────────────────┐
│ 1. ST-Transformer action classifier (10 classes)      │
│ 2. Pose autoencoder (anomaly score)                   │
│ 3. Kinematics: drop speed, torso angle, aspect ratio  │
│ 4. Lying context, supported posture, convulsion rhythm│
└───────────────────────────────────────────────────────┘
   │
   ▼
Risk engine: 0.35·action + 0.15·anomaly + 0.35·kinematics + 0.15·immobility
   │
   ▼
Debounced state machine (NORMAL → SUSPICIOUS → ABNORMAL → HIGH_RISK → ALERT_SENT)
   │
   ▼
Alert manager → Telegram (snapshot + clip) · WebSocket · dashboard · event log
```

### Risk thresholds (defaults in `configs/thresholds.yaml`)

| Stage | Risk score at least | Must hold for |
|---|---|---|
| `SUSPICIOUS` | 0.60 | 0.5 s |
| `ABNORMAL` | 0.76 | 1.5 s |
| `HIGH_RISK` | 0.88 | 2.0 s |
| Back to `NORMAL` | risk below 0.60 with walking / standing / getting up | 3.0 s |

Escalation requires the risk to stay high for the **whole** debounce window, and the physical drop signal and the action model must **both** agree before they can override the weighted score. One noisy signal alone cannot trigger an alarm.

### Monitoring scope for people who are lying down

By default (`monitoring_scope.safe_when_lying: true`) a person who lies down **deliberately** (sat down first, lowered themselves slowly, or was already lying when monitoring started) is treated as safe, with risk 0 and no alarm. The exception is lying down because of a **fall, a collapse or convulsions**. Set `safe_when_lying: false` to analyse lying people in full (trying to get up, motionless on the floor, and so on).

Note that a person who has a stroke while asleep and does not move cannot be told apart from a person sleeping using video alone. The system does not claim otherwise.

---

## Installation

### Requirements

| | Packaged app (recommended) | From source |
|---|---|---|
| OS | Windows 10 / 11 (64-bit) | Windows 10 / 11 |
| Python | Not needed | 3.11 |
| GPU | Optional, any DirectX 12 GPU (CPU fallback works) | Same |
| Camera | USB / built-in webcam or IP camera (RTSP) | Same |
| Disk | roughly 1 GB | roughly 1 GB |

### Option A: Windows installer (for end users)

1. Download `Fall_and_Stroke_Warning_System_Setup.exe` from the public releases page: [QuocCuong66/elderly-ai-installer → Releases](https://github.com/QuocCuong66/elderly-ai-installer/releases).
2. Run it. It installs to your user folder and **does not need administrator rights**.
3. Open **Fall and Stroke Warning System** from the Start menu or desktop.

The installer is code-signed through [SignPath Foundation](#release-code-signing-and-auto-update).

### Option B: Run from source with the helper scripts

```bat
git clone https://github.com/QuocCuong66/Project-monitoring-model.git
cd Project-monitoring-model

1_Cai_Dat_Tu_Dong.bat      :: creates .venv and installs requirements.txt (about 1-2 minutes)
2_Chay_He_Thong.bat        :: starts the system
```

### Option C: Run from source manually

```powershell
git clone https://github.com/QuocCuong66/Project-monitoring-model.git
cd Project-monitoring-model

py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

python main.py
```

The three ONNX models (`rtmo-s.onnx`, `best_st_transformer.onnx`, `best_pose_autoencoder.onnx`) ship inside the repository under `models/`, so there is nothing else to download.

---

## Quick start

```powershell
# Default: first webcam, dashboard on http://localhost:8000
python main.py

# IP camera
python main.py --source rtsp --rtsp-url "rtsp://user:password@192.168.1.100:554/stream1"

# Test with a video file
python main.py --source video --video-path "data/videos/test_fall.mp4"

# Headless server mode, no OpenCV window
python main.py --no-gui

# Other webcam, other port, low-power mode
python main.py --device-index 1 --api-port 8080 --fast
```

### Command-line options

| Flag | Default | Meaning |
|---|---|---|
| `--source {webcam,rtsp,video}` | `webcam` | Video source of the main camera |
| `--device-index N` | `0` | Webcam index |
| `--rtsp-url URL` | none | RTSP stream address |
| `--video-path PATH` | none | MP4 / AVI file to analyse |
| `--blur-faces` | off | Blur faces in the output |
| `--gui` | off | Also open an OpenCV desktop window |
| `--no-gui` | off | Headless mode |
| `--no-server` | off | Do not start the FastAPI server |
| `--api-port N` | `8000` | Port of the web server |
| `--fast` | off | Lightweight mode (nano model at 480 px) |
| `--sequential` | off | Old single-thread mode instead of the decoupled pipeline |

### Step-by-step first run for a family

1. Connect the webcam and double-click the app.
2. Place the camera so it sees the main living area, at about chest height or higher, with the whole body visible.
3. Open **Settings (⚙)** and set up **Telegram** (see below).
4. Turn on **Auto-start with Windows** so protection resumes after every reboot.
5. Close the window if you like: the system keeps running in the background. Reopen it any time at `http://localhost:8000`.

### Setting up Telegram alerts

1. In Telegram, talk to **@BotFather**, create a bot and copy the **bot token**.
2. Send any message to your new bot from the account (or group) that should receive alerts.
3. In the app: **Settings → Telegram**, paste the token and use **auto-detect chat ID**.
4. Press **Send test message**.

The token and chat ID are stored outside the install folder, in `%APPDATA%\FallStrokeWarning`, so they are never packed into the installer or committed to git and survive updates.

---

## Using the dashboard

Open `http://localhost:8000` on the computer that runs the system.

- **Live view** with a bounding box, track ID, current action label and a risk gauge for each person.
- **Alert feed** that updates instantly over WebSocket.
- **Event history** with snapshots and replayable clips, with single or bulk delete.
- **Camera panel:** switch cameras, test a connection, scan the LAN for cameras, add extra cameras, pause or resume the stream.
- **Settings:** face blur, auto-start, Telegram, camera permission check, account and password, update check, and a red **Shut down system** button.

Closing the window does **not** stop monitoring. Use **Shut down system** in the settings to stop it completely.

---

## Configuration

Everything lives in `configs/` (YAML) plus an optional `.env`. Copy `.env.example` to `.env` to override values.

| File | What it controls |
|---|---|
| `configs/camera.yaml` | Source type, webcam size / FPS, RTSP URL, transport (TCP / UDP), timeouts, reconnect backoff, LAN discovery ports, extra cameras |
| `configs/model.yaml` | Model paths, pose and tracker thresholds, motion gating, kinematic gate, temporal model size and window |
| `configs/inference.yaml` | Device (CUDA / DirectML / CPU), target FPS, scheduling, temporal sampling rate, UI overlay options |
| `configs/thresholds.yaml` | Risk weights, state-machine thresholds and durations, lying scope, convulsion detector, kinematic reliability, lying context |
| `configs/notifications.yaml` | Telegram API settings, timeouts, where user settings are stored |
| `configs/security.yaml` | Authentication, token lifetimes, host check, public paths, CORS |
| `configs/update.yaml` | Auto-update source, allowed download hosts, silent-install arguments |
| `configs/training.yaml` | Training hyper-parameters |
| `configs/qa_targets.yaml` | Quality targets used by the evaluation scripts |

### Tuning tips

- **Too many false alarms?** Raise `thresh_abnormal` / `thresh_high_risk` or lengthen `min_duration_high_risk` in `configs/thresholds.yaml`.
- **Missing real events?** Lower the thresholds slowly and test with recorded videos (`--source video`).
- **Low-end PC?** Use `--fast`, keep motion gating on, and lower the webcam resolution in `configs/camera.yaml`.
- **Remote camera over the Internet?** Prefer a VPN (Tailscale or ZeroTier) to port-forwarding, and keep `transport: tcp`.

---

## HTTP / WebSocket API

The FastAPI server is started together with the app (default `http://localhost:8000`).

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/status` | Current system status |
| `GET` | `/stream/mjpeg` | Live MJPEG video |
| `WS` | `/ws/alerts` | Real-time alert stream |
| `GET` | `/events`, `/events/{id}`, `/events/{id}/clip` | Event list, details and recorded clip |
| `DELETE` / `POST` | `/events`, `/events/{id}`, `/events/clear` | Delete events |
| `GET` | `/camera/devices`, `/camera/info`, `/camera/discover` | List, inspect and discover cameras |
| `POST` | `/camera/switch`, `/camera/test`, `/camera/start`, `/camera/stop`, `/camera/reset`, `/camera/save-config` | Control the main camera |
| `GET` / `POST` / `DELETE` | `/api/cameras`, `/api/cameras/{id}`, `/api/cameras/{id}/stream` | Manage extra cameras |
| `GET` / `POST` | `/api/camera/blur` | Face blur on / off |
| `GET` / `POST` | `/api/settings/autostart` | Start with Windows |
| `POST` | `/api/settings/check-camera`, `/api/settings/open-camera-settings` | Camera diagnostics and OS permission page |
| `GET` / `POST` | `/api/telegram`, `/api/telegram/detect-chat`, `/api/telegram/test` | Telegram setup |
| `POST` | `/api/auth/setup`, `/login`, `/refresh`, `/logout`, `/change-password`; `GET` `/api/auth/status`, `/me` | Authentication |
| `GET` / `POST` | `/api/update/status`, `/check`, `/download`, `/install` | Updater (admin only for actions) |
| `POST` | `/api/system/shutdown` | Stop the whole system |

Requests from `localhost` are allowed without a login; requests from other devices need a token (see below).

---

## Security and privacy

**Privacy by design**
- Camera frames are **processed locally** and are never uploaded.
- Alerts go **only** to the Telegram bot or webhook that the user configures. The program makes no other outbound connections except the optional update check against GitHub Releases.
- A **face-blur** switch is available in the UI and via `--blur-faces`.
- Snapshots and clips are stored only on the local disk, and old events can be deleted from the dashboard.

**Network hardening** (relevant when you open the dashboard from a phone or another PC)
- **JWT authentication** (HS256) with short-lived access tokens (60 min) and refresh tokens (30 days). Changing the password or logging out revokes all tokens at once.
- **Passwords hashed with bcrypt.**
- **Rate limiting** on sensitive routes (slowapi).
- **LAN authentication required:** other devices on Wi-Fi / LAN must log in. The first admin account can only be created on the machine that runs the system, or with `python -m src.auth.cli create-admin <username>` on a server.
- **DNS-rebinding protection:** only requests addressed to the machine's own names are accepted. Add custom domains in `allowed_hosts` or the `ALLOWED_HOSTS` environment variable.
- The JWT secret is generated on first run and stored in `%APPDATA%\FallStrokeWarning\secret.key`.
- **Safe updates:** the updater only downloads from GitHub's own hosts and limits the installer size.

> Do **not** expose port 8000 directly to the Internet. If you need remote access, use a VPN (Tailscale, ZeroTier) and keep authentication on.

To report a security problem, please open a private security advisory on the repository's **Security** tab, or contact the maintainer by email.

---

## Training the models

The action ST-Transformer and the pose autoencoder are trained on the 2D (COCO-17) skeletons of [NTU RGB+D 60](https://rose1.ntu.edu.sg/dataset/actionRecognition/) released by MMAction2. Its medical classes cover visible warning signs of stroke and acute illness: falling, staggering (loss of balance), clutching the head, chest pain, nausea or vomiting.

```powershell
# 1. Download the 2D skeleton annotations
curl -L -o data\raw\ntu\ntu60_2d.pkl https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl

# 2. Training needs PyTorch (inference does not)
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3. Prepare windows, train, export
.venv\Scripts\python.exe training\prepare_ntu.py      # -> data\processed\ntu_windows.npz
.venv\Scripts\python.exe training\train_action.py     # -> models\checkpoints\best_st_transformer.onnx + metrics json
.venv\Scripts\python.exe training\train_anomaly.py    # -> best_pose_autoencoder.onnx + calibration json
```

After training the autoencoder, copy the calibration values into `configs/model.yaml`. Use `training/evaluate.py` and `scripts/benchmark.py` to measure accuracy and latency, and `scripts/export_onnx.py` / `scripts/export_tensorrt.py` for model export.

---

## Model performance and known limitations

**Action classifier** (cross-subject validation, i.e. people never seen in training, current checkpoint):

| Metric | Value |
|---|---|
| Accuracy | 0.90 |
| Macro-F1 | 0.86 |
| Recall: falling | 0.99 |
| Recall: staggering | 0.82 |
| Recall: clutching head / chest pain / nausea | 0.89 |
| F1: lying, immobile | about 0.6 (weaker) |

**Please read these limitations honestly:**

- NTU RGB+D has no *lying* or *immobile* class, so those classes are synthesised from post-fall frames and rotated poses and are weaker.
- The pose autoencoder does **not** separate abnormal from normal movement on NTU (AUROC about 0.5). That is why its risk weight is kept low (0.15).
- There is **no public video dataset of real strokes**. The system flags observable warning signs and never diagnoses.
- Pose estimation from a 2D camera degrades with heavy occlusion (furniture, blankets), very low light, or a camera that cannot see the whole body.
- Anything that happens outside the camera's view, or in a room without a camera, is not detected.
- Results have been verified on the project's own tests and demonstration videos, not in a clinical trial. Treat it as an assistive tool.

---

## Testing

```powershell
pip install pytest
python -m pytest tests -q
```

The test suite covers, among others: the risk state machine, false-alarm guards, normal-activity filters, lying context and lying-safe scope, convulsion detection, fall-versus-stroke stream separation, action fusion, sequence features, multi-camera handling, webcam reconnect and device discovery, camera diagnostics and settings, remote cameras, event clips, Telegram notifications, privacy storage, authentication security, the updater, and ONNX Runtime provider selection.

To inspect a specific video offline: `python tests/diagnose_video.py <video>`.

---

## Project structure

```text
.
├── main.py                    # Entry point: pipeline + web server (+ optional OpenCV window)
├── app_runner.py              # Entry point of the packaged desktop app
├── configs/                   # YAML configuration (camera, model, thresholds, security, ...)
├── models/                    # Bundled ONNX models (pose, ST-Transformer, autoencoder)
├── src/
│   ├── camera/                # Webcam / RTSP / video readers, discovery, diagnostics, permissions
│   ├── detection/             # Person detection, motion gating (Tier 0)
│   ├── pose/                  # RTMO-s pose estimator, keypoints, One-Euro smoothing
│   ├── tracking/              # ByteTrack-based tracker and track manager
│   ├── temporal/              # Sequence buffer/features, ST-Transformer, gated trigger,
│   │                          #   lying context, stroke signs, convulsion detector
│   ├── action/                # Action and behaviour classifiers
│   ├── anomaly/               # Pose autoencoder and anomaly scoring
│   ├── risk/                  # Risk engine, threshold manager, debounced state machine
│   ├── alerts/                # Alert manager, Telegram, event logger, clip recorder
│   ├── pipeline/              # Real-time, decoupled (60 FPS) and multi-camera pipelines
│   ├── api/                   # FastAPI server and Telegram routes
│   ├── auth/                  # JWT, users, middleware, request guard, admin CLI
│   ├── update/                # In-app updater (GitHub Releases)
│   └── utils/                 # Config, privacy/face blur, ORT providers, profiler, SSL
├── frontend/                  # Web dashboard (HTML / CSS / JS)
├── landing/, docs/            # Landing page and user guide (GitHub Pages)
├── training/                  # NTU preparation, action / anomaly / temporal training, evaluation
├── inference/                 # Batch and real-time inference scripts
├── scripts/                   # Benchmarks, export, dataset tools, Windows auto-start helpers
├── installer/                 # Inno Setup script (Vietnamese + English)
├── tests/                     # Pytest suite
├── docs-dev/                  # Architecture, model selection, dataset, training, evaluation
├── .github/workflows/         # Build, sign and release pipeline
├── 1_Cai_Dat_Tu_Dong.bat      # One-click environment setup
├── 2_Chay_He_Thong.bat        # One-click start
├── Build_App_PyInstaller.bat  # Build the standalone .exe and installer
└── requirements.txt
```

---

## Release, code signing and auto-update

- **Build:** `Build_App_PyInstaller.bat` packages the app with PyInstaller (`fall_and_stroke_monitor.spec`) and builds the Windows installer with Inno Setup (`installer/setup.iss`).
- **Release:** pushing a tag such as `v0.3.4` (it must match the version in `pyproject.toml`) runs the [GitHub Actions workflow](.github/workflows/release.yml), which builds on `windows-latest`, signs the binaries and publishes them to GitHub Releases.
- **Auto-update:** the app checks GitHub Releases once at startup (cached for 60 minutes) and can download and silently install a newer version from **Settings → Update**.

### Code signing policy

Free code signing provided by [SignPath.io](https://about.signpath.io), certificate by [SignPath Foundation](https://signpath.org).

Windows release binaries (`Fall_and_Stroke_Warning_System.exe` and `Fall_and_Stroke_Warning_System_Setup.exe`) are built from this repository by the GitHub Actions release workflow and signed through SignPath. Only binaries built from this repository's source are signed; bundled third-party libraries keep their original publishers' signatures.

**Team roles**
- Committers and reviewers: [QuocCuong66](https://github.com/QuocCuong66)
- Approvers: [QuocCuong66](https://github.com/QuocCuong66)

**Privacy policy**
This program will not transfer any information to other networked systems unless specifically requested by the user or the person installing or operating it. Camera frames are processed locally. Alerts are sent only to the Telegram bot or webhook that the user configures.

---

## Documentation

- [`docs-dev/ARCHITECTURE.md`](docs-dev/ARCHITECTURE.md): system architecture, data flows and the state machine
- [`docs-dev/BEHAVIOR_CLASS_CONTRACT.md`](docs-dev/BEHAVIOR_CLASS_CONTRACT.md): the verified contract for the 10 behaviour classes across the pipeline
- [`docs-dev/MODEL_SELECTION.md`](docs-dev/MODEL_SELECTION.md): benchmark matrix and the reasons for each model choice
- [`docs-dev/DATASET.md`](docs-dev/DATASET.md): action taxonomy, public benchmarks, hard negatives and leak-free splits
- [`docs-dev/TRAINING.md`](docs-dev/TRAINING.md): training pipeline, augmentations, losses and ONNX / TensorRT export
- [`docs-dev/EVALUATION.md`](docs-dev/EVALUATION.md): accuracy metrics, latency profiler and benchmark reports
- [`HUONG_DAN_SU_DUNG.txt`](HUONG_DAN_SU_DUNG.txt) and the [online user guide](https://quoccuong66.github.io/Project-monitoring-model/huong-dan.html) (Vietnamese): installing, connecting IP cameras and Telegram, troubleshooting

---

## License and third-party notices

The source code is released under the [MIT License](LICENSE).

Bundled model weights keep their original licenses:

- **RTMO-s pose model** from [OpenMMLab MMPose](https://github.com/open-mmlab/mmpose): Apache-2.0.
- **Temporal models** (`best_st_transformer.onnx`, `best_pose_autoencoder.onnx`) are trained on NTU RGB+D, which is released for **non-commercial research use only**. These weights inherit that restriction. Retrain on data you are allowed to use commercially before any commercial deployment.

Built with [ONNX Runtime](https://onnxruntime.ai/), [OpenCV](https://opencv.org/), [FastAPI](https://fastapi.tiangolo.com/), [ByteTrack](https://github.com/ifzhang/ByteTrack), [MMPose / RTMO](https://github.com/open-mmlab/mmpose) and [NTU RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/).

---

## Hướng dẫn nhanh bằng tiếng Việt

**Hệ thống cảnh báo té ngã và đột quỵ** dùng camera (webcam hoặc camera IP) để phát hiện người cao tuổi bị té, đổ gục, co giật hoặc có dấu hiệu bất thường, rồi gửi cảnh báo qua Telegram kèm ảnh và đoạn video ngắn. Toàn bộ video được xử lý ngay trên máy tính của bạn, không gửi lên đám mây.

> ⚠️ Hệ thống chỉ **cảnh báo dấu hiệu nhìn thấy được**, không chẩn đoán bệnh và **không thay thế** chăm sóc y tế hay số cấp cứu.

**Cài đặt nhanh (người dùng cuối)**
1. Tải `Fall_and_Stroke_Warning_System_Setup.exe` tại [trang Releases](https://github.com/QuocCuong66/elderly-ai-installer/releases) và cài đặt (không cần quyền Admin).
2. Cắm webcam, mở ứng dụng. Giao diện web chạy tại `http://localhost:8000`.
3. Vào **⚙ Cài Đặt**: nhập Telegram Bot Token, bấm dò Chat ID, gửi thử tin nhắn.
4. Bật **Tự động chạy cùng Windows** để hệ thống luôn bảo vệ 24/7.

**Chạy từ mã nguồn:** chạy `1_Cai_Dat_Tu_Dong.bat` rồi `2_Chay_He_Thong.bat`.

**Lưu ý:** đóng cửa sổ thì hệ thống **vẫn chạy ngầm**. Muốn tắt hẳn, vào Cài đặt và bấm nút đỏ **TẮT HỆ THỐNG**. Hướng dẫn chi tiết (kết nối camera IP, Telegram, xử lý sự cố): [huong-dan.html](https://quoccuong66.github.io/Project-monitoring-model/huong-dan.html).
