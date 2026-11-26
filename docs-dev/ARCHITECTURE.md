# KIẾN TRÚC HỆ THỐNG — AI CAMERA MONITORING & ABNORMAL BEHAVIOR DETECTION

> **Mục tiêu**: Giám sát video stream liên tục, hiểu chuỗi hành vi theo thời gian (temporal sequence), phát hiện sớm các nguy cơ y tế khẩn cấp và hành vi bất thường ở người lớn tuổi mà **không tự ý kết luận chẩn đoán y tế**. Cảnh báo hướng tới con người: *"Possible medical emergency / abnormal behavior detected"*.

---

## 1. TỔNG QUAN HỆ THỐNG (SYSTEM OVERVIEW)

Hệ thống được thiết kế theo kiến trúc **Multi-Stage Temporal Video Intelligence Pipeline**, phân tách rành mạch giữa các tầng: **Perception (Nhận thức)** $\rightarrow$ **Representation (Biểu diễn chuỗi xương & chuyển động)** $\rightarrow$ **Temporal & Anomaly Modeling (Mô hình hóa thời gian & bất thường)** $\rightarrow$ **Multi-Signal Fusion & Risk Engine (Lập luận rủi ro & máy trạng thái)** $\rightarrow$ **Human Alert Dispatch (Điều phối cảnh báo)**.

```mermaid
flowchart TD
    subgraph S1["1. Video Ingestion Layer"]
        C1["RTSP Camera / IP Cam"] --> CS["CameraStream Interface"]
        C2["Webcam / USB Cam"] --> CS
        C3["MP4 / Video File"] --> CS
        CS --> PP["Frame Buffer & Preprocessor (FPS Sync, Resize)"]
    end

    subgraph S2["2. Perception & Tracking Layer"]
        PP --> DET["Person Detector (YOLOv8/v11)"]
        DET --> TRK["Multi-Object Tracker (ByteTrack / BoT-SORT)"]
        TRK --> |Track ID + Bounding Box History| TRACK_MGR["Track Manager"]
    end

    subgraph S3["3. Pose & Spatial-Temporal Feature Extraction"]
        TRACK_MGR --> POSE["Human Pose Estimator (YOLO-Pose / RTMPose)"]
        POSE --> KEYPOINTS["17 Keypoints Normalization (x, y, conf)"]
        KEYPOINTS --> BUF["Sliding Temporal Sequence Buffer (T=30-120 frames)"]
        BUF --> FEAT["Temporal & Kinematic Feature Extractor<br/>(Velocity, Acceleration, Aspect Ratio, Ground Distance)"]
    end

    subgraph S4["4. Dual-Stream AI Inference Engine"]
        FEAT --> ACT_MODEL["Supervised Temporal Action Recognizer<br/>(Spatial-Temporal Transformer / ST-GCN / TCN)"]
        FEAT --> ANOM_MODEL["Unsupervised Behavior Anomaly Detector<br/>(Pose Sequence Autoencoder / VAE)"]
        PP -.-> |Optional Crop Sampling| RGB_FEAT["RGB Motion / Kinematics"]
    end

    subgraph S5["5. Multi-Signal Fusion & Clinical Safety Risk Engine"]
        ACT_MODEL --> |P(Action)| FUSION["Multi-Signal Fusion Engine"]
        ANOM_MODEL --> |Anomaly Score S_anom| FUSION
        RGB_FEAT -.-> FUSION
        FEAT --> |Kinematic Signals| FUSION
        FUSION --> RISK_ENG["Contextual Risk Engine & Severity Scorer"]
        RISK_ENG --> FSM["Debounced State Machine (Hysteresis Filter)"]
    end

    subgraph S6["6. Alert, Logging & Human Verification Layer"]
        FSM -->|State: HIGH_RISK| ALERT_MGR["Alert Dispatcher & Cooldown Manager"]
        ALERT_MGR --> WS["WebSocket Dashboard Live Feed"]
        ALERT_MGR --> API["REST API / Webhooks"]
        ALERT_MGR --> NOTIF["Notification Services (Telegram / Discord / SMS)"]
        ALERT_MGR --> EVT_LOG["Encrypted Local Event Logger (Privacy Compliant)"]
    end
```

---

## 2. CHI TIẾT TỪNG MODULE (MODULE BREAKDOWN)

### 2.1. Ingestion Layer (`src/camera/`)
* **Interface `CameraStream`**: Lớp trừu tượng định nghĩa các phương thức đồng bộ `read()`, `is_opened()`, `release()`, và `get_metadata()`.
* **Cơ chế Threading & Queue Buffer**: Tách rời luồng đọc frame (I/O Bound) và luồng xử lý AI (Compute Bound). Sử dụng `Double-Buffering` hoặc `Drop-Oldest Queue` để đảm bảo hệ thống không bị tích lũy độ trễ (latency drift) khi camera phát 30 FPS còn pipeline chạy 15-20 FPS.
* **Hỗ trợ đa nguồn**:
  * `WebcamStream`: Hỗ trợ chuẩn V4L2 (Linux) và DirectShow/MSMF (Windows).
  * `RTSPStream`: Hỗ trợ kết nối RTSP từ camera an ninh Hikvision, Dahua, Imou, TP-Link Tapo kèm cơ chế tự động kết nối lại (Auto-reconnect with exponential backoff).
  * `VideoStream`: Xử lý video file MP4, AVI phục vụ offline benchmark và evaluation.

### 2.1.1. Tier-0 Motion Gating (`src/detection/motion_gater.py`)
* **Mục tiêu**: Tiết kiệm tối đa năng lượng và nhiệt độ GPU khi căn phòng tĩnh lặng (không có người hoặc người rời khỏi phòng).
* **Thuật toán trừ nền**:
  * Hỗ trợ kép: **MOG2** (`cv2.createBackgroundSubtractorMOG2`) và **Frame Differencing** siêu nhẹ.
  * Xử lý trên frame thu nhỏ (320x240), thời gian thực thi cực thấp (< 0.3 ms trên CPU).
* **Cơ chế Zero GPU Inference**:
  * Khi phòng tĩnh lặng (`motion_ratio < min_motion_ratio`): **Tắt 100% suy luận AI** (RTMO-s, YOLOv8, ST-GCN, Autoencoder đều bị bypass), GPU tiêu thụ = 0%.
  * Giảm FPS xử lý xuống mức tiết kiệm **10–15 FPS** (`idle_stride = 2`).
* **Cơ chế đánh thức tức thì & Chống mù (Anti-Blindness Safeguards)**:
  * **Instant Wake-up**: Khi phát hiện chuyển động người bước vào, lập tức bật lại AI ở tốc độ tối đa **30 FPS**.
  * **Cooldown Debouncing (45 frames ~ 3.0s)**: Duy trì mở cổng suy luận khi người tạm đứng yên hoặc di chuyển chậm.
  * **Active Track Override**: Nếu camera vẫn đang theo dõi đối tượng trong phòng, cổng luôn giữ mở hoặc giám sát liên tục.
  * **Periodic Keep-alive**: Tự động đánh thức suy luận 1 frame sau mỗi 45 frames tĩnh lặng để rà soát phòng ngừa người ngất lịm bất động.

### 2.2. Detection & Multi-Object Tracking (`src/detection/`, `src/tracking/`)

* **`PersonDetector` Interface**:
  * Model cốt lõi: YOLOv8n/s hoặc YOLOv11n (được lọc chỉ lấy class `0: person`).
  * Trả về danh sách đối tượng `Detection(bbox=[x1, y1, x2, y2], confidence, class_id=0)`.
* **`TrackManager` & Multi-Object Tracking**:
  * Thuật toán: **ByteTrack** (sử dụng Kalman Filter + Hungarian matching trên cả high-score và low-score detections để giữ track khi người bị che khuất một phần).
  * Cấu trúc `PersonTrack`:
    ```python
    @dataclass
    class PersonTrack:
        track_id: int
        bbox: tuple[float, float, float, float]
        bbox_history: deque[tuple[float, float, float, float]]
        trajectory: deque[tuple[float, float]]  # Center bottom coordinates
        last_seen_timestamp: float
        velocity_history: deque[float]
        standing_height_baseline: float  # Chiều cao tham chiếu khi đứng bình thường
        ground_plane_y: float            # Ước lượng mặt sàn cục bộ
    ```

### 2.3. One-Stage Perception & Pose Estimation (`src/pose/`)
* **`PoseEstimator` Interface (RTMO-s)**:
  * Model: **RTMO-s** (One-stage Multi-Person Pose Estimation compiled to TensorRT FP16).
  * Gộp đồng thời Detection + 17 điểm COCO keypoints:
    * Mũi, 2 mắt, 2 tai (Head/Face)
    * 2 vai, 2 khuỷu tay, 2 cổ tay (Upper Body)
    * 2 hông, 2 đầu gối, 2 mắt cá chân (Lower Body)
  * Mỗi keypoint gồm $(x, y, c)$ với $c \in [0, 1]$ là confidence score.
* **Chuẩn hóa Keypoints (Pose Normalization)**:
  * Trọng tâm chuẩn hóa: Dời gốc tọa độ về trung điểm của 2 hông (Mid-Hip root centered).
  * Co giãn chuẩn hóa: Scale theo kích thước chiều dài thân người (Torso length: khoảng cách giữa Mid-Shoulder và Mid-Hip).
  * Giúp mô hình bất biến với khoảng cách camera (xa/gần) và vị trí trong khung hình.

### 2.4. Temporal Representation & Sequence Buffer (`src/temporal/`)
* **`SequenceBuffer`**:
  * Mỗi `track_id` có một bộ đệm vòng tròn (Ring Buffer / `deque`) lưu trữ $T$ frames liên tiếp (mặc định $T=30$ frames tại 15 FPS $\approx 2$ giây cho short-term, và $T=90$ frames $\approx 6$ giây cho long-term).
  * Tự động interpolate (nội suy tuyến tính) nếu mất keypoint 1-2 frames do che khuất.
* **Feature Engineering Kép (Kinematic Features)**:
  * **Trọng tâm (Center of Mass - CoM)**: Tốc độ rơi thẳng đứng $V_y = \frac{d(y_{hip})}{dt}$, gia tốc rơi $A_y = \frac{d^2(y_{hip})}{dt^2}$.
  * **Tỷ lệ khung bao (Aspect Ratio)**: Tỷ lệ $\frac{W}{H}$ thay đổi đột ngột từ $<0.5$ (đang đứng) lên $>1.5$ (nằm trên sàn).
  * **Góc nghiêng cơ thể (Torso Angle)**: Góc giữa trục thân (Mid-Shoulder đến Mid-Hip) so với trục thẳng đứng.
  * **Chỉ số bất động (Immobility Index)**: Độ biến thiên tổng thể các khớp trong cửa sổ trượt $3-10$ giây:
    $$\text{Motion}(t) = \frac{1}{K} \sum_{k=1}^K \|\mathbf{p}_k(t) - \mathbf{p}_k(t-1)\|$$

### 2.5. Dual AI Models: Supervised Behavior + Unsupervised Anomaly (`src/action/`, `src/anomaly/`)
* **Supervised Action Recognizer (`Spatial-Temporal Transformer / ST-GCN`)**:
  * Dự đoán xác suất của 10 lớp hành vi:
    1. `walking`
    2. `standing`
    3. `sitting`
    4. `lying`
    5. `bending` (cúi người nhặt đồ)
    6. `falling` (đang trong quá trình ngã)
    7. `getting_up` (tự đứng dậy)
    8. `stumbling` (loạng choạng, vấp)
    9. `abnormal_movement` (co giật, chuyển động bất thường)
    10. `immobile` (bất động trên sàn)
* **Unsupervised Anomaly Detector (`Pose Sequence Autoencoder / VAE`)**:
  * Huấn luyện chỉ trên dữ liệu hành vi bình thường (`normal: walking, sitting, standing, cooking, reading...`).
  * Khi gặp chuỗi động học bất thường (ngã, lăn, co giật), lỗi tái tạo (Reconstruction Error) sẽ tăng vọt:
    $$S_{anom} = \frac{1}{T \cdot K} \sum_{t=1}^T \sum_{k=1}^K \|\mathbf{p}_{t, k} - \hat{\mathbf{p}}_{t, k}\|^2$$

### 2.6. Multi-Signal Fusion Engine (`src/risk/`)
Hợp nhất các tín hiệu không đồng nhất để ra quyết định tin cậy, triệt tiêu false positive:
$$\text{RiskScore} = w_{act} \cdot P(\text{Emergency Action}) + w_{anom} \cdot S_{anom} + w_{kin} \cdot S_{kinematic} + w_{immob} \cdot S_{immobility}$$

Trong đó các trọng số $w$ được cấu hình qua `configs/thresholds.yaml` và được chuẩn hóa $\sum w = 1$.

### 2.7. Event State Machine (Hysteresis & Anti-Flapping Filter) (`src/risk/`)
Ngăn chặn hiện tượng nhấp nháy cảnh báo (alert bouncing) khi độ tin cậy dao động quanh ngưỡng:

```text
               ┌───────────────┐
               │    NORMAL     │
               └───────┬───────┘
                       │ Risk > T_suspicious (0.60) trong 0.5s
                       ▼
               ┌───────────────┐
         ┌────►│  SUSPICIOUS   │◄────┐
         │     └───────┬───────┘     │
         │             │ Risk > T_abnormal (0.75) trong 1.5s
         │             ▼             │
         │     ┌───────────────┐     │
         │     │   ABNORMAL    │     │
         │     └───────┬───────┘     │
         │             │ Risk > T_high (0.88) duy trì > 3.0s
         │             ▼             │
         │     ┌───────────────┐     │
         │     │   HIGH_RISK   │     │
         │     └───────┬───────┘     │
         │             │ Emit Event  │
         │             ▼             │
         │     ┌───────────────┐     │
         │     │  ALERT_SENT   │     │
         │     └───────┬───────┘     │
         │             │ Cooldown active (60s)
         │             ▼             │
         │     ┌───────────────┐     │
         └─────┤ WAITING_CONF  ├─────┘
               └───────────────┘
```

* **Logic giải tỏa (Recovery logic)**: Nếu đối tượng tự đứng dậy (`action == getting_up` hoặc `standing`) và chỉ số chuyển động trở lại bình thường trong $>3$ giây, trạng thái tự động hạ cấp về `NORMAL` mà không kích hoạt chuông cảnh báo khẩn cấp (ghi log `EVENT_RECOVERED_AUTONOMOUSLY`).

### 2.8. Alert & Notification System (`src/alerts/`)
* **Nguyên tắc An toàn Y tế (Non-diagnostic principle)**:
  * Tuyệt đối không xuất thông điệp: "Bệnh nhân bị đột quỵ" hay "Người già đã gãy xương".
  * Luôn phát cảnh báo dạng:
    > *"Cảnh báo: Phát hiện tình huống khẩn cấp / hành vi bất thường. Vui lòng kiểm tra đối tượng ID: {person_id}."*
* **Payload cấu trúc JSON chuẩn**:
  ```json
  {
    "event_id": "evt_20260903_140501_trk3",
    "event_type": "possible_medical_emergency",
    "person_id": 3,
    "timestamp": "2026-09-03T14:05:01.240Z",
    "risk_score": 0.942,
    "duration_seconds": 6.8,
    "evidence": {
      "primary_action": "falling",
      "action_confidence": 0.96,
      "anomaly_score": 0.89,
      "fall_velocity_mps": 2.8,
      "aspect_ratio_change": "0.42 -> 1.85",
      "post_fall_immobility_seconds": 4.5
    },
    "snapshot_uri": "/events/evt_20260903_140501_trk3/snapshot.jpg",
    "recommended_action": "Verify person status immediately."
  }
  ```

---

## 3. THIẾT KẾ BẢO MẬT & QUYỀN RIÊNG TƯ (PRIVACY BY DESIGN)

1. **Local Edge Computing**: Toàn bộ luồng video được giải mã, suy luận AI và phân tích cục bộ trên máy trạm/server nội bộ. Không truyền stream video raw lên public cloud.
2. **Face Anonymization (Làm mờ mặt)**: Khi lưu trữ frame bằng chứng (`snapshot`), hệ thống tự động làm mờ (Gaussian Blur / Pixelation) vùng khuôn mặt dựa trên tọa độ keypoints (Mũi, Mắt, Tai) nếu cờ cấu hình `privacy.blur_faces: true` được kích hoạt.
3. **Chính sách lưu trữ có thời hạn (Data Retention)**: Snapshot và metadata chỉ lưu trong $N$ ngày (`retention_days: 7`), sau đó tự động xoay vòng xóa (auto-purge).

---

## 4. TƯƠNG THÍCH PHẦN CỨNG & HẠ TẦNG (HARDWARE MAPPING)

* **Thiết bị mục tiêu kiểm tra**:
  * **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, CUDA 12.x).
  * **CPU**: 12th Gen Intel Core i5-12450HX (8 cores, 12 threads).
  * **RAM**: 16 GB DDR4/DDR5.
  * **OS**: Windows 11 64-bit, Python 3.11.9.
* **Chiến lược phân bổ VRAM (Ngân sách tối đa 6GB)**:
  * YOLOv8s Person Detector: ~0.8 GB VRAM.
  * YOLOv8s-Pose Estimator: ~1.1 GB VRAM.
  * ByteTrack (CPU-bound Kalman Filter): 0 GB VRAM.
  * Temporal ST-Transformer / TCN: ~0.5 GB VRAM.
  * Pose Autoencoder: ~0.3 GB VRAM.
  * CUDA Context & PyTorch Runtime: ~0.8 GB VRAM.
  * **Tổng sử dụng**: $\approx 3.5$ GB VRAM (Dư dả >2.5 GB headroom, hoàn toàn an toàn tránh Out-Of-Memory).
