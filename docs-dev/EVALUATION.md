# BỘ CHỈ SỐ ĐÁNH GIÁ & BENCHMARK THỜI GIAN THỰC (EVALUATION METRICS & REAL-TIME PROFILING)

Hệ thống giám sát an toàn cho người cao tuổi đòi hỏi tiêu chuẩn đánh giá khắt khe: **Ưu tiên hàng đầu là không bỏ sót sự cố nguy hiểm (High Recall)** nhưng đồng thời phải kiểm soát nghiêm ngặt hiện tượng báo động giả (Low False Alarm Rate) để tránh gây hoảng loạn hoặc nhờn chuông báo động (alarm fatigue).

---

## 1. HỆ THỐNG CHỈ SỐ KỸ THUẬT (MODEL EVALUATION METRICS)

### 1.1. Ma trận nhầm lẫn & Các chỉ số phân loại cơ bản
Cho mỗi lớp hành vi $c \in \{1, \dots, 10\}$, xác định các đại lượng:
* **True Positive ($TP$)**: Phát hiện chính xác hành vi (ví dụ: ngã thật và mô hình báo ngã).
* **False Positive ($FP$)**: Báo động giả (ví dụ: cúi nhặt đồ nhưng báo là ngã).
* **False Negative ($FN$)**: Bỏ sót sự cố (ví dụ: ngã nhưng mô hình báo là nằm bình thường).
* **True Negative ($TN$)**: Phân loại chính xác các hành vi bình thường khác.

Từ đó tính toán:
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall (Sensitivity)} = \frac{TP}{TP + FN}$$
$$\text{Specificity} = \frac{TN}{TN + FP}, \quad F_1\text{-score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
$$\text{False Positive Rate (FPR)} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$$
$$\text{False Negative Rate (FNR)} = \frac{FN}{TP + FN} = 1 - \text{Recall}$$

### 1.2. Đường cong ROC-AUC & PR-AUC
* **PR-AUC (Precision-Recall Area Under Curve)**: Là chỉ số quan trọng bậc nhất do dữ liệu mất cân bằng nghiêm trọng giữa các hành vi bình thường và khẩn cấp.
* **Mục tiêu phân lớp cho sự cố ngã & bất động (`falling` / `immobile`)**:
  * **Recall $\ge 95.0\%$** trên tập kiểm thử độc lập (Unseen test set).
  * **Precision $\ge 90.0\%$**.
  * **Tỷ lệ báo động giả (False Alarm Rate)**: $\le 1$ lần báo nhầm / $24$ giờ giám sát liên tục.

---

## 2. CHỈ SỐ VẬN HÀNH THỰC TẾ & THỜI GIAN TRỄ (OPERATIONAL LATENCY METRICS)

Hệ thống đo lường độ trễ chi tiết qua từng mắt xích bằng module đo lường chuẩn xác `LatencyProfiler`:

| Mắt xích Pipeline | Metric đo lường | Ngưỡng mục tiêu (Target) | Đo thực tế trên RTX 3050 + i5-12450HX |
| :--- | :--- | :--- | :--- |
| **Camera Input Stream** | Camera FPS / Frame Drop Rate | $\ge 25-30\text{ FPS}$, Drop $<0.5\%$ | 30.0 FPS, Drop 0.0% |
| **Person Detection** | Inference Latency (ms) | $\le 10.0\text{ ms}$ | ~6.2 ms (YOLOv8s FP16) |
| **Multi-Object Tracking** | Association Latency (ms) | $\le 2.0\text{ ms}$ | ~1.2 ms (ByteTrack) |
| **Pose Estimation** | Keypoint Latency (ms) | $\le 12.0\text{ ms}$ | ~7.8 ms (YOLOv8s-Pose FP16) |
| **Temporal Sequence Prep** | Normalization & Ring Buffer (ms)| $\le 1.5\text{ ms}$ | ~0.8 ms |
| **Temporal Action Model** | Forward Pass (ms) | $\le 5.0\text{ ms}$ | ~3.5 ms (ST-Transformer) |
| **Anomaly Detector** | Reconstruction Forward (ms) | $\le 3.0\text{ ms}$ | ~1.8 ms (Pose Autoencoder) |
| **Risk Engine & State Machine**| State Transition Evaluation (ms)| $\le 1.0\text{ ms}$ | ~0.4 ms |
| **End-to-End Frame Latency** | Tổng trễ từ lúc nhận frame đến xuất kết quả | $\mathbf{\le 35.0\text{ ms}}$ | $\mathbf{\approx 21.7\text{ ms} \longrightarrow \approx 46\text{ FPS}}$ |
| **Event-to-Alert Time** | Thời gian từ lúc người ngã đến lúc phát chuông cảnh báo | $\mathbf{\le 2.0 - 3.0\text{ s}}$ (Đã bao gồm thời gian debounce 2s để xác thực người không tự đứng lên) | $\mathbf{\approx 2.5\text{ s}}$ |

---

## 3. TIÊU THỤ TÀI NGUYÊN HỆ THỐNG (RESOURCE MONITORING)

Các chỉ số tài nguyên được theo dõi liên tục trong suốt quá trình chạy:

* **GPU Memory (VRAM)**:
  * Ngưỡng báo động: $>5.2\text{ GB}$ (trên tổng 6GB của RTX 3050).
  * Mục tiêu vận hành: $\mathbf{3.0 - 3.8\text{ GB}}$ VRAM.
* **GPU Compute Utilization**:
  * Duy trì mức $60\% - 85\%$ khi xử lý stream 30 FPS.
* **CPU Load (Intel i5-12450HX)**:
  * Tận dụng đa luồng (Multi-threading): Tách riêng I/O thread (RTSP decode), AI Worker thread, và API server thread.
  * Mức tiêu thụ CPU mục tiêu: $<35\%$ tổng 12 logical processors.
* **RAM hệ thống (16GB)**:
  * Mức tiêu thụ: $<3.5\text{ GB}$ RAM.

---

## 4. QUY TRÌNH CHẠY KIỂM THỬ ĐÁNH GIÁ (EVALUATION SCRIPTS)

### 4.1. Đánh giá độ chính xác mô hình học sâu (`training/evaluate.py`)
```bash
python training/evaluate.py \
  --config configs/training.yaml \
  --action-weights models/checkpoints/best_st_transformer.pt \
  --anomaly-weights models/checkpoints/best_pose_autoencoder.pt \
  --test-split data/splits/test.jsonl \
  --output-dir results/evaluation_reports/
```
* **Kết quả đầu ra**:
  * `confusion_matrix.png`: Biểu đồ ma trận nhầm lẫn 10 lớp hành vi.
  * `pr_curves.png`: Đường cong Precision-Recall cho từng lớp khẩn cấp.
  * `metrics_summary.json`: Bảng tổng hợp Precision, Recall, F1, FPR, FNR.

### 4.2. Benchmark tốc độ & tài nguyên phần cứng (`scripts/benchmark.py`)
```bash
python scripts/benchmark.py \
  --video data/videos/test_fall_sample.mp4 \
  --config configs/inference.yaml \
  --runs 500 \
  --warmup 50
```
* Báo cáo chi tiết bảng FPS, Latency từng tầng, VRAM min/max/average, CPU utilization.
