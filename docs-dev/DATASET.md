# ĐẶC TẢ DỮ LIỆU & CHIẾN LƯỢC TẬP DỮ LIỆU (DATASET SPECIFICATION & HARD NEGATIVES)

Tài liệu này xác định quy chuẩn cấu trúc dữ liệu, nhãn phân loại, danh mục các bộ dữ liệu công khai (public benchmark), bộ dữ liệu phản biện khó (Hard Negatives), và chiến lược phân chia tập dữ liệu chống rò rỉ (leak-free split).

---

## 1. PHÂN LOẠI HÀNH VI & MỨC ĐỘ NGUY HIỂM (TAXONOMY & SEVERITY)

Hệ thống phân định 10 hành vi cơ bản kết hợp cùng mức độ nguy cơ:

| Nhóm | Tên hành vi (`action`) | Mô tả chi tiết | Mức độ nguy cơ (`severity`) | Thời gian xác thực tối thiểu |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | `walking` | Đi lại bình thường, tốc độ ổn định | `NONE` (0.0) | N/A |
| **Normal** | `standing` | Đứng thẳng, duy trì vị trí | `NONE` (0.0) | N/A |
| **Normal** | `sitting` | Ngồi trên ghế, giường hoặc sofa | `NONE` (0.0) | N/A |
| **Normal** | `bending` | Cúi người nhặt đồ, buộc dây giày rồi đứng lên | `LOW` (0.15) | $< 3.0\text{ s}$ |
| **Normal** | `lying` | Nằm có chủ ý trên giường/sofa | `LOW` (0.20) | N/A |
| **Normal** | `getting_up` | Đối tượng tự đứng dậy từ sàn/ghế | `NONE` (Giảm risk) | N/A |
| **Abnormal** | `stumbling` | Mất thăng bằng, bước loạng choạng | `MEDIUM` (0.65) | $\ge 1.0\text{ s}$ |
| **Abnormal** | `abnormal_movement`| Chuyển động run giật cơ, vùng vẫy bất thường | `HIGH` (0.80) | $\ge 2.0\text{ s}$ |
| **Emergency**| `falling` | Đột ngột thay đổi độ cao trọng tâm, va chạm sàn | `CRITICAL` (0.95) | $\ge 0.5\text{ s}$ |
| **Emergency**| `immobile` | Nằm bất động trên sàn sau khi ngã, không phản ứng | `CRITICAL` (0.98) | $\ge 5.0\text{ s}$ |

---

## 2. QUY CHUẨN ĐẶC TẢ BẢNG GHI (METADATA SCHEMA)

Mỗi đoạn video/sequence được gán nhãn theo định dạng JSON chuẩn `manifest.json`:

```json
{
  "video_id": "cam01_20260903_fall_test_04",
  "source_dataset": "UP-Fall",
  "fps": 30.0,
  "resolution": [1920, 1080],
  "camera_angle": "oblique_high",
  "environment": "living_room",
  "lighting_condition": "ambient_daylight",
  "subjects": [
    {
      "subject_id": "subj_12",
      "age_group": "elderly",
      "mobility_aid": "none"
    }
  ],
  "events": [
    {
      "event_id": "evt_01",
      "person_id": 1,
      "start_time": 4.20,
      "end_time": 5.80,
      "action": "stumbling",
      "severity": "medium",
      "keypoint_occlusion_rate": 0.05
    },
    {
      "event_id": "evt_02",
      "person_id": 1,
      "start_time": 5.80,
      "end_time": 7.10,
      "action": "falling",
      "severity": "critical",
      "fall_direction": "forward"
    },
    {
      "event_id": "evt_03",
      "person_id": 1,
      "start_time": 7.10,
      "end_time": 18.00,
      "action": "immobile",
      "severity": "critical",
      "location": "floor"
    }
  ]
}
```

---

## 3. CÁC BỘ DỮ LIỆU CHUẨN CÔNG KHAI (BENCHMARK DATASETS)

Dự án tích hợp các bộ dữ liệu tiêu chuẩn quốc tế về phát hiện ngã và hành vi con người:

1. **UR Fall Detection (URFD)**:
   * 70 video (30 video ngã, 40 video hoạt động thường ngày ADL).
   * Cung cấp 2 góc nhìn camera (Frontal + Ceiling).
2. **Multiple Cameras Fall Dataset (Multicam)**:
   * 24 kịch bản thực tế từ 8 camera đồng bộ góc nhìn.
   * Rất tốt cho việc kiểm tra tính bất biến theo góc camera.
3. **UP-Fall Detection Dataset**:
   * 17 đối tượng thực hiện 11 hoạt động khác nhau (ngã từ tư thế đứng, ngã từ ghế, nhặt đồ, nằm, đi bộ).
4. **Le2i Fall Detection Dataset**:
   * Gồm các môi trường thực tế: Phòng khách, Phòng ngủ, Văn phòng, Hành lang.
   * Đi kèm hiện tượng che khuất một phần (ghế, bàn).
5. **NTU RGB+D 60 / 120 (Action Classes subset)**:
   * Tập trung vào các lớp: *A43 (falling down)*, *A8 (sitting down)*, *A9 (standing up)*, *A41 (headache)*, *A42 (chest pain)*, *A44 (nausea/vomit)*.

---

## 4. BỘ DỮ LIỆU PHẢN BIỆN KHÓ (HARD NEGATIVES CATALOG)

Để **triệt tiêu báo động giả (False Positives)**, hệ thống bắt buộc phải được kiểm thử và huấn luyện với tập Hard Negatives chuyên biệt:

| Tình huống Hard Negative | Đặc điểm động học dễ gây nhầm lẫn | Cơ chế loại trừ & xác thực |
| :--- | :--- | :--- |
| **Cúi nhặt đồ trên sàn** | Trọng tâm hạ thấp nhanh, lưng cong, đầu gần sàn. | Tỷ lệ khung bao $W/H$ không biến dạng thành nằm ngang; thời gian hạ thấp ngắn ($< 2.5\text{ s}$); đối tượng tự đứng lên ngay. |
| **Tập thể dục, Yoga, gập bụng** | Nằm trên thảm sàn, tư thế chân tay co duỗi trên mặt phẳng nằm ngang. | Chuyển động liên tục tuần hoàn (`Immobility Index` thấp); không có pha rơi gia tốc đột ngột trước đó. |
| **Nằm ngủ trên sofa / giường** | Nằm bất động trong thời gian dài. | Vị trí nằm ở độ cao cách mặt sàn $\ge 40\text{ cm}$ (không phải sàn nhà); tư thế chuyển từ ngồi sang nằm êm dịu, không có va chạm. |
| **Ngồi xuống ghế nhanh / ngả lưng** | Tốc độ hạ hông $V_y$ khá lớn trong $0.5\text{ s}$. | Dừng lại ở vị trí độ cao ghế; thân trên giữ phương thẳng đứng hoặc nghiêng nhẹ; không tiếp xúc sàn. |
| **Ngã có chủ ý khi chơi với trẻ em/thú cưng** | Ngã xuống sàn nhưng có tương tác liên tục. | Sau khi xuống sàn, các khớp tay và đầu vẫn chuyển động tích cực $\rightarrow$ Bị loại khỏi điều kiện `immobile`. |
| **Rung lắc camera hoặc thay đổi ánh sáng** | Bounding box jitter, bóng đổ trên sàn làm sai lệch detector. | Khớp xương được chuẩn hóa theo tỷ lệ thân người; kiểm tra tính liên tục của trọng tâm $y_{hip}$. |

---

## 5. ĐỊNH DẠNG LƯU TRỮ SEQUENCE (KEYPOINT PIPELINE FORMAT)

Dữ liệu chuỗi xương sau khi trích xuất qua `YOLOv8-Pose` được lưu trữ dưới dạng mảng nhị phân nén **`.npz`** hoặc tệp hàng đợi `Parquet` cho tốc độ nạp dữ liệu (data loading) tức thì:

* **Ma trận chuỗi xương (Pose Array)**:
  $$\mathbf{X} \in \mathbb{R}^{T \times K \times C}$$
  * $T = 30$ (Số khung hình thời gian, tương đương 2 giây tại 15 FPS).
  * $K = 17$ (17 khớp xương COCO).
  * $C = 3$ (Tọa độ chuẩn hóa $x_{norm}, y_{norm}$ và điểm tin cậy $c$).
* **Vector đặc trưng động học (Kinematics Array)**:
  $$\mathbf{F}_{kin} \in \mathbb{R}^{T \times 8}$$
  Gồm: $[V_y, A_y, \text{TorsoAngle}, \text{AspectRatio}, \text{GroundDistance}, \text{MotionEnergy}, \text{LeftRightBalance}, \text{OcclusionRatio}]$.

---

## 6. CHIẾN LƯỢC PHÂN CHIA KHÔNG RÒ RỈ (LEAK-FREE SPLIT STRATEGY)

> [!CAUTION]
> **Quy tắc bất biến**: Tuyệt đối **KHÔNG** chia train/val/test ngẫu nhiên theo từng frame hoặc clip cắt vụn của cùng một video. Điều này gây rò rỉ dữ liệu (Data Leakage) nghiêm trọng do ngoại hình đối tượng và góc phòng giống nhau.

* **Phương pháp**: Sử dụng **GroupKFold (Stratified by Subject & Scene)**:
  * **Train Set (70%)**: Toàn bộ dữ liệu của đối tượng $A, B, C, \dots$ trong phòng 1, 2.
  * **Validation Set (15%)**: Đối tượng hoàn toàn mới $D, E$ trong môi trường phòng 3 để tune hyperparameter và threshold.
  * **Test Set (15%)**: Đối tượng chưa từng thấy $F, G$ trong môi trường mới toanh (Unseen subjects & Unseen environments) để đo lường năng lực tổng quát hóa thực tế.
