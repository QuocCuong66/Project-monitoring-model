# Model RGB mở rộng đã tích hợp — 2026-11-03

Đã tải thêm 2.526.022.103 byte (~2,526 GB): NTU120 2D, UCF101 2D, HMDB51 2D và
5 ZIP UP-Fall. Huấn luyện thêm model tương thích camera RGB, sau đó tích hợp vào
`models/checkpoints/best_st_transformer.onnx` và bật cấu hình trong `configs/model.yaml`.
Không dùng model NTU 3D 25 khớp làm đầu vào trực tiếp cho camera COCO-17.

## Dữ liệu thực sự dùng

| Nguồn | Train | Validation | Test |
|---|---:|---:|---:|
| NTU60 cũ, giữ nhãn 10 lớp | 17.446 | 14.629 | — |
| NTU120 2D, hoạt động mới A061–A105 | 28.100 | 3.886 | 43.837 |
| UCF101, nhóm hoạt động dễ nhầm | 1.875 | 354 | 905 |
| HMDB51, nhóm hoạt động dễ nhầm | 895 | 233 | 490 |
| URFD | 433 | 175 | 153 |
| UP-Fall, cửa sổ chạm đất đầu tiên | 61 | 5 | 16 |
| **Tổng cửa sổ** | **48.810** | **19.282** | **45.401** |

Tổng 113.493 cửa sổ chuẩn hóa, không phải 113.493 video độc lập. Các cửa sổ cùng
clip có tương quan. Lượt train lấy mẫu có trọng số 24.000 lần/epoch, chạy 12 epoch
với early stopping, lưu checkpoint epoch 6. Số lần lấy mẫu lặp không tính là dữ liệu mới.

- NTU120 chỉ dùng các lớp mới, không đếm lại NTU60. Tách theo người; test bổ sung
  chỉ lấy người >40 để tránh dùng lại những người trong benchmark NTU60 cũ.
- UCF101 tách theo nhóm video gXX, HMDB51 theo video trong split chính thức.
  Không tuyên bố độc lập theo người cho hai nguồn không có ID người đầy đủ này.
- Hoạt động bình thường có nhãn thô được tối ưu bằng tổng xác suất của các lớp an
  toàn được phép, không tự gán mọi clip thành đứng hoặc nằm.
- UP-Fall tải thực tế có 82 file CSV/góc quay của 5 người, chỉ hoạt động A1–A5.
  Không có ADL hay giấc ngủ trong các ZIP này. Giữ người 1/2/3 train, 4 val, 5 test;
  hai góc quay cùng người không được trộn split. Chỉ dùng đoạn tới lần impact đầu
  tiên, không gán mọi frame nằm sau ngã thành `falling`.
- Adapter UP-Fall dùng x/y ảnh chuẩn hóa, bỏ z, ánh xạ MediaPipe 33 sang COCO-17,
  theo crop 500×700 của tác giả. CSV không lưu độ tin cậy; kênh thứ ba là mặt nạ
  hiện diện hữu hạn, không phải confidence đo được. Không có timestamp để chứng
  minh chính xác tốc độ lấy mẫu hay độ trễ cảnh báo từ nguồn này.
- UCF dùng 25 FPS, NTU 30 FPS; HMDB dùng giả định danh nghĩa 30 FPS do pickle không
  có FPS từng clip. Chuẩn hóa timeline sang 15 Hz trước khi tạo cửa sổ 30 bước.
- Preprocessing dùng chung `build_model_sequence` với runtime. Không diễn giải
  tọa độ z của skeleton 3D thành confidence.

## Cách tích hợp cuối cùng

Model ONNX xuất xác suất kết hợp 50% model gốc và 50% student đã train, vẫn có
đầu vào `(batch,30,17,3)` và 10 lớp cũ. Giữ tỷ lệ phân biệt nằm/bất động từ model
gốc trong nhóm tư thế này, tránh model mới đổi nhãn nằm sang bất động. Các nhóm
hành động khác vẫn hưởng dữ liệu mới. Mô hình này chứa cả hai mạng nhỏ trong một
ONNX; runtime không cần tải file PyTorch teacher riêng.

Khi model chắc chắn thuộc nhóm bình thường và tín hiệu rơi yếu, classifier giữ
bằng chứng học được thay vì luôn ghi đè bằng prior bất động từ tư thế ngang.
Các ngưỡng ở `temporal.normal_activity_policy`. Cú rơi mạnh không đi qua nhánh này.

Risk engine giữ lịch sử ngã: đã quan sát descent đột ngột, người nằm xuống và được
ngữ cảnh xác định `collapsed`, bất động đủ mức, ngoài vùng nghỉ, thì nhãn model
`lying` không xóa bằng chứng ngã. Mức điểm vẫn đi qua state machine và thời gian
xác nhận hiện có. Người đang dùng điện thoại/chuyển tư thế chưa được coi là collapsed.
Tham số bổ sung ở `lying_context.confirmed_fall_immobility_min`.

## Kết quả

Trên validation NTU60 cũ (đã dùng trong phát triển trước đây):

| Chỉ số | Gốc | Bản đã tích hợp |
|---|---:|---:|
| Accuracy 10 lớp | 83,30% | 87,18% |
| Macro-F1 | 80,09% | 83,57% |
| Normal → emergency FPR | 11,70% | 7,50% |
| Recall ngã | 98,72% | 98,14% |
| Precision ngã | 76,42% | 86,83% |
| Recall nằm | 61,24% | 62,27% |
| Recall bất động | 55,22% | 55,82% |

Trên split test hoạt động mới, FPR phân loại cửa sổ:

| Nguồn | Gốc | Bản đã tích hợp |
|---|---:|---:|
| NTU120 | 34,60% | 13,63% |
| UCF101 | 16,13% | 8,84% |
| HMDB51 | 15,92% | 10,41% |
| Trung bình ba nguồn, trọng số bằng nhau | 22,22% | 10,96% |

UP-Fall người giữ riêng: nhận đúng lớp ngã ở cửa sổ impact 6/16 → 11/16. Đây là
các góc quay của một người, không phải 16 sự kiện/người độc lập. URFD test transition
window F1 37,68% → 37,29%, nên không khẳng định cải thiện ở mọi bộ dữ liệu.

Không train trên test. Checkpoint/ngưỡng hiệu chỉnh chọn bằng validation. Các tập
test đã được xem trong phiên phát triển này; URFD còn đã dùng ở lượt trước. Đây
là kết quả phát triển trên các split tách train, không phải xác nhận tiền cứu độc lập.
FPR trên cửa sổ không phải số chuông báo sai/giờ. Chưa có dữ liệu ngủ qua đêm để
khẳng định hết báo sai khi ngủ, cũng không có bằng chứng hiệu quả chẩn đoán bệnh.

## Kiểm tra runtime và quay lại

ONNX khớp PyTorch trên pose thật (sai lệch logits tối đa ~1,43e-6), nạp và chạy
trực tiếp qua `DmlExecutionProvider` thành công. Đo riêng model batch 1, 50 lần:
median ~9,08 ms, p95 ~12,45 ms; không phải độ trễ toàn pipeline camera.
Model active nặng 4.719.444 byte (~4,72 MB).
Toàn bộ bộ kiểm thử dự án sau khi tích hợp: **228 passed** (5 cảnh báo deprecation
của thư viện hiện có), bao gồm ca nằm bình thường, tín hiệu rơi mạnh và ngã thật
rồi model gọi tư thế sau đó là nằm.

Bản gốc đã sao lưu tại:
`models/checkpoints/backups/rgb_20261103T132429Z/`
(gồm ONNX và `model.yaml`). Đường dẫn và SHA-256 chính xác trong
`models/checkpoints/rgb_expanded.deployment.json`.
`best_st_transformer.pt` được giữ nguyên làm teacher gốc cho tái kiểm tra;
trọng số runtime thực tế là ONNX đã tích hợp. Cần khởi động lại app đang mở để nạp lại.

## Chạy lại và nguồn

```powershell
.venv/Scripts/python.exe training/download_rgb_expansion.py
.venv/Scripts/python.exe training/prepare_rgb_expansion.py
.venv/Scripts/python.exe training/train_rgb_expansion.py
# Hiệu chỉnh checkpoint đã có trên validation theo lớp nằm/bất động:
.venv/Scripts/python.exe training/reselect_rgb_expansion.py
.venv/Scripts/python.exe training/promote_rgb_expansion.py
```

Các ZIP UP-Fall nằm trong `data/raw/upfall`; URL tác giả và SHA-256 lưu trong
`data/processed/rgb_expansion/upfall.json`. Các download MMAction có `.download.json`.
Config, lịch sử train, kết quả trước/sau, và deployment lưu ở `configs/rgb_expansion.yaml`
và `models/checkpoints/rgb_expanded.*`. Script promotion từ chối nếu chưa qua gate,
kiểm tra input/output, parity CPU/GPU; tự khôi phục file cũ nếu kích hoạt lỗi.

Nguồn cần trích dẫn và tuân thủ điều kiện sử dụng:

- [MMAction2 skeleton annotations](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md).
- [NTU RGB+D / NTU120](https://rose1.ntu.edu.sg/dataset/actionRecognition/).
- [UCF101 và phân nhóm video](https://www.crcv.ucf.edu/data/UCF101.php).
- [UP-Fall cải tiến và các file thực tế](https://zenodo.org/records/12773013),
  [bài mô tả crop và gán nhãn impact](https://arxiv.org/abs/2502.19048).
- [URFD](https://fenix.ur.edu.pl/~mkepski/ds/uf.html).

SAFER-Activities chưa tải vì kho yêu cầu quyền tài khoản và chấp thuận điều kiện;
không đưa vào số mẫu hoặc kết quả ở trên.
