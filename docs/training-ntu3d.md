# NTU60 / NTU120 skeleton 3D

Hai link ZIP text `.skeleton` trong kho tác giả tại
https://github.com/shahroudy/NTURGB-D báo Google Drive "Quota exceeded" khi kiểm tra.
Lượt tải này dùng bản skeleton 3D công bố bởi MMAction2:
https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md

- NTU60: `ntu60_3d.pkl`, 1.185.880.751 byte (~1,19 GB).
- NTU120: `ntu120_3d.pkl`, 2.079.477.249 byte (~2,08 GB).
- Định dạng là pickle chứa tọa độ khớp 3D; không phải ZIP text `.skeleton` gốc.
- NTU120 bao gồm NTU60. Không nối hai file thành một tập và đếm trùng mẫu.
- Tuân thủ điều kiện dữ liệu gốc: https://rose1.ntu.edu.sg/dataset/actionRecognition/

## Chạy

```powershell
.venv/Scripts/python.exe training/download_ntu3d.py
.venv/Scripts/python.exe training/train_ntu3d.py --dataset ntu60
.venv/Scripts/python.exe training/train_ntu3d.py --dataset ntu120
```

Các lượt train dùng GPU lần lượt sau URFD. Thông số ở `configs/ntu3d.yaml`.
Đây là lượt benchmark khởi đầu 10 epoch cho mỗi bộ, chưa phải kết quả hội tụ hay
bằng chứng đạt chất lượng để xuất bản. Lịch sử loss/validation lưu kèm kết quả.

## Đầu vào và cách chia

- Giữ 25 khớp × 3 tọa độ x/y/z = 75 đặc trưng, giữ nguyên 60 hoặc 120 nhãn gốc.
- Tọa độ z là độ sâu vật lý, không bao giờ dùng làm độ tin cậy như COCO-17.
- Chọn cơ thể có nhiều khớp hợp lệ nhất trong clip; tương tác nhiều người được
  đơn giản hóa còn một người, cần xem đây là giới hạn của baseline này.
- Lấy mẫu đều 30 bước trên toàn clip; chuẩn hóa theo gốc hông ở frame hợp lệ đầu
  tiên và chiều dài thân trung vị. Giữ dịch chuyển cả cơ thể theo thời gian.
- Test theo danh sách cross-subject chính thức trong file MMAction2. Chọn 20%
  người trong phần train chính thức làm validation; người giữa ba tập không trùng.
- Chỉ chọn checkpoint bằng validation. Chạy test sau cùng, không chọn model/ngưỡng
  từ test. Hai benchmark NTU60/120 riêng biệt, không giả định chúng độc lập dữ liệu.

## Artifact và phạm vi sử dụng

Raw và SHA-256: `data/raw/ntu3d/`.
Đặc trưng đã chuẩn hóa và chỉ số chia: `data/processed/ntu3d/{ntu60,ntu120}/`.
Checkpoint, ONNX và metrics: `models/checkpoints/ntu3d/{ntu60,ntu120}_3d.*`.

Các model này yêu cầu đầu vào 3D 25 khớp, **không tương thích trực tiếp** với
camera hiện tại dùng 2D COCO-17. Không thay file trọng số runtime bằng model 3D.
Muốn áp dụng cho camera RGB cần bước ánh xạ/ước lượng 3D hoặc distillation riêng
và kiểm tra lại toàn bộ pipeline. Nhãn hành động trong NTU không xác nhận bệnh lý
và không thay thế dữ liệu nằm/ngủ thực tế.

## Kết quả lượt chạy 2026-11-03

Mỗi bộ train 10 epoch từ đầu, checkpoint tốt nhất theo macro-F1 validation ở
epoch 9. Kết quả trên test cross-subject chính thức:

| Bộ | Train / val / test | Accuracy đủ lớp | Macro-F1 | Precision ngã A043 | Recall ngã A043 |
|---|---|---:|---:|---:|---:|
| NTU60 | 26.480 / 13.611 / 16.487 | 48,74% | 47,33% | 89,67% | 97,82% |
| NTU120 | 45.455 / 17.571 / 50.919 | 35,19% | 34,98% | 58,22% | 95,27% |

Lớp A043 có 275 mẫu test trong mỗi bộ; không cộng chúng thành 550 mẫu độc lập.
NTU120 bổ sung nhiều lớp hành động, không có nghĩa lượng sự kiện ngã tăng tương ứng.
Các model vẫn còn yếu khi phân loại đầy đủ 60/120 hành động; không diễn giải recall
riêng ngã là accuracy toàn hệ thống.

Phân tích lỗi hậu nghiệm: model NTU120 nhầm ngã nhiều nhất ở A092 di chuyển vật
nặng, A080 ngồi xổm, A114 mang đồ; cần đưa các hành động này vào phân tích tiếp theo.
Đây là quan sát từ test, không dùng để điều chỉnh lượt train vừa báo cáo.

ONNX đã đối chiếu với PyTorch trên mẫu 3D thật, sai lệch logits tối đa lần lượt
~9,54e-7 (NTU60) và ~1,43e-6 (NTU120). Trọng số runtime camera không đổi.
