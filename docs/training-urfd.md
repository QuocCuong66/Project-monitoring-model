# UR Fall Detection: tải dữ liệu và fine-tune

Nguồn chính thức: https://fenix.ur.edu.pl/~mkepski/ds/uf.html

URFD có 30 lần ghi ngã và 40 lần ghi sinh hoạt. Ngã có hai camera, sinh hoạt chỉ
có camera 0: tổng cộng 100 chuỗi RGB và 100 chuỗi depth. Script tải 200 ZIP và
142 CSV, lưu SHA-256 và kiểm tra CRC từng archive. Không cần giải nén PNG ra đĩa.

Trích dẫn dữ liệu: Bogdan Kwolek, Michal Kepski, *Human fall detection on embedded
platform using depth maps and wireless accelerometer*, Computer Methods and
Programs in Biomedicine 117(3), 489–501, 2014. Nguồn công bố giấy phép
CC BY-NC-SA 4.0, dành cho sử dụng học thuật phi thương mại.

## Chạy lại

Tại thư mục dự án, dùng Python 3.11 trong `.venv`:

```powershell
.venv/Scripts/python.exe training/download_urfd.py
.venv/Scripts/python.exe training/prepare_urfd.py
.venv/Scripts/python.exe training/train_urfd.py
```

Có thể chạy `prepare_urfd.py --wait-download` khi download còn tiếp diễn. Tất cả
đường dẫn, tỷ lệ chia, thông số chất lượng pose và huấn luyện ở `configs/urfd.yaml`.
Download có thể tiếp tục từ file `.part`, không cần tải lại các file đã xác thực.

## Nhãn và giới hạn đánh giá

CSV chính thức chứa nhãn tư thế theo frame camera 0: -1 không nằm, 0 chuyển tiếp,
1 nằm trên mặt đất. Không gán mọi frame trong clip ngã thành `falling`, và không
gán `standing` cho mọi tư thế -1. Đối với lớp không xác định được chính xác,
loss tối ưu tổng xác suất của tập lớp được nhãn gốc cho phép.

- Nhãn 1 ánh xạ `lying` ở cả clip ngã và clip sinh hoạt. Không suy ra `immobile`
  chỉ từ tư thế nằm.
- Cửa sổ trong clip ngã có chứa frame chuyển tiếp 0 được dùng học `falling`.
- Frame chuyển tiếp 0 trong clip sinh hoạt không được gán `falling`.
- Chuỗi RGB camera 0 ghép nhãn theo chỉ số frame của cặp RGB/depth. Camera 1 tải
  về nhưng không train vì CSV tư thế chính thức chỉ chú thích camera 0.
- Pose RTMO-s dùng cùng model như runtime. Depth được lưu trữ, không phải đầu vào
  của model hành động RGB-pose đang fine-tune.
- Chia theo lần ghi, có phân tầng ngã/sinh hoạt: 42 train, 14 validation, 14 test.
  Đây **không phải** đánh giá độc lập theo người; mã người không có trong metadata
  đang sử dụng. Không gọi hai góc quay của cùng một lần ngã là hai sự kiện độc lập.
- Giữ lại một phần NTU train để giảm quên các lớp hành động khác.
- Chọn checkpoint và ngưỡng trên validation, chỉ đánh giá test sau khi chọn xong.
- Báo cáo cả kết quả theo cửa sổ và khả năng nhận ra giai đoạn chuyển tiếp theo
  lần ghi. Các chỉ số này không phải độ chính xác cảnh báo của toàn bộ risk engine,
  cũng không phải số báo động sai mỗi giờ sinh hoạt liên tục.

URFD không phải bộ theo dõi giấc ngủ qua đêm. Kết quả trên URFD không chứng minh
khả năng phân biệt ngủ với cấp cứu hoặc hiệu quả y khoa.

## Artifact

- Raw: `data/raw/urfd/`, kèm `download_manifest.json`, nguồn HTML và CSV gốc.
- Pose từng lần ghi: `data/poses/urfd/`.
- Cửa sổ và danh sách chia: `data/processed/urfd_windows.npz`, `.json`.
- Ứng viên: `models/checkpoints/urfd_candidate.pt`, `.onnx`, `.metrics.json`.
- Dự đoán test lưu trong `urfd_candidate.test_predictions.npz` để kiểm toán.

Các trọng số runtime hiện tại không bị ghi đè.

## Kết quả lượt chạy 2026-11-03

Đã tải và xác thực đủ 342 file: 200 ZIP + 142 CSV, tổng 8.748.079.383 byte
(~8,748 GB), không còn `.part`. File ZIP khớp kích thước công bố trên máy chủ;
đã kiểm tra CRC và lưu SHA-256 trong manifest. Giữ ZIP, không nhân đôi dung lượng
bằng cách giải nén toàn bộ PNG.

Tạo pose cho đủ 70 clip camera 0. Có 761 cửa sổ hợp lệ (433 train, 175 validation,
153 test). Ba clip `fall-16`, `fall-21`, `fall-22` chỉ có 28 bước sau lấy mẫu, ngắn
hơn cửa sổ 30 bước nên không tạo cửa sổ; không clip nào trong ba clip này thuộc test.
Lượt này không đệm thêm frame cho các clip ngắn. Ngoài ra loại 66 cửa sổ vì chất
lượng pose và 5 cửa sổ thiếu nhãn. Train 20 epoch, replay
3.000 cửa sổ NTU từ train; chọn epoch 16 bằng validation.

| Chỉ số test | Model cũ | Ứng viên URFD |
|---|---:|---:|
| Window F1 | 36,62% | 51,06% |
| Window recall chuyển tiếp ngã | 65,00% | 60,00% |
| Window FPR | 28,57% | 11,28% |
| Clip ngã phát hiện trong cửa sổ chuyển tiếp | 4/6 | 5/6 |
| Clip ADL có dự đoán ngã sai | 7/8 | 4/8 |

Ngưỡng mỗi model chọn riêng trên validation: model cũ 0,2; ứng viên 0,7.
Không lấy test để chọn ngưỡng. Tập test chỉ có 6 lần ngã và 8 ADL, nên chưa đủ để
khẳng định hiệu quả triển khai; 4/8 clip ADL vẫn bị dự đoán sai ít nhất một cửa sổ.
Không thay model runtime.

Kiểm tra hồi quy trên NTU 2D validation mở rộng (14.629 cửa sổ): accuracy 88,67%,
macro-F1 80,51%, normal→emergency FPR 8,61%; báo cáo ở
`results/urfd_ntu_regression/evaluation_summary.json`. Đây không phải test độc lập.
ONNX/PyTorch đã đối chiếu trên mẫu pose thật, sai lệch logits tối đa ~4,77e-7.
