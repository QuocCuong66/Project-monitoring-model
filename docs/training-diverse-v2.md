# Đợt dữ liệu có nhãn bổ sung: diverse-v2

Đợt này so sánh với **model ONNX thực sự đang chạy trước đợt train**, SHA256
`ae1668159bc4a3fae04bc1ac9a46cc852492315f06a4d663b283bb10631bb244`.
Không dùng lại model `.pt` gốc làm mốc thấp hơn. Bản đóng băng gồm ONNX, hai
checkpoint thành phần, cấu hình model và thresholds tại
`models/checkpoints/diverse_v2_baseline/`. Trainer kiểm tra parity trước khi học.

## Nguồn và kiểm tra chất lượng

Nguồn video/nhãn pha: [Zenodo, Bouhafs Abdelkader, bản 1.0](https://zenodo.org/records/17170592).
Đây là nhãn 10 pha của bản phát hành này, không phải mặc định taxonomy 16 lớp của
mọi phiên bản OmniFall. Nguồn gốc:
[CAUCAFall](https://data.mendeley.com/datasets/7w7fccy7ky/4),
[GMDCSA24](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos).
Giữ thông tin trích dẫn và quyền của từng nguồn; trang bản nhãn tổng hợp không ghi
rõ license trong metadata. Video tải về không được đưa vào Git.

Tải 3.862.149.617 byte: CAUCAFall 173.504.611, EDF 991.221.216,
GMDCSA24 1.671.761.923 và OOPS 1.025.661.867. Mỗi ZIP được kiểm MD5 do bên phát
hành công bố, SHA256 nội bộ và CRC; biên nhận nằm cạnh ZIP trong
`data/raw/diverse_v2/`. Dung lượng tải gồm cả OOPS đã bị loại, không phải toàn bộ
dữ liệu có thể dùng để train.

- Chỉ nhận video được liệt kê trong file annotation chuẩn của từng ZIP; không
  đếm các bản xuất khác trong ZIP thành mẫu mới. Kiểm SHA256 khi CRC/kích thước
  gợi ý trùng; trùng giữa hai split làm quy trình dừng. Hai nhãn mâu thuẫn trên
  cùng nội dung bị cách ly cả hai bản.
- EDF có một số video bắt đầu đúng thời điểm nhãn nhưng dư phần cuối. Đã đối
  chiếu khung hình chồng lấp giữa các pha ở nhiều người/góc quay, lưu bằng chứng
  `edf_overlap_audit.json`. Chỉ đọc khoảng `end-start` từ đầu video được cắt.
  Với nguồn chưa được kiểm chứng, thời lượng không khớp bị loại; không đoán vị
  trí nhãn chỉ từ độ dài file. Mã chuẩn bị/cache đã phân phiên bản để không dùng
  lại cửa sổ sai từ bước khảo sát.
- OOPS bị loại **trước khi train hoặc xem kết quả model** do file thiếu, các bản
  trùng khác nhãn và sai thời lượng trên phần lớn file. Xem
  `data/raw/diverse_v2/oops_duration_audit.json`. Không tuyên bố đã kiểm chứng trên
  tai nạn thực tế bằng bản phát hành lỗi này.
- RTMO-s giống bản chạy thực tế, COCO-17 `(x,y,confidence)`, lấy mẫu theo FPS thật
  về 15 Hz rồi dùng `build_model_sequence` chung với inference. Video FPS thấp
  lặp khung đo gần nhất theo dòng thời gian, không nén nhiều giây thành 2 giây.
  Loại đoạn có nhiều người gây mơ hồ và cửa sổ pose chất lượng thấp. Lưu timestamp,
  bbox, confidence và ID nguồn; không tạo nhãn đoán cho `other`.

## Nhãn, chia tập và train

| Nguồn mới | Người train | Validation | Test |
|---|---|---|---|
| CAUCAFall | 1–6 | 7–8 | 9–10 |
| GMDCSA24 | 1–2 | 3 | 4 |
| EDF | 1–3 | 4 | 5 |

Mọi pha và camera của cùng người ở cùng split. Cửa sổ cùng đoạn có tương quan;
số cửa sổ không phải số người/video độc lập.

`walk → walking`, `fall → falling`, `fallen/lying → lying`,
`sit_down/sitting → sitting`, `stand_up → getting_up`, `standing → standing`.
`lie_down` là nhãn tập hợp `{sitting, lying, bending}`, tối ưu xác suất tổng,
không gán cứng một tư thế cho cả chuyển động. `other` bị loại.
Không tự gán `fallen` thành `immobile`: ngữ cảnh ngã và bất động do risk engine
theo thời gian xử lý. Các đoạn nằm trong recording có ngã không được tính là
negative lành tính khi đo false positive.

Train student khởi tạo từ checkpoint đã học đợt RGB trước. Lấy mẫu 55% dữ liệu
cũ / 45% dữ liệu mới, cân bằng nguồn và nhãn; 18.000 lượt/epoch, tối đa 30 epoch,
AdamW 2,5e-5, distillation với ensemble đang chạy, augmentation lật/nhiễu/khuyết
khớp. Chọn epoch và trọng số blend **chỉ trên validation**. Giữ tỷ lệ cặp
lying/immobile của teacher trong blend để tránh làm mất khả năng nhận ra bất động.
Không đổi thresholds chỉ để làm đẹp số báo giả.

Tiêu chí đóng băng trước khi đánh giá: giảm ít nhất 10% FPR cửa sổ trên nguồn mới;
không giảm recall ngã nguồn mới; các giới hạn hồi quy fall/lying/immobile/emergency
cũ lần lượt 0,5/1/2/1 điểm phần trăm, FPR sinh hoạt cũ không tăng. Test chỉ đánh
giá sau khi chọn xong; OOPS bị loại bằng kiểm tra file, không dựa trên metric model.

## Chạy lại

```powershell
.venv/Scripts/python.exe -u training/download_diverse_v2.py
.venv/Scripts/python.exe -u training/audit_diverse_v2.py
.venv/Scripts/python.exe -u training/prepare_diverse_v2.py --sources caucafall edf gmdcsa24
.venv/Scripts/python.exe -u training/train_diverse_v2.py
.venv/Scripts/python.exe -u training/replay_diverse_v2.py
.venv/Scripts/python.exe -u training/promote_diverse_v2.py
.venv/Scripts/python.exe -m pytest tests -q
```

Trainer ghi checkpoint, lịch sử mọi phép chọn validation và test tại
`models/checkpoints/diverse_v2.metrics.json`. Nếu không vượt tiêu chí, vẫn lưu
student để tái lập nhưng không xuất/tích hợp bản kém hơn. Promotion kiểm tra
held-out regression, parity Torch/ONNX/DirectML, input `(30,17,3)`, 10 output,
p95 ≤25 ms, hash bản đang chạy; sao lưu và rollback nếu nạp runtime thất bại.

Replay dùng action classifier, anomaly scorer, kinematics và risk/debounce thực,
trên recording lành tính của người thuộc test. Khoảng thiếu/nhãn không rõ cắt
lịch sử; không nối các video khác nhau để giả lập một lần ngã. Đây là phát lại
pose với ID người cố định, không phải thử toàn bộ camera/tracking/motion gate.
Ghi số phút quan sát và độ dài liên tục; không ngoại suy thành báo giả/đêm.

Chưa có nhãn ngủ qua đêm hoặc đủ bằng chứng riêng cho mọi kiểu lăn trở mình.
Độ chính xác cửa sổ không thay thế đánh giá cảnh báo theo sự kiện trên camera
thực tế, và lượng GB dữ liệu không đảm bảo xuất bản khoa học.

## Kết quả thực nghiệm đã hoàn tất — 2026-11-03

| Dữ liệu mới sau lọc | Train | Validation | Test | Tổng |
|---|---:|---:|---:|---:|
| CAUCAFall | 328 | 103 | 75 | 506 |
| EDF | 551 | 98 | 282 | 931 |
| GMDCSA24 | 134 | 81 | 43 | 258 |
| **Tổng mới** | **1.013** | **282** | **400** | **1.695** |

Trộn với 48.810 cửa sổ train cũ: tổng pool train 49.823. Chạy 12 epoch do early
stopping; chọn epoch 4, alpha 0,25 (75% ensemble đang chạy + 25% student mới).
CAUCAFall/EDF/GMDCSA24 lần lượt có 182/354/107 clip giải mã được giữ qua bước
lọc mơ hồ người; tiếp tục loại cửa sổ pose kém. GMDCSA24 loại 279 clip có nhiều
người gây mơ hồ; không tự gán nhãn người đứng cạnh cho người đang ngã/nằm.
EDF cắt phần dư ở 445 clip. Raw của ba nguồn dùng được là 2.836.487.750 byte;
OOPS 1.025.661.867 byte chỉ giữ làm audit, không train/test.

| Chỉ số | Bản đang chạy trước đợt này | Bản đã tích hợp |
|---|---:|---:|
| Validation cũ NTU60: accuracy | 87,18% | 88,41% |
| Validation cũ NTU60: fall recall | 98,14% | 97,79% |
| Validation cũ NTU60: lying recall | 62,27% | 62,06% |
| Validation cũ NTU60: immobile recall | 55,82% | 55,82% |
| Validation cũ NTU60: emergency recall | 84,63% | 84,07% |
| Validation cũ NTU60: normal FPR | 7,50% | 6,04% |
| Validation nguồn mới: normal FPR, trung bình đều 3 nguồn | 25,10% | 17,90% |
| Validation nguồn mới: fall recall, trung bình đều 3 nguồn | 30,17% | 40,64% |
| Test sinh hoạt cũ: normal FPR, trung bình đều NTU120/UCF101/HMDB51 | 10,96% | 9,37% |
| Test sinh hoạt cũ: số cửa sổ gán nhầm nhóm nguy hiểm / 45.232 | 6.108 | 5.432 |
| Test nguồn mới: normal FPR, trung bình đều 3 nguồn | 17,46% | 16,12% |
| Test nguồn mới: số cửa sổ gán nhầm nhóm nguy hiểm / 219 | 29 | 28 |
| Test nguồn mới: fall recall, trung bình đều 3 nguồn | 28,33% | 31,78% |
| Test nguồn mới: số cửa sổ ngã nhận đúng / 67 | 24 | 26 |
| Test UP-Fall: cửa sổ chạm đất nhận đúng | 11/16 | 11/16 |
| Test URFD: F1 cửa sổ chuyển động ngã | 0,3729 | 0,4727 |

Đây là **chỉ số phân loại cửa sổ**, nhóm nguy hiểm gồm falling, stumbling,
abnormal_movement, immobile; không phải số thông báo cảnh báo của risk engine.
Trung bình đều theo nguồn khác tỷ lệ gộp mọi cửa sổ. Ví dụ 6.108→5.432 tương
đương giảm 11,07% số cửa sổ nhầm trong 45.232 cửa sổ, không phải giảm 11,07%
thông báo/giờ. Cửa sổ có tương quan, đặc biệt GMD test chỉ có 6 negative hợp lệ.
Test nguồn mới còn có recall ngã thấp; không tuyên bố đã đạt độ chính xác cao
trên mọi người/góc quay hay có thể bỏ các lớp kinematics/risk/debounce.

Replay lành tính chỉ có **22,17 giây hợp lệ** từ 8 recording, đoạn liên tục dài
nhất **7,10 giây**. Cả baseline và candidate đều 0 cảnh báo: không có hồi quy
quan sát được, **chưa chứng minh giảm cảnh báo thực tế**. Riêng nhóm recording
chắc chắn lành tính có rất ít cửa sổ nằm ở validation/test; không suy ra model
đã giải quyết ngủ lâu, nằm từ đầu camera hoặc mọi kiểu lăn trở mình.

Đã thay `models/checkpoints/best_st_transformer.onnx`, giữ nguyên cấu hình risk
và cơ chế debounce đang có. Model gồm ba transformer nhỏ, 7.084.160 byte,
input `(30,17,3)`, 10 lớp. DirectML: median **6,54 ms**, p95 **7,83 ms** trên
60 lần batch 1 sau warmup; sai số parity ONNX/Torch tối đa **9,54e-7**.

- SHA256 mới: `e8f21aae295f285e0b61d01d53fb4fb5a8414981b12f80040262ae7b95b43fbd`.
- Backup: `models/checkpoints/backups/diverse_v2_20261103T150056Z/`.
- Artifact: `diverse_v2.pt`, `.onnx`, `.metrics.json`, `.replay.json`,
  `.deployment.json` trong `models/checkpoints/`.
- Khởi động lại tiến trình chạy từ dự án để nạp ONNX mới; tiến trình đã nạp
  model vào bộ nhớ không tự đổi chỉ vì file trên đĩa được thay.

Sau tích hợp: **241 kiểm thử đạt**, 5 cảnh báo deprecation của dependency hiện có.
Kết quả kiểm thử và hash kiểm tra lại được lưu trong deployment report.
