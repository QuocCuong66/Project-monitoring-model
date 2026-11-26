# Huấn luyện giảm nhầm hành động bình thường

## Dữ liệu và giới hạn

- Nguồn gốc: [NTU RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/),
  bản COCO-17 skeleton 2D do [MMAction2](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md) công bố.
- Dùng lại `data/raw/ntu/ntu60_2d.pkl` có sẵn; không tải thêm bản trùng lặp.
- Bổ sung A08 (sit down, nhãn zero-based 7) trước đây bị bỏ qua, ánh xạ sang
  `sitting`. Cúi nhặt đồ, mang/cởi giày, đứng dậy và đi bộ vẫn là các mẫu đối chứng.
- Tập mới: `data/processed/ntu_hard_negatives.npz`, 49.695 cửa sổ, mỗi cửa sổ
  30 frame × 17 khớp × 3 giá trị; 35.066 train và 14.629 validation.
- Giữ quy trình cross-subject của NTU: người trong validation không nằm trong train.
  Các cửa sổ cùng clip có tương quan, không phải 49.695 video độc lập.
- Nhãn nằm/bất động một phần là dữ liệu tổng hợp từ tư thế xoay và đoạn cuối clip ngã.
  Đây không phải bằng chứng xác nhận chất lượng phát hiện cấp cứu thực tế.
- Validation dùng chọn checkpoint, không phải một test set độc lập. Cần kiểm tra
  video camera thực, đặc biệt ngồi nhanh, nằm chủ động, tập thể dục và các động tác giả.

## Tái lập (PowerShell, Python 3.11 của dự án)

```powershell
.venv/Scripts/python.exe training/prepare_ntu.py --out data/processed/ntu_hard_negatives.npz
.venv/Scripts/python.exe training/evaluate.py --data data/processed/ntu_hard_negatives.npz --weights models/checkpoints/best_st_transformer.pt --output-dir results/hard_negative_baseline
.venv/Scripts/python.exe training/train_action.py --data data/processed/ntu_hard_negatives.npz --resume models/checkpoints/best_st_transformer.pt --epochs 5 --batch-size 64 --lr 0.00005 --workers 0 --out models/checkpoints/hard_negative_candidate.pt
.venv/Scripts/python.exe training/evaluate.py --data data/processed/ntu_hard_negatives.npz --weights models/checkpoints/hard_negative_candidate.pt --output-dir results/hard_negative_candidate
```

`evaluate.py` yêu cầu dữ liệu và trọng số thật. Không sinh dữ liệu ngẫu nhiên để
báo cáo chất lượng. `--split 2` chỉ dùng được nếu archive có test split riêng.
Báo cáo chứa confusion matrix, precision/recall/F1 từng lớp và từng nhãn nguồn NTU.

`normal_emergency_fpr` đo tỷ lệ cửa sổ nhãn bình thường bị dự đoán thành một
trong `falling`, `stumbling`, `abnormal_movement`, `immobile`. Chỉ số này **không
phải** số chuông cảnh báo sai mỗi giờ: runtime còn lọc vật lý, ngữ cảnh và state machine.

Model ứng viên được lưu riêng, chọn theo macro-F1 và so với baseline trên cùng dữ liệu.
Không được chỉ tăng ngưỡng hoặc đổi model theo accuracy tổng mà bỏ qua recall ngã.

Kết quả lượt fine-tune 5 epoch ngày 2026-11-03 (cùng 14.629 cửa sổ validation):

| Chỉ số | Model hiện tại | Ứng viên |
|---|---:|---:|
| Accuracy | 83,30% | 89,79% |
| Macro-F1 | 80,09% | 81,87% |
| Normal → emergency FPR | 11,70% | 8,10% |
| Emergency recall (gộp) | 84,98% | 89,56% |
| Falling recall | 98,7% | 97,4% |

Chưa thay trọng số runtime: recall riêng ngã giảm dù chỉ số gộp tốt hơn.
Ứng viên: `models/checkpoints/hard_negative_candidate.pt` và `.onnx`.
Báo cáo đầy đủ: `results/hard_negative_baseline/evaluation_summary.json` và
`results/hard_negative_candidate/evaluation_summary.json`.

## Giữ ngữ cảnh nằm/ngủ khi tư thế bị nhiễu

`lying_context.upright_reset_seconds` yêu cầu tư thế thẳng liên tục trước khi xoá
ngữ cảnh nằm hoặc ghi nhận lịch sử đứng/ngồi thẳng. Một mẫu khung xương thẳng giả
khi trở mình không còn tạo ra chuỗi “đứng → ngã” mới. Vận tốc hông lớn khi vừa nhận
một người đã nằm cũng không tự tạo cú rơi nếu chưa quan sát tư thế thẳng trước đó.

Các kiểm thử trong `tests/test_lying_safe_scope.py` bao gồm ngủ với nhiễu tư thế,
vận tốc nhiễu lúc bắt đầu theo dõi, và nằm rồi đứng dậy thật trước khi ngã.
Nằm chủ động nhanh có thể giống ngã về hình học; khi biết vị trí giường/sofa,
khai báo đúng `lying_context.rest_zones` cho camera giúp phân biệt ngữ cảnh này.

## Nguồn bổ sung cho nằm chủ động

[SAFER-Activities](https://safer-activities.github.io/) có nhãn `Lying posture`,
`Lying down`, `Fall`, skeleton 2D và phân chia theo người/camera; phù hợp để bổ sung
những ví dụ nằm chủ động thật thay cho chỉ xoay tư thế tổng hợp. Chưa đưa nguồn này
vào lượt train hiện tại: [kho dữ liệu](https://huggingface.co/datasets/SAFER-Activities/SAFER-Activities)
yêu cầu chủ tài khoản đăng nhập và đồng ý chia sẻ thông tin Hugging Face trước khi
truy cập file. Không coi nhãn `Lying posture` là bằng chứng một người đang ngủ.
