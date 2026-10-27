# Thử nghiệm chọn checkpoint bằng validation gốc và nhiễu

Mục tiêu: chạy **A và B riêng biệt**; trong mỗi lần train, so sánh hai cách chọn checkpoint.
Không sửa hoặc ghi đè file và kết quả A/B cũ. A dùng loss phân loại; B thêm 0,1 × symmetric JS như B trước.

## Các file mới

- `dataset_paired_dual_validation.py`: bản riêng của dataset train cặp A, giữ nguyên phép trộn.
- `prepare_fixed_noise_validation.py`: kiểm tra và đóng băng kế hoạch validation đã đánh giá trước.
- `config_fixed_noise_validation.yaml`: nguồn kế hoạch cũ và thư mục lưu kế hoạch mới.
- `fixed_noise_validation.py`: đánh giá các hỗn hợp cố định, dùng lại hàm preprocessing và trộn của evaluator cũ.
- `train_paired_dual_validation.py`: train ngân sách cố định và lưu hai loại best checkpoint.
- `config_train_paired_dual_validation_A.yaml`, `config_train_paired_dual_validation_B.yaml`: hai cấu hình độc lập. File không có hậu tố vẫn là A, giữ để tương thích lệnh trước.
- `test_dual_validation.py`: kiểm tra logic chọn checkpoint, tính chỉ số và trộn nhiễu.

## Lệnh chạy trong PowerShell

```powershell
cd C:\Users\Uyen\Documents\Master\LuanVan01\Code\ViSpeech
uv run --locked python prepare_fixed_noise_validation.py --config config_fixed_noise_validation.yaml
uv run --locked python train_paired_dual_validation.py --config config_train_paired_dual_validation_A.yaml --check-only
uv run --locked python train_paired_dual_validation.py --config config_train_paired_dual_validation_B.yaml --check-only
uv run --locked python train_paired_dual_validation.py --config config_train_paired_dual_validation_A.yaml
# Doi A hoan tat roi chay B:
uv run --locked python train_paired_dual_validation.py --config config_train_paired_dual_validation_B.yaml
```

Lệnh chuẩn bị tạo `dataset/fixed_noise_validation_dual`: chỉ CSV/YAML/JSON, không tạo audio mới.
Nếu đã tồn tại, lệnh chỉ kiểm tra tính toàn vẹn, không ghi đè. Nếu dữ liệu thay đổi, chương trình báo lỗi.
Lệnh `--check-only` không train và không tạo checkpoint. Hai lệnh không có cờ này chạy hai phiên riêng, tuần tự A rồi B. B không lấy checkpoint của A.
Mỗi lần gọi lệnh train tạo một phiên mới từ checkpoint xuất phát; chưa có chế độ resume khi bị ngắt.

## Cấu hình cố định trước khi chạy

- Xuất phát từ best sampler 1:1:1 ban đầu, không phải best A hoặc B.
- Train giống A: sampler 1:1:1, 16 cặp/batch, 8 file tiếng nền train, SNR bốc đều 0–20 dB.
- Learning rate 0,00001; weight decay 0,0005; seed 42; tối đa 3.060 bước.
- Kiểm tra validation tại bước 0 và mỗi 306 bước. **Không early stopping**.
- Validation gốc dùng toàn bộ 1.225 câu; noisy validation dùng cùng những câu đó ở 7 × 3 điều kiện.
- SNR noisy validation: 20/10/0 dB; Whole-clip RMS; không clip/rescale, giống evaluator trước.
- Ngưỡng accuracy gốc: accuracy bước 0 trừ **0,5 điểm phần trăm**. Đây là lựa chọn thử ban đầu, không phải ngưỡng đã chứng minh tối ưu.
- Chỉnh `validation.max_clean_accuracy_drop_pp` trước khi chạy nếu cần; `0.5` nghĩa là 0,5 điểm phần trăm, không phải 50%.
- Tiêu chí noisy: trung bình macro-F1 của 21 điều kiện, mỗi điều kiện có trọng số bằng nhau.
- Điểm bằng nhau giữ checkpoint sớm hơn. Bước 0 cũng là ứng viên của cả hai cách chọn.

## Theo dõi kết quả

Trước khi có STEP 0, chương trình đọc/cất tạm âm thanh trong RAM và đánh giá toàn bộ noisy validation.
Đây không phải treo hoặc train chưa khởi động được. Có thanh tiến trình validation.
Mỗi 50 bước có train loss trung bình; mỗi lần validation có train loss/accuracy, clean loss/accuracy/F1,
noisy F1 tại 20/10/0 dB, noisy F1 trung bình, ngưỡng clean và bước của hai checkpoint tốt nhất.
Noisy validation gồm 25.725 hỗn hợp mỗi lần nên tổng thời gian dài hơn A/B trước; không ước lượng chỉ từ thời gian train.

Kết quả trong `checkpoints/dialect/paired_dual_validation_A_<thời gian>/` và `paired_dual_validation_B_<thời gian>/`:

- `best_clean.pth`: accuracy validation gốc cao nhất.
- `best_robust.pth`: macro-F1 nhiễu trung bình cao nhất trong các checkpoint đạt ngưỡng accuracy gốc.
- `best_clean_info.json`, `best_robust_info.json`: bước được chọn và các chỉ số.
- `last.pth`: model tại lần validation cuối.
- `training_history.csv`: các chỉ số và thời gian train/validation.
- `validation_conditions_step_XXXXXX.csv`: chi tiết từng nhiễu/SNR, gồm recall từng miền.
- `augmentation_log.csv`: từng lượt trộn ngẫu nhiên trong train.
- `run_info.json`: cấu hình, trạng thái và checkpoint nguồn; `fixed_validation/`: kế hoạch cố định đã sao lưu.

## Phạm vi kết luận

Không tự động gọi noisy validation là test. Nguồn nhiễu đang dùng là 7 file đánh giá cũ và giờ được dùng để chọn model.
Chương trình phát hiện file nhiễu train/val trùng nội dung byte, nhưng không chứng minh chúng đến từ bản ghi nguồn độc lập.
Muốn báo cáo khả năng tổng quát sang nguồn nhiễu chưa gặp, cần kiểm tra nguồn và giữ nhiễu test độc lập.
Final test dùng câu test riêng và cùng cách trộn ở 20/10/0 dB. Bộ này chưa tạo hoặc chạy final test.
Hai checkpoint có thể giống nhau; đó là một kết quả hợp lệ. Nếu khác nhau, cần đánh giá trên test riêng để biết tiêu chí mới giúp tổng quát hơn không.

## So sánh với A/B trước

Giữ checkpoint xuất phát, dữ liệu, sampler, batch, seed, learning rate, optimizer, SNR, công thức loss và trọng số như A/B trước. Thay đổi quy trình validation/lựa chọn và bỏ early stopping để cả hai chạy đủ 3.060 bước. B trước dừng ở bước 2.448 và chọn bước 918: không quy mọi khác biệt so với B trước cho riêng noisy validation.

So sánh best_clean và best_robust trong cùng một phiên để xét tiêu chí chọn; so sánh A và B mới theo cùng tiêu chí để xét loss. Noisy validation không cập nhật trọng số và các trạng thái RNG được giữ trong lúc đánh giá. Không bảo đảm kết quả bit-for-bit với phiên cũ trên mọi môi trường GPU.

Cột train_loss là tổng loss; train_classification_loss và train_consistency_js được ghi riêng. Khi đối chiếu với val_original_loss, đọc loss phân loại và nhớ train gồm gốc+nhiễu còn validation loss là trên bản gốc.
