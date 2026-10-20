# Chạy sampler 1:1:1 bằng uv trong terminal VS Code

Giữ nguyên file baseline và checkpoint 1:2:1 cũ. Chỉ train thêm bản 1:1:1.
Chưa chạy train, đánh giá hoặc kiểm tra cú pháp trong lúc viết các file này.

Hướng dẫn dưới đây dùng **PowerShell trong terminal tích hợp của VS Code trên Windows**.
Nếu model đang train rồi, không nhập lại lệnh train và không bấm Ctrl+C để làm theo hướng dẫn;
chờ lần chạy đó hoàn tất. Các bước 1–2 dành cho lúc chưa bắt đầu train.

## Thứ tự thực hiện

1. Vào `Code/ViSpeech`, kiểm tra Python và CUDA bằng uv.
2. Chạy `uv run --locked python -u train_sampler111.py` và chờ train hoàn tất.
3. Điền đường dẫn `best_1.pth` mới vào `config_noise_eval_sampler111.yaml`, rồi lưu file.
4. Chạy `uv run --locked python evaluate_noise_robustness.py --config config_noise_eval_sampler111.yaml`.
5. Đối chiếu kết quả mới với kết quả 1:2:1 đã lưu; không train lại bản 1:2:1.

Lệnh đầy đủ và vị trí xem kết quả nằm trong từng bước bên dưới. `-u` giúp log train xuất ngay
trên terminal. Các lệnh kiểm tra, train và đánh giá đều dùng `uv run --locked python`.

## File dùng trong lần mới

- `sampler111.py`: giống cách lấy mẫu cũ nhưng không lấy thêm lớp Trung.
- `config_train_sampler111.yaml`: cấu hình train riêng.
- `train_sampler111.py`: dùng lại mô hình, dataset, augmentation và Adam của code cũ.
- `config_noise_eval_sampler111.yaml`: cấu hình riêng cho công cụ `evaluate_noise_robustness.py` cũ.

## Bước 1. Mở terminal VS Code và vào đúng thư mục

Trong VS Code, chọn **Terminal → New Terminal**. Dùng terminal **PowerShell**:
nếu terminal hiện tại là CMD hoặc Git Bash, chọn PowerShell ở menu cạnh nút `+` của vùng terminal.
Sau đó nhập:

```powershell
Set-Location 'C:\Users\Uyen\Documents\Master\LuanVan01\Code\ViSpeech'
```

Sau lệnh trên, đường dẫn ở đầu dòng terminal cần kết thúc bằng `Code\ViSpeech`.
Theo `README.md` và `LOCAL_SETUP.md`, dự án dùng **uv** với `pyproject.toml` và `uv.lock`.
Các lệnh bên dưới dùng `uv run --locked python` để chạy trong môi trường dự án;
**không cần chạy Activate.ps1** và không phụ thuộc chữ `(.venv)` ở đầu dòng.
`--locked` yêu cầu tệp khóa hiện tại hợp lệ, không tự cập nhật `uv.lock`; uv vẫn có thể đồng bộ
môi trường theo tệp khóa nếu cần. Không chạy `uv add`, `uv lock` hoặc nâng cấp thư viện cho lần đối chứng.

Có thể kiểm tra đúng Python và GPU bằng lệnh sau (chỉ in thông tin, không train):

```powershell
uv run --locked python -c "import sys, torch; print(sys.executable); print('CUDA:', torch.cuda.is_available())"
```

Mong đợi đường dẫn Python nằm trong `Code\ViSpeech\.venv` và `CUDA: True`.
Nếu không thấy CUDA, xử lý môi trường trước khi chạy train.

Có thể tự kiểm tra cú pháp trước khi train (không tải model hoặc train):

```powershell
uv run --locked python -m py_compile train_sampler111.py sampler111.py
```

## Bước 2. Train bản 1:1:1

```powershell
uv run --locked python -u train_sampler111.py
```

Chạy lệnh **một lần** trong terminal trên; không cần nút Run Python File hoặc Run All Cells.
Terminal sẽ in thư mục checkpoint và thanh tiến độ với tổng `91800` bước.
Cứ mỗi 306 bước sẽ xuất hiện dòng `STEP ... | EPOCH ... | Train Loss ... | Val Acc ...`.
Log đồng thời được lưu vào thư mục checkpoint; không cần tự chuyển hướng đầu ra sang file khác.

Giữ terminal đang chạy, máy được cấp nguồn và không chuyển sang sleep trong lúc train.
Bạn vẫn có thể đọc/sửa tài liệu trong VS Code; không sửa code hoặc dữ liệu đang dùng cho lần train đó.

Script đọc `config_train_sampler111.yaml`, bắt đầu model mới từ seed 42.
Không train tiếp từ checkpoint 1:2:1. Không thay đổi model, augmentation hoặc danh sách dữ liệu.

- 91.800 bước cập nhật = 300 epoch × 306 bước của lần cũ.
- Validation mỗi 306 bước; vẫn chọn top 3 checkpoint theo validation accuracy.
- Epoch mới có 230 batch: chạy 399 epoch đầy đủ và 30 batch của epoch 400.
- Một epoch đầy đủ lấy 2.448 lượt mẫu mỗi miền; các lớp ít file hơn được lặp lại.
- Batch cuối mỗi epoch mới có 16 mẫu; cùng số bước không đồng nghĩa chính xác cùng số lượt mẫu.
- Cùng seed không bảo đảm chuỗi augmentation giống hệt sau khi đổi sampler.

Kết quả vào thư mục riêng `checkpoints/dialect/resnet34_sampler111_<ngày_giờ>/`:

- `best_1.pth`, `best_2.pth`, `best_3.pth`: model tốt nhất theo validation accuracy.
- `best_1_info.json` (và bản 2/3): step, epoch, accuracy tương ứng.
- `last.pth`: trọng số ở lần validation gần nhất.
- `training_history.csv`, `training_log.txt`: kết quả theo mỗi lần validation.
- Bản sao cấu hình, script train, sampler và danh sách train/validation.

Các cột `draws_*` trong CSV đếm số lượt mẫu kể từ lần validation trước, không phải tổng cả quá trình.
Bản đơn giản **không hỗ trợ resume optimizer/RNG**; chạy lệnh lần nữa bắt đầu một lần train mới.
Không sửa dữ liệu hoặc code khi đang train.

## Bước 3. Đổi checkpoint trong cấu hình đánh giá mới

Mở `config_noise_eval_sampler111.yaml`, thay dòng checkpoint bằng đúng thư mục được in lúc train:

```yaml
model:
  checkpoint: checkpoints/dialect/THAY_BANG_THU_MUC_SAMPLER111/best_1.pth
```

Giữ nguyên các phần còn lại: validation, nguồn nhiễu, seed 42, batch size 4, SNR 20/10/0 dB,
repeats=1. Không sửa `config_noise_eval.yaml` cũ.

## Bước 4. Dùng lại công cụ đánh giá nhiễu cũ

Chỉ chạy sau khi terminal báo `Train xong` và bạn đã sửa checkpoint ở bước 3.
Nếu mở một terminal VS Code mới, chạy lại lệnh `Set-Location` ở bước 1 trước.

```powershell
uv run --locked python evaluate_noise_robustness.py --config config_noise_eval_sampler111.yaml
```

Đánh giá model mới trên bản gốc và bản thêm nhiễu của validation. Không train model.
Kết quả vào `logs/noise_robustness_sampler111/<mã_lần_chạy>/`.
Không cần chạy lại model 1:2:1; kết quả cũ ở:

```text
logs/noise_robustness/20261018_221428_db87c290/
```

## Bước 5. So sánh kết quả

Mở `summary.csv`, `predictions.csv` và các hình trong `figures/` của hai lần đánh giá:

1. Macro-F1 bản gốc và từng điều kiện nhiễu.
2. Recall từng miền, đặc biệt miền Trung.
3. Số Bắc/Nam bị nhầm sang Trung trong ma trận nhầm lẫn.
4. `correct_to_wrong_rate`: câu đúng ở bản gốc nhưng sai sau thêm nhiễu.
   Tập câu gốc đúng có thể khác nhau giữa hai model; không đọc riêng chỉ số này.

Trước khi diễn giải, kiểm tra `mixture_plan.csv` của hai lần có cùng mẫu, nguồn nhiễu, vị trí,
SNR và hệ số trộn. Khi dữ liệu không đổi, cùng cấu hình/seed sẽ tạo cùng kế hoạch trong công cụ hiện tại.
Lệnh kiểm tra nhanh dưới đây cần thay đường dẫn bằng thư mục kết quả mới thật:

```powershell
$newResults = 'logs/noise_robustness_sampler111/THAY_BANG_MA_LAN_CHAY'
Compare-Object (Get-Content 'logs/noise_robustness/20261018_221428_db87c290/mixture_plan.csv') (Get-Content "$newResults/mixture_plan.csv")
```

Không có dòng khác biệt nghĩa là nội dung kế hoạch khớp trong phép đối chiếu này.
Nếu có khác biệt, xem nguyên nhân trước khi kết luận tác động sampler.
Không coi việc giảm dự đoán Trung là cải thiện nếu lỗi chỉ chuyển sang lớp khác.

## Có cần chạy evaluate.ipynb không?

- Kiểm tra sampler và nhiễu trên validation: dùng **bước 4**.
- `evaluate.ipynb` đánh giá hai tập **clean test và noisy test có sẵn** của ViSpeech;
  không tự thêm nhiễu vào validation. Chưa cần dùng test để quyết định các chỉnh sửa đang nghiên cứu.
- Khi cần báo cáo test cuối cùng: sao chép notebook thành `evaluate_sampler111.ipynb`, sao chép
  `config_infer.yaml` thành `config_infer_sampler111.yaml`, đổi checkpoint trong cấu hình mới,
  rồi sửa lời gọi trong notebook mới thành `config = load_config('config_infer_sampler111.yaml')`.
  Chọn kernel `.venv`, restart kernel và chạy từ đầu để không giữ kết quả/model cũ trong bộ nhớ.

Folder hiện tại không có `evaluate.py`; có `evaluate.ipynb` và `evaluate_noise_robustness.py`.
