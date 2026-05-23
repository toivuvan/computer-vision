# Mô hình Phát hiện Đối tượng Anchor-Free từ đầu với Backbone ResNet-34 (ResNetYOLO)

Mô hình **ResNetYOLO** được xây dựng và tối ưu hóa tối đa nhằm mục đích phát hiện đối tượng từ đầu (Object Detection from scratch) trên GPU T4 của nền tảng **Lightning AI** (hoặc các GPU NVIDIA khác). Mô hình sử dụng mạng trích xuất đặc trưng `ResNet-34` mạnh mẽ kết hợp cùng Detection Head tùy biến tích chập mịn ($14 \times 14$), hàm mất mát tối tân **CIoU Loss**, tăng cường dữ liệu nhiều kích thước (**Multi-Scale Training**) và tăng tốc huấn luyện bằng **Độ chính xác hỗn hợp (Mixed Precision - AMP)**.

---

## 📂 Cấu trúc Thư mục Nộp bài

```
<my_submission>/
├── models/
│   ├── __init__.py
│   └── detector.py            # Định nghĩa kiến trúc mô hình (ResNet-34 + Custom Head)
├── utils/
│   ├── __init__.py
│   ├── dataset.py             # Bộ đọc dữ liệu nâng cao, Augmentations & Multi-Scale
│   ├── loss.py                # Hàm mất mát tùy chỉnh CIoU Loss + Smooth L1
│   └── nms.py                 # Giải mã hộp bao & thuật toán Class-wise NMS
├── train.py                   # Script huấn luyện chính trên GPU (AMP + Cosine Decay)
├── predict.py                 # Script chạy suy luận chuẩn định dạng đầu ra
├── README.md                  # Tài liệu hướng dẫn sử dụng (tệp tin này)
└── requirements.txt           # Danh sách các thư viện Python cần thiết
```

---

## 🚀 Hướng dẫn Thiết lập và Sử dụng trên Lightning AI (GPU T4)

### Bước 1: Cài đặt Môi trường
Sau khi mở Studio hoặc Terminal trên Lightning AI (chọn cấu hình GPU T4), hãy chạy lệnh sau để cài đặt đầy đủ các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

### Bước 2: Huấn luyện Mô hình
Chạy lệnh huấn luyện bắt buộc sau để bắt đầu tối ưu hóa mô hình. Quá trình này sẽ sử dụng Mixed Precision (AMP) giúp chạy cực kỳ nhanh và lưu checkpoint tốt nhất vào thư mục `./models/best.pth`.
```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/ \
  --epochs 35 \
  --batch_size 32 \
  --lr 1e-3
```
*Lưu ý:* Mặc định tham số `--multi_scale` được kích hoạt để huấn luyện mô hình ở nhiều độ phân giải khác nhau từ $384 \times 384$ đến $480 \times 480$ pixel, tạo ra độ bền vững tuyệt đối cho kết quả.

### Bước 3: Chạy Suy luận (Inference)
Sử dụng mô hình tốt nhất đã được huấn luyện để tạo ra tệp dự đoán `predictions.json` trên tập ảnh bất kỳ bằng lệnh bắt buộc sau:
```bash
python predict.py \
  --image_dir ./public/val/images \
  --output predictions.json \
  --checkpoint ./models/best.pth
```

### Bước 4: Tự chấm điểm và Đánh giá (mAP@0.5)
Bạn có thể kiểm tra trực tiếp chất lượng của kết quả vừa dự đoán trên tập kiểm định để đo lường điểm số bằng script đánh giá được cung cấp:
```bash
python public/tools/evaluate_predictions.py \
  --ground_truth ./public/annotations/val.json \
  --predictions predictions.json \
  --output score.json
```
Xem nội dung tệp `score.json` để biết điểm số mAP@0.5 thực tế đạt được của mô hình!

---

## 🛠️ Điểm nhấn Công nghệ của Giải pháp

1. **Kiến trúc Anchor-Free Lưới Mịn ($14 \times 14$):**
   * Giảm thiểu sự phức tạp của các hộp neo (Anchor Box), tăng tốc độ hội tụ và giảm lỗi định vị. Với độ phân giải $448 \times 448$, mạng lưới $14 \times 14$ ô lưới giúp bắt trọn cả các đối tượng có kích thước nhỏ hoặc xếp sát nhau.
2. **Hàm mất mát Complete IoU (CIoU Loss):**
   * Được tự triển khai 100% từ đầu trong `utils/loss.py`. Khác với Smooth L1 thông thường, CIoU Loss tối ưu hóa trực tiếp sự trùng khớp diện tích (IoU), khoảng cách tâm hộp bao, và sự tương đồng về tỉ lệ khung hình (Aspect Ratio), giúp đường bao dự đoán khớp khít tối đa với nhãn thực tế.
3. **Tăng cường dữ liệu nhiều kích thước (Multi-Scale Training):**
   * Ở mỗi epoch, mô hình tự động thay đổi độ phân giải đầu vào ngẫu nhiên trong khoảng $384$ đến $480$ pixel. Việc này bắt buộc mạng nơ-ron phải học cách trích xuất đặc trưng có tính bất biến với tỉ lệ kích thước (scale invariant), cải thiện mạnh mẽ khả năng phát hiện trên tập kiểm tra ẩn.
4. **Tăng tốc Mixed Precision (AMP):**
   * Tận dụng lõi Tensor Cores trên GPU T4 để tính toán song song với độ chính xác hỗn hợp FP16 và FP32, giúp giảm một nửa thời gian huấn luyện và tiết kiệm bộ nhớ đồ họa.
5. **Class-wise Non-Maximum Suppression (NMS):**
   * Triển khai thuật toán NMS độc lập trên từng lớp đối tượng để loại bỏ trùng lặp mà không gây triệt tiêu chéo giữa các đối tượng khác loại ở cùng một vị trí.
