# Mô hình Phát hiện Đối tượng Anchor-Free từ đầu với Backbone ResNet-50 & FPN (ResNetYOLO)

Mô hình **ResNetYOLO** được xây dựng và tối ưu hóa tối đa nhằm mục đích phát hiện đối tượng từ đầu (Object Detection from scratch) trên GPU T4 của nền tảng **Lightning AI** (hoặc các GPU NVIDIA khác). Mô hình sử dụng mạng trích xuất đặc trưng `ResNet-50` mạnh mẽ kết hợp cùng **bộ đầu FPN (Feature Pyramid Network)** dung hợp đặc trưng đa quy mô (stride-16 và stride-32) cho ra lưới dự đoán mịn gấp đôi (**$28 \times 28$**), hàm mất mát tối tân **CIoU Loss**, tăng cường dữ liệu nhiều kích thước (**Multi-Scale Training**) và tăng tốc huấn luyện bằng **Độ chính xác hỗn hợp (Mixed Precision - AMP)**.

---

## 📂 Cấu trúc Thư mục Nộp bài

```
<my_submission>/
├── models/
│   ├── __init__.py
│   └── detector.py            # Định nghĩa kiến trúc mô hình (ResNet-50 + FPN Fusion)
├── utils/
│   ├── __init__.py
│   ├── dataset.py             # Bộ đọc dữ liệu nâng cao, Augmentations & Multi-Scale
│   ├── loss.py                # Hàm mất mát tùy chỉnh CIoU Loss + Smooth L1 + Focal Loss
│   └── nms.py                 # Giải mã hộp bao & thuật toán Class-wise NMS
├── train.py                   # Script huấn luyện chính trên GPU (AMP + Cosine Decay + LR vi sai)
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
  --epochs 50 \
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

1. **Kiến trúc FPN Dung hợp Đa quy mô & Lưới Mịn ($28 \times 28$):**
   * Sử dụng **ResNet-50** kết hợp cấu trúc **FPN (Feature Pyramid Network)** dung hợp đặc trưng Stride-16 (`layer3`) và Stride-32 (`layer4`) để tạo ra bản đồ đặc trưng giàu ngữ nghĩa và sắc nét về không gian.
   * Lưới dự đoán nâng lên **$28 \times 28$** (784 ô lưới), giúp phát hiện xuất sắc các vật thể nhỏ và crowded ở cự ly xa (như `chair` và `car`).
2. **Hàm mất mát Focal Loss & CIoU Loss:**
   * Tự triển khai **CIoU Loss (Complete IoU)** tối ưu hóa trực tiếp độ trùng khớp, khoảng cách tâm và tỉ lệ khung hình.
   * Sử dụng **Focal Loss** cho Objectness để cân bằng triệt để sự mất cân xứng tiền cảnh/hậu cảnh của lưới $28 \times 28$ (tỉ lệ 1:100).
   * Tích hợp **Trọng số phân phối lớp** để giải quyết mất cân bằng nhãn.
3. **Tăng cường dữ liệu nhiều kích thước (Multi-Scale Training):**
   * Huấn luyện co dãn kích thước động từ $384$ đến $480$ pixel giúp mô hình có tính bất biến tỷ lệ (scale invariant).
4. **Tốc độ học vi sai (Differential Learning Rates):**
   * Backbone ResNet-50 chạy với learning rate $1e-4$ cực nhỏ, trong khi FPN và Custom Head chạy với learning rate $1e-3$ để tối ưu hóa hiệu năng tốt nhất.
5. **AMP & NMS:**
   * Tăng tốc Mixed Precision FP16 bằng `torch.amp` (không cảnh báo) và áp dụng Class-wise NMS để triệt tiêu hộp bao trùng lặp một cách tối ưu.
