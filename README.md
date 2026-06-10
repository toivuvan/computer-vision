# Anchor-Free Object Detection với ConvNeXt-Tiny Backbone

Project này cài đặt một pipeline phát hiện đối tượng tự xây dựng bằng PyTorch cho 5 lớp: `person`, `car`, `dog`, `cat`, `chair`. Mô hình không dùng detector hoàn chỉnh có sẵn như YOLO, Detectron2, MMDetection, Faster R-CNN hay SSD có sẵn. Phần detector, target grid, loss, decode và NMS được tự cài đặt; backbone trích xuất đặc trưng dùng `ConvNeXt-Tiny` pretrained ImageNet, phù hợp với yêu cầu cho phép dùng mạng trích xuất đặc trưng đã huấn luyện trước khi được giảng viên cho phép.

Kiến trúc chính:

- Backbone: `torchvision.models.convnext_tiny` pretrained.
- Neck: FPN top-down kết hợp một nhánh PANet bottom-up.
- Head: decoupled detection heads, tách nhánh object/class và bbox regression.
- Target: anchor-free grid, `S = resolution // 16`; với `448x448` tạo grid `28x28`.
- Loss: focal loss cho objectness, weighted cross entropy cho class, CIoU + Smooth L1 cho bbox.
- Inference: confidence threshold, decode bbox về tọa độ ảnh gốc, class-wise NMS.

## Cấu Trúc Thư Mục

```text
<my_submission>/
├── models/
│   └── detector.py
├── utils/
│   ├── dataset.py
│   ├── loss.py
│   └── nms.py
├── train.py
├── predict.py
├── README.md
└── requirements.txt
```

Thư mục `public/` là dữ liệu được cung cấp trong đề bài. Khi nộp bài, không cần nộp lại `public/` nếu hệ thống chấm đã cung cấp dữ liệu riêng.

## Cài Đặt

```bash
pip install -r requirements.txt
```

Các thư viện chính gồm PyTorch, Torchvision, NumPy, OpenCV, Pillow, tqdm và Albumentations.

## Huấn Luyện

Lệnh huấn luyện bắt buộc theo đề:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

Có thể truyền thêm tham số, ví dụ:

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

Script sẽ lưu checkpoint tốt nhất theo validation mAP@0.5 tại:

```text
./models/best.pth
```

Ngoài ra, `latest.pth` lưu trạng thái gần nhất, và `best_no_aug.pth` lưu checkpoint tốt nhất trong giai đoạn fine-tuning cuối khi tắt augmentation mạnh.

## Suy Luận

Lệnh suy luận bắt buộc theo đề:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

Mặc định script sẽ đọc checkpoint:

```text
./models/best.pth
```

Có thể chỉ định checkpoint khác:

```bash
python predict.py \
  --image_dir ./public/val/images \
  --output predictions.json \
  --checkpoint ./models/best.pth
```

Định dạng output:

```json
[
  {
    "image_id": "img_7fd91a4c2e30.jpg",
    "boxes": [
      {
        "class": "person",
        "confidence": 0.91,
        "bbox": [48, 72, 210, 356]
      }
    ]
  }
]
```

Ảnh không phát hiện đối tượng vẫn được ghi với `"boxes": []`. Bbox là `[xmin, ymin, xmax, ymax]` theo tọa độ ảnh gốc.

## Đánh Giá

Có thể tự đánh giá trên validation set bằng script được cung cấp:

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth ./public/annotations/val.json \
  --predictions predictions.json \
  --output score.json
```

Metric chính là mAP@0.5.

## Ghi Chú Thiết Kế

- Dữ liệu được đọc từ JSON theo schema của đề bài và gom annotation theo từng ảnh.
- Augmentation gồm horizontal flip, random crop/resize, affine, thay đổi màu, noise/blur, coarse dropout, Mosaic và Mixup.
- Multi-scale training dùng các resolution `384`, `416`, `448`, `480`; validation và inference mặc định dùng `448`.
- Backbone pretrained chỉ đóng vai trò trích xuất đặc trưng. Các thành phần detection chính gồm FPN/PANet, head, loss, target generation, decode và NMS được tự cài đặt trong project.
