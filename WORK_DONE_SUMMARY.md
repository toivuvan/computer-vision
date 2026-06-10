# Báo Cáo Quá Trình Thực Hiện

Tài liệu này tóm tắt những phần đã triển khai trong project phát hiện đối tượng anchor-free. Nội dung được viết theo đúng code hiện tại: mô hình sử dụng backbone `ConvNeXt-Tiny` pretrained ImageNet, còn pipeline detection phía sau được tự cài đặt bằng PyTorch.

## 1. Mục Tiêu

Project xây dựng một mô hình phát hiện đối tượng cho 5 lớp:

- `person`
- `car`
- `dog`
- `cat`
- `chair`

Mô hình cần dự đoán đồng thời bbox, class và confidence/objectness. Thiết kế hiện tại đi theo hướng anchor-free grid detector kiểu YOLO nhỏ, nhưng không dùng detector hoàn chỉnh có sẵn. Backbone pretrained được dùng làm mạng trích xuất đặc trưng, phù hợp với phần đề bài cho phép sử dụng mạng trích xuất đặc trưng đã huấn luyện trước nếu được giảng viên cho phép.

## 2. Dữ Liệu Và Tiền Xử Lý

File annotation nằm trong `public/annotations/train.json` và `public/annotations/val.json`, gồm:

- `classes`: danh sách 5 lớp.
- `images`: `id`, `file_name`, `width`, `height`.
- `annotations`: `image_id`, `class`, `bbox`.

Trong `utils/dataset.py`, dữ liệu được gom theo từng ảnh. Mỗi sample gồm ảnh, danh sách bbox, danh sách label và metadata ảnh gốc. Bbox dùng định dạng `[xmin, ymin, xmax, ymax]` theo tọa độ ảnh gốc.

Khi validation và inference, ảnh được resize về resolution cấu hình, normalize theo ImageNet mean/std, rồi chuyển sang tensor.

## 3. Augmentation

Khi training, dataset dùng Albumentations và một số augmentation tự cài đặt:

- Horizontal flip.
- Random resized crop.
- Resize về resolution đang train.
- Affine transform.
- CLAHE, brightness/contrast, hue/saturation/value.
- Gaussian noise, motion blur.
- Coarse dropout.
- Mosaic augmentation.
- Mixup augmentation.

Mosaic ghép 4 ảnh vào một canvas có kích thước bằng resolution hiện tại, đồng thời scale bbox tương ứng. Mixup trộn hai ảnh đã qua transform và gộp bbox/label của hai ảnh.

Trong các epoch cuối, `disable_strong_augmentations()` tắt Mosaic, Mixup và các augmentation mạnh để fine-tune trên phân phối ảnh gần với validation/inference hơn.

## 4. Target Grid Anchor-Free

Mỗi ảnh được chuyển thành target tensor dạng:

```text
10 x S x S
```

Trong đó `S = resolution // 16`. Với resolution `448`, grid là `28 x 28`.

Ý nghĩa các channel:

- Channel 0: objectness.
- Channel 1-5: one-hot class.
- Channel 6-7: offset tâm bbox trong cell.
- Channel 8-9: width và height chuẩn hóa theo kích thước ảnh sau transform.

Khi nhiều object rơi vào cùng một cell, code giữ object có diện tích nhỏ hơn để ưu tiên vật thể nhỏ và tránh ghi đè không kiểm soát.

## 5. Kiến Trúc Mô Hình

Mô hình nằm trong `models/detector.py`, class `ResNetYOLO`. Tên class giữ lại từ phiên bản ban đầu, nhưng backbone thực tế hiện tại là `ConvNeXt-Tiny`, không phải ResNet-50.

### 5.1. Backbone ConvNeXt-Tiny Pretrained

Backbone dùng:

```python
torchvision.models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
```

Các feature map được lấy từ các stage của `backbone.features`:

- `c2`: stride 8, 192 channels.
- `c3`: stride 16, 384 channels.
- `c4`: stride 32, 768 channels.

Backbone pretrained giúp tận dụng đặc trưng ảnh tổng quát từ ImageNet, trong khi các thành phần detection được train cho bài toán 5 lớp của dataset.

### 5.2. FPN Và PANet

Các feature map `c2`, `c3`, `c4` được chiếu về 256 channels bằng convolution `1x1`.

FPN top-down:

```text
p4 = proj(c4)
p3 = proj(c3) + upsample(p4)
p2 = proj(c2) + upsample(p3)
```

PANet bottom-up:

```text
n3 = p3 + ConvStride2(p2)
```

Feature dùng cho head là concat giữa `n3` và `p4` đã upsample về stride 16, tạo tensor 512 channels.

### 5.3. Decoupled Detection Heads

Mô hình tách hai nhánh dự đoán:

- `cls_head`: output 6 channels gồm objectness và 5 class logits.
- `reg_head`: output 4 channels gồm tọa độ bbox dạng raw logits.

Hai output được concat thành tensor cuối:

```text
batch x 10 x S x S
```

## 6. Loss

Loss nằm trong `utils/loss.py`, gồm ba phần chính:

- Objectness loss: focal loss với logits để xử lý mất cân bằng foreground/background.
- Classification loss: cross entropy có class weights và label smoothing.
- Box loss: CIoU loss kết hợp Smooth L1 trên tọa độ sau sigmoid.

Tổng loss là tổng có trọng số của objectness, class và bbox loss.

## 7. Training

`train.py` triển khai toàn bộ vòng lặp huấn luyện:

- Tạo train/validation dataset và dataloader.
- Khởi tạo model với `pretrained=True`.
- Dùng class weights theo thống kê train set.
- Dùng AdamW optimizer.
- Dùng differential learning rate: backbone học chậm hơn head.
- Warm-up learning rate trong 3 epoch đầu.
- CosineAnnealingLR sau mỗi epoch.
- AMP mixed precision khi có CUDA.
- Gradient clipping.
- Đánh giá validation loss và mAP@0.5 sau mỗi epoch.
- Lưu `best.pth`, `latest.pth`, và `best_no_aug.pth`.

Multi-scale training chọn resolution trong:

```text
384, 416, 448, 480
```

Trong no-augmentation phase cuối, resolution được cố định về `448`.

## 8. Inference Và NMS

`predict.py` đọc ảnh từ `--image_dir`, resize/normalize giống validation, chạy model rồi gọi các hàm trong `utils/nms.py`.

Decode prediction:

- Objectness qua sigmoid.
- Class logits qua softmax.
- Bbox raw logits qua sigmoid.
- Tọa độ cell được chuyển về bbox trên ảnh gốc.

Sau decode, class-wise NMS loại các bbox trùng lặp theo IoU threshold. Output được lưu dưới dạng `predictions.json` đúng schema của đề bài:

```json
[
  {
    "image_id": "image_name.jpg",
    "boxes": []
  }
]
```

## 9. Đánh Giá

Trong training, validation dùng logic gần với script chấm:

- Decode prediction.
- NMS theo từng class.
- Match prediction với ground truth theo IoU >= 0.5.
- Tính precision/recall.
- Tính AP từng class.
- Lấy trung bình thành mAP@0.5.

Ngoài ra, có thể dùng trực tiếp script của đề:

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth ./public/annotations/val.json \
  --predictions predictions.json \
  --output score.json
```

## 10. Những Thành Phần Đã Hoàn Thành

- Dataset loader theo schema của đề bài.
- Tiền xử lý ảnh và bbox.
- Augmentation gồm flip, crop, affine, color/noise/blur, dropout, Mosaic, Mixup.
- Target generation dạng anchor-free grid.
- Backbone ConvNeXt-Tiny pretrained.
- FPN + PANet tự cài đặt.
- Decoupled detection heads tự cài đặt.
- Focal loss, weighted CE, CIoU và Smooth L1.
- Training loop với AMP, AdamW, warm-up, cosine scheduler và checkpoint.
- Decode prediction về tọa độ ảnh gốc.
- Class-wise NMS.
- Script predict sinh `predictions.json` đúng định dạng yêu cầu.

## 11. Kết Luận

Project đáp ứng hướng làm detector tự cài đặt theo đề bài: không dùng framework detector hoàn chỉnh, tự xây dựng pipeline dữ liệu, head, loss, training, inference và NMS. Backbone ConvNeXt-Tiny pretrained được sử dụng như mạng trích xuất đặc trưng, còn các phần quyết định hành vi detection được triển khai trực tiếp trong source code của project.
