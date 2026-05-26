# Báo Cáo Quá Trình Thực Hiện

Tài liệu này tóm tắt chi tiết những gì đã được xây dựng trong bài toán phát hiện đối tượng anchor-free của project này, từ khâu đọc dữ liệu, augmentation, tạo nhãn, kiến trúc mô hình, hàm loss, suy luận, NMS cho đến huấn luyện và đánh giá.

## 1. Mục tiêu của bài toán

Mục tiêu của project là xây dựng một mô hình phát hiện đối tượng từ đầu, không dùng anchor box, để dự đoán đồng thời:

- Vị trí của đối tượng trong ảnh.
- Nhãn lớp của đối tượng.
- Độ tin cậy objectness cho từng ô lưới.

Mô hình được thiết kế theo hướng nhẹ hơn các detector hai giai đoạn, nhưng vẫn đủ mạnh nhờ backbone ResNet-50, FPN và chiến lược huấn luyện có tăng cường dữ liệu.

## 2. Luồng dữ liệu đầu vào

### 2.1. Đọc dữ liệu annotation

File dữ liệu huấn luyện và validation được lưu theo định dạng JSON trong thư mục `public/annotations/`. Trong đó có hai phần chính:

- `images`: thông tin ảnh, gồm `id`, `file_name`, `width`, `height`.
- `annotations`: danh sách bounding box và nhãn lớp tương ứng với từng ảnh.

Trong dataset, mỗi annotation được ánh xạ về một trong 5 lớp cố định:

- `person`
- `car`
- `dog`
- `cat`
- `chair`

### 2.2. Tạo danh sách mẫu

Trong `DetectionDataset`, dữ liệu được gom theo từng ảnh. Với mỗi ảnh, chương trình lấy toàn bộ box và label tương ứng, rồi lưu thành một example gồm:

- id ảnh
- tên file ảnh
- width/height gốc
- danh sách bounding box
- danh sách label

Mỗi ảnh sau đó được load trực tiếp từ thư mục ảnh tương ứng bằng tên file đã có trong JSON.

## 3. Augmentation và tiền xử lý ảnh

### 3.1. Augmentation khi train

Khi `is_train=True`, dataset áp dụng một chuỗi tăng cường dữ liệu bằng Albumentations để tăng khả năng tổng quát hóa của mô hình. Các phép biến đổi chính gồm:

- Horizontal Flip
- RandomResizedCrop
- Resize về đúng kích thước mong muốn
- ShiftScaleRotate
- ColorJitter
- CoarseDropout để giả lập vùng bị che khuất
- Normalize theo ImageNet mean/std
- Chuyển sang tensor bằng `ToTensorV2`

Mục tiêu của augmentation là làm mô hình bền hơn trước thay đổi về vị trí, tỉ lệ, góc quay, ánh sáng và hiện tượng che khuất một phần.

### 3.2. Tiền xử lý khi validation và inference

Khi không train, ảnh chỉ được resize và normalize, không áp dụng augmentation ngẫu nhiên. Điều này giúp kết quả đánh giá và suy luận ổn định, nhất quán.

## 4. Chuyển bbox thành target dạng lưới

### 4.1. Cấu trúc target

Mỗi ảnh được chuyển thành một target tensor có dạng `10 x S x S`, với `S = resolution // 16`.

Ý nghĩa 10 kênh gồm:

- Kênh 0: objectness
- Kênh 1 đến 5: one-hot cho 5 lớp
- Kênh 6 đến 9: thông tin bbox gồm tâm x, tâm y, width, height

### 4.2. Gán đối tượng vào ô lưới

Mỗi bounding box được quy đổi sang tọa độ chuẩn hóa trên ảnh sau augmentation. Sau đó tâm box xác định ô lưới tương ứng. Ô đó sẽ được gán:

- objectness = 1
- class one-hot
- offset tâm trong cell
- width/height chuẩn hóa

### 4.3. Xử lý khi nhiều box rơi vào cùng một cell

Nếu có nhiều object cùng rơi vào một ô lưới, hệ thống chọn object có diện tích nhỏ hơn để ưu tiên các object nhỏ và tránh ghi đè không kiểm soát. Đây là một lựa chọn phù hợp cho bài toán anchor-free trên grid cố định.

## 5. Kiến trúc mô hình

### 5.1. Backbone ResNet-50

Mô hình dùng ResNet-50 làm backbone trích xuất đặc trưng. Các tầng chính của ResNet được giữ lại để học đặc trưng từ ảnh đầu vào.

### 5.2. FPN fusion

Đặc trưng từ `layer3` và `layer4` được đưa qua các lớp chiếu kênh 1x1, sau đó upsample và ghép lại theo chiều kênh. Cách này giúp kết hợp:

- thông tin ngữ nghĩa mạnh từ layer sâu
- thông tin không gian chi tiết hơn từ layer nông hơn

### 5.3. Detection head

Sau khi fusion, feature map đi qua head convolution để dự đoán đầu ra `10 x S x S`.

Đầu ra này tương ứng với:

- objectness
- class logits
- bbox regression

Như vậy mô hình hoạt động theo hướng anchor-free, dự đoán trực tiếp trên grid thay vì sinh anchor boxes.

## 6. Hàm loss

### 6.1. Objectness loss

Phần objectness dùng focal loss để giảm ảnh hưởng của các cell nền. Đây là điểm rất quan trọng vì trên grid số lượng ô không có vật thể thường lớn hơn rất nhiều so với ô có vật thể.

Focal loss giúp mô hình tập trung hơn vào các mẫu khó thay vì bị chi phối bởi quá nhiều background dễ.

### 6.2. Class loss

Phần phân loại lớp sử dụng CrossEntropyLoss có trọng số lớp. Việc gán class weights giúp xử lý mất cân bằng dữ liệu giữa các lớp. Trong code, trọng số đã được tính theo tần suất xuất hiện của từng lớp và chuẩn hóa lại để trung bình bằng 1.

### 6.3. Box loss

Phần hồi quy bbox kết hợp hai thành phần:

- CIoU loss: tối ưu độ chồng lắp, khoảng cách tâm và tỉ lệ khung hình.
- Smooth L1 loss trên tọa độ sau sigmoid: giúp quá trình học ổn định hơn.

Đây là cách ghép loss khá hợp lý vì CIoU tối ưu trực tiếp chất lượng hình học của box, còn Smooth L1 hỗ trợ giảm dao động khi học tọa độ.

### 6.4. Tổng loss

Tổng loss cuối cùng là tổng có trọng số của:

- objectness loss
- class loss
- box loss

Việc chia theo batch và số object positive giúp loss không bị lệch quá mạnh theo số lượng object của từng batch.

## 7. Huấn luyện mô hình

### 7.1. Khởi tạo dữ liệu

Train và validation được tạo thành hai dataset riêng, sau đó đưa vào DataLoader. Bài toán dùng custom `collate_fn` để tránh PyTorch stack sai các metadata có kích thước thay đổi.

### 7.2. Multi-scale training

Trong quá trình train, kích thước ảnh có thể thay đổi giữa các epoch theo danh sách:

- 384
- 416
- 448
- 480

Khi đổi resolution, grid size cũng thay đổi theo. Mục tiêu là giúp mô hình học được tính bền vững với nhiều mức scale khác nhau.

### 7.3. Differential learning rate

Backbone ResNet-50 được học với learning rate nhỏ hơn head, để tránh phá hỏng đặc trưng đã học sẵn từ pretrain. Phần head được cập nhật mạnh hơn vì cần thích nghi nhanh với bài toán detection hiện tại.

### 7.4. Warm-up và scheduler

Trong 3 epoch đầu, learning rate được warm-up tuyến tính để ổn định quá trình fine-tuning. Sau đó dùng CosineAnnealingLR để giảm dần learning rate theo dạng cosine, giúp hội tụ mượt hơn.

### 7.5. Mixed Precision và gradient clipping

Huấn luyện dùng AMP để tăng tốc và giảm bộ nhớ GPU. Trước khi optimizer step, gradient được unscale và clip norm để tránh gradient exploding.

### 7.6. Validation và lưu checkpoint

Sau mỗi epoch, mô hình được đánh giá bằng mAP@0.5 trên validation set. Nếu kết quả tốt nhất hiện tại được cải thiện, checkpoint `best.pth` sẽ được lưu lại. Ngoài ra, `latest.pth` luôn được ghi đè để lưu trạng thái gần nhất.

## 8. Suy luận và hậu xử lý

### 8.1. Forward khi predict

Trong `predict.py`, ảnh đầu vào được resize về 448x448, normalize rồi đưa qua model để thu output theo grid.

### 8.2. Decode prediction

Output raw của model được giải mã bằng hàm `decode_predictions`:

- objectness được qua sigmoid
- class logits được qua softmax
- bbox offsets/sizes được qua sigmoid

Sau đó box được đổi từ tọa độ lưới sang tọa độ gốc của ảnh.

### 8.3. NMS

Sau khi decode, các box trùng lặp được lọc bằng class-wise NMS. Nghĩa là box của từng lớp được xử lý riêng, box nào có IoU quá cao với box tốt hơn sẽ bị loại bỏ.

Điều này giúp giảm số dự đoán dư thừa và làm kết quả cuối cùng gọn, chính xác hơn.

## 9. Đánh giá mô hình

### 9.1. Tính mAP@0.5

Trong quá trình validation, mô hình được đánh giá theo cách gần với file chấm điểm cuối cùng. Các bước chính gồm:

- decode output của model
- áp dụng NMS
- so khớp prediction với ground truth theo IoU >= 0.5
- tính precision/recall cho từng lớp
- tính AP cho từng lớp
- lấy trung bình để ra mAP

### 9.2. Ý nghĩa của mAP@0.5

mAP@0.5 là thước đo quan trọng để biết mô hình không chỉ dự đoán đúng lớp, mà còn phải định vị box đủ tốt. Đây là chỉ số tổng hợp phản ánh chất lượng detection ở mức thực tế.

## 10. Kết quả đạt được trong bài

Từ toàn bộ pipeline trên, project đã hoàn thiện được một hệ thống object detection hoàn chỉnh gồm:

- đọc và chuẩn hóa dữ liệu annotation
- augmentation dữ liệu khi train
- mã hóa bbox thành lưới anchor-free
- mô hình ResNet-50 + FPN + head detection
- loss kết hợp focal loss, CE loss, CIoU và Smooth L1
- training có multi-scale, warm-up, AMP và differential LR
- suy luận với decode prediction và NMS
- đánh giá bằng mAP@0.5

Nói ngắn gọn, đây là một quy trình detection end-to-end từ dữ liệu thô đến prediction cuối cùng.

## 11. Kết luận

Điểm mạnh của bài này nằm ở việc xây dựng được toàn bộ pipeline detection từ đầu, không phụ thuộc vào framework detector có sẵn. Mỗi thành phần đều phục vụ trực tiếp cho mục tiêu cuối cùng:

- dataset xử lý dữ liệu và augmentation
- model học đặc trưng và dự đoán theo grid
- loss tối ưu đồng thời objectness, class và bbox
- NMS làm sạch đầu ra
- evaluation đo chất lượng thực tế

Đây là nền tảng tốt để tiếp tục cải tiến về sau, ví dụ tinh chỉnh loss, cải tiến head, thay đổi decoder hoặc thử các chiến lược augmentation mạnh hơn.