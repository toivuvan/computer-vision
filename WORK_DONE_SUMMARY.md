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

Khi `is_train=True`, dataset áp dụng một chuỗi tăng cường dữ liệu mở rộng bằng Albumentations để tăng khả năng tổng quát hóa và độ bền bỉ của mô hình. Các phép biến đổi chính gồm:

- **Horizontal Flip**: Lật ảnh theo chiều ngang ngẫu nhiên.
- **RandomResizedCrop**: Cắt ảnh ngẫu nhiên và resize về kích thước lưới, giúp mô hình nhận diện đối tượng ở nhiều quy mô khác nhau.
- **Resize**: Đảm bảo kích thước ảnh đầu ra trùng khớp chính xác với độ phân giải huấn luyện mong muốn.
- **Affine**: Tăng cường biến dạng hình học bằng cách áp dụng các bộ biến đổi xoay ngẫu nhiên ($\pm 15^\circ$), dịch chuyển ($\pm 5\%$), co giãn ($0.9 - 1.1$), và cắt nghiêng ($\pm 10^\circ$). Sử dụng `border_mode=0` để tương thích hoàn hảo với các phiên bản Albumentations mới.
- **CLAHE**: Cân bằng biểu đồ độ tương phản thích ứng cục bộ giúp nâng cao các chi tiết khó nhận dạng của ảnh.
- **RandomBrightnessContrast**: Thay đổi ngẫu nhiên độ sáng và độ phản chiếu của hình ảnh.
- **HueSaturationValue**: Biến đổi ngẫu nhiên các giá trị sắc thái (hue), độ bão hòa màu (saturation) và cường độ sáng (value).
- **GaussNoise**: Thêm nhiễu Gaussian ngẫu nhiên để mô hình chống nhiễu tốt hơn trong môi trường thực tế.
- **MotionBlur**: Làm mờ chuyển động ngẫu nhiên với kích thước kernel tối đa bằng 3 để mô phỏng ảnh chụp khi đối tượng hoặc camera di chuyển.
- **CoarseDropout (Cutout)**: Giả lập các vùng bị che khuất bằng cách tạo ngẫu nhiên từ 1 đến 8 lỗ trống có kích thước từ $8 \times 8$ đến $24 \times 24$ pixel.
- **Normalize**: Chuẩn hóa pixel theo trị trung bình (mean) và độ lệch chuẩn (std) của bộ dữ liệu ImageNet.
- **ToTensorV2**: Chuyển đổi dữ liệu ảnh và bounding box sang tensor để truyền trực tiếp vào GPU.

Mục tiêu của augmentation là làm mô hình bền vững tuyệt đối trước các thay đổi phức tạp về vị trí, góc xoay, ánh sáng, nhiễu và hiện tượng che khuất một phần trong quá trình huấn luyện thực tế.

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

### 5.1. Backbone ConvNeXt-Tiny (thay cho ResNet-50)

Mô hình sử dụng `ConvNeXt-Tiny` làm backbone trích xuất đặc trưng, thay thế cho ResNet-50 ban đầu. Các đặc trưng đa cấp độ được trích xuất trực tiếp từ các Stage con của backbone:

- **Stage 1 (`c2`)**: Stride 8, 192 channels — mang thông tin không gian, đường biên và chi tiết định vị rất sắc nét của đối tượng.
- **Stage 2 (`c3`)**: Stride 16, 384 channels — sự cân bằng tối ưu giữa thông tin không gian và thông tin ngữ nghĩa.
- **Stage 3 (`c4`)**: Stride 32, 768 channels — mang thông tin ngữ nghĩa bậc cao cực kỳ mạnh mẽ cho việc phân loại.

Việc chuyển đổi sang backbone ConvNeXt mang lại khả năng biểu diễn không gian vượt trội và nâng cao đáng kể độ chính xác nhận diện tổng thể.

### 5.2. FPN & PANet Fusion (Path Aggregation Network)

Mô hình đã được tích hợp thêm cấu trúc **PANet (Path Aggregation Network)** song hành cùng **FPN (Feature Pyramid Network)** giúp kết hợp tối đa các luồng thông tin:

1. **Top-Down Pathway (FPN)**:
   * Các đặc trưng `c4`, `c3`, `c2` được chiếu kênh qua lớp tích chập $1 \times 1$ về cùng 256 kênh để thu được $P_4, P_3, P_2$.
   * Đặc trưng cấp cao được upsample tuyến tính 2 lần (Bilinear Upsample) và cộng gộp (element-wise add) với đặc trưng cấp thấp hơn:
     $$P_4 = \text{proj}(c_4)$$
     $$P_3 = \text{proj}(c_3) + \text{Upsample}(P_4)$$
     $$P_2 = \text{proj}(c_2) + \text{Upsample}(P_3)$$
   * Đường dẫn từ trên xuống này giúp mang thông tin ngữ nghĩa mạnh mẽ bổ trợ cho các tầng thấp.

2. **Bottom-Up Pathway (PANet)**:
   * Nhằm tránh hao hụt chi tiết định vị từ các tầng cực nông, PANet bổ sung một đường dẫn ngược từ dưới lên:
     $$N_2 = P_2$$
     $$N_3 = P_3 + \text{SiLU}(\text{BatchNorm}(\text{Conv}_{3\times3,\,stride\,2}(N_2)))$$
   * Việc sử dụng tích chập $3 \times 3$ với Stride 2 giúp nén đặc trưng định vị sắc nét ở mức $N_2$ (stride 8) truyền trực tiếp lên mức $N_3$ (stride 16) chỉ qua 1 kết nối cực ngắn.

3. **Feature Fusion**:
   * Đặc trưng $N_3$ (tích hợp không gian Bottom-Up) và đặc trưng $P_4\_upsampled$ (stride 16 upsample từ Stride 32 FPN) được ghép nối dọc theo chiều kênh (`torch.cat`), tạo ra đặc trưng hợp nhất có kích thước `512 x S x S` cung cấp cho các nhánh dự đoán.

### 5.3. Decoupled detection heads (Nhánh dự đoán tách biệt)

Thay vì sử dụng một detection head chung cho tất cả các nhiệm vụ, mô hình đã được cải tiến sử dụng **Decoupled Heads** (hai nhánh tách biệt hoàn toàn để dự đoán phân loại và hồi quy tọa độ). Sau khi FPN fusion tạo ra feature map có kích thước `512 x S x S`, luồng dữ liệu sẽ được chia làm hai nhánh song song:

1. **Classification & Objectness Branch (`self.cls_head`)**:
   - Nhận đầu vào 512 kênh, đi qua chuỗi tích chập: `Conv2d(512, 256) -> BatchNorm -> SiLU -> Dropout(0.3) -> Conv2d(256, 128) -> BatchNorm -> SiLU -> Dropout(0.3) -> Conv2d(128, 6)`.
   - Đầu ra gồm **6 kênh**: 1 kênh độ tin cậy vật thể (`objectness`) và 5 kênh cho xác suất các lớp đối tượng (`class probabilities`).
   
2. **Regression Branch (`self.reg_head`)**:
   - Nhận đầu vào 512 kênh, đi qua chuỗi tích chập độc lập tương tự: `Conv2d(512, 256) -> BatchNorm -> SiLU -> Dropout(0.3) -> Conv2d(256, 128) -> BatchNorm -> SiLU -> Dropout(0.3) -> Conv2d(128, 4)`.
   - Đầu ra gồm **4 kênh**: tọa độ bounding box (`x, y, w, h`).

Đầu ra từ hai nhánh này sau đó được ghép lại (concatenate) theo chiều kênh (`torch.cat([cls_out, reg_out], dim=1)`) để tạo ra tensor dự đoán cuối cùng dạng `10 x S x S` tương thích với cấu trúc target và luồng xử lý loss/suy luận phía sau.

Việc thiết kế Decoupled Head giúp giảm xung đột đặc trưng giữa hai nhiệm vụ phân loại (cần các đặc trưng bất biến với các phép xoay/tịnh tiến) và định vị (cần đặc trưng nhạy cảm với không gian/tọa độ), từ đó cải thiện đáng kể tốc độ hội tụ và độ chính xác mAP.

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

Backbone được học với learning rate nhỏ hơn head, để tránh phá hỏng đặc trưng đã học sẵn từ pretrain. Do chuyển sang `ConvNeXt-Tiny`, code hiện phân nhóm tham số bằng cách kiểm tra tên tham số chứa `backbone_features` để xác định nhóm backbone (lr nhỏ hơn). Phần head vẫn được cập nhật mạnh hơn để thích nghi nhanh với bài toán detection hiện tại.

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
- mô hình ConvNeXt-Tiny + FPN + PANet + Decoupled Detection Heads (đã nâng cấp kiến trúc mạng định vị sắc nét đa chiều)
- loss nâng cao kết hợp focal loss (cho objectness), CE loss có trọng số phân bổ lớp nghịch đảo (cho phân loại lớp), CIoU loss và Smooth L1 loss (cho hồi quy hộp)
- training tối ưu hóa bằng multi-scale, warm-up, AMP (Mixed Precision) và differential LR (tốc độ học phân biệt cho backbone và head)
- suy luận hiệu quả với decode prediction trên grid độ phân giải cao kết hợp class-wise NMS
- đánh giá hiệu năng chính xác bằng mAP@0.5

Nói ngắn gọn, đây là một quy trình detection end-to-end từ dữ liệu thô đến prediction cuối cùng.

## 11. Kết luận

Điểm mạnh của bài này nằm ở việc xây dựng được toàn bộ pipeline detection từ đầu, không phụ thuộc vào framework detector có sẵn. Mỗi thành phần đều phục vụ trực tiếp cho mục tiêu cuối cùng:

- dataset xử lý dữ liệu và augmentation
- model học đặc trưng và dự đoán theo grid
- loss tối ưu đồng thời objectness, class và bbox
- NMS làm sạch đầu ra
- evaluation đo chất lượng thực tế

Đây là nền tảng tốt để tiếp tục cải tiến về sau, ví dụ tinh chỉnh loss, cải tiến head, thay đổi decoder hoặc thử các chiến lược augmentation mạnh hơn.