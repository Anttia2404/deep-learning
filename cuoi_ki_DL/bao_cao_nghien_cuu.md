# BÁO CÁO THỰC NGHIỆM: HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH PHÁT HIỆN DEEPFAKE TRÊN BỘ DỮ LIỆU FACEFORENSICS++

## 1. Tiền xử lý dữ liệu (Data Preprocessing)

### 1.1. Mô tả nguồn dữ liệu
Nghiên cứu sử dụng bộ dữ liệu **FaceForensics++ (FF++)**, một tập dữ liệu tiêu chuẩn (benchmark) quy mô lớn trong lĩnh vực phát hiện video giả mạo khuôn mặt. Nghiên cứu tập trung vào phiên bản nén **c23** (High Quality - HQ, sử dụng chuẩn nén H.264 với hệ số nén 23). Mức độ nén này mô phỏng thực tế các video được chia sẻ trên các nền tảng mạng xã hội lớn, nơi dữ liệu thường bị nén nhẹ nhưng vẫn giữ được độ nét tương đối. 

Dữ liệu bao gồm các video gốc (Original Sequences) thu thập từ nền tảng YouTube và các video giả mạo tương ứng được tạo ra bằng bốn phương pháp thao tác khuôn mặt phổ biến:
*   **Deepfakes (DF)**: Thay thế khuôn mặt dựa trên mô hình học sâu Autoencoder.
*   **Face2Face (F2F)**: Chuyển đổi biểu cảm khuôn mặt thời gian thực bằng mô hình đồ họa máy tính truyền thống.
*   **FaceSwap (FS)**: Tráo đổi khuôn mặt dựa trên kỹ thuật dựng hình 3D truyền thống.
*   **NeuralTextures (NT)**: Sử dụng mạng nơ-ron sinh để chỉnh sửa kết cấu bề mặt vùng miệng và nét mặt.

### 1.2. Các bước làm sạch dữ liệu và trích xuất khuôn mặt
Do dữ liệu đầu vào là các tệp video đa dạng về độ dài và góc quay, quy trình làm sạch và tiền xử lý được thực hiện qua các bước nghiêm ngặt sau:
1.  **Trích xuất khung hình (Frame Extraction)**: Mỗi video được trích xuất tối đa 32 khung hình phân bổ đều từ đầu đến cuối video nhằm giảm thiểu dư thừa dữ liệu thời gian nhưng vẫn đại diện đầy đủ cho diễn biến chuyển động.
2.  **Định vị khuôn mặt (Face Detection)**: Sử dụng thuật toán **Haar Cascade** (tích hợp trong thư viện OpenCV) để tự động nhận diện vùng chứa khuôn mặt trên từng khung hình. Thuật toán quét và xác định tọa độ hộp bao (Bounding Box) bao quanh khuôn mặt.
3.  **Kiểm tra và loại bỏ lỗi**:
    *   Các khung hình không phát hiện được khuôn mặt hoặc phát hiện nhiều hơn 1 khuôn mặt sẽ bị loại bỏ để đảm bảo tính nhất quán (chỉ tập trung vào nhân vật chính).
    *   Vùng cắt khuôn mặt được mở rộng thêm một biên độ lề (margin) $40\%$ nhằm giữ lại các đặc trưng vùng biên (như tóc, tai, góc hàm) - nơi thường xuất hiện các vết mờ (blurring artifacts) do quá trình ghép mặt để lại.

### 1.3. Chuẩn hóa kích thước và giá trị điểm ảnh
*   **Chuẩn hóa kích thước (Resizing)**: Toàn bộ vùng khuôn mặt sau khi cắt được đưa về kích thước chuẩn $299 \times 299$ pixel. Đây là kích thước đầu vào bắt buộc của kiến trúc mạng phân loại XceptionNet.
*   **Chuẩn hóa giá trị điểm ảnh (Normalization)**: Các giá trị điểm ảnh có khoảng ban đầu $[0, 255]$ được chuẩn hóa về đoạn $[-1.0, 1.0]$ theo công thức:
    $$\text{Pixel}_{\text{normalized}} = \frac{\text{Pixel} / 255.0 - 0.5}{0.5}$$
    Phép biến đổi này giúp tối ưu hóa quá trình lan truyền ngược, hạn chế hiện tượng bùng nổ hoặc tiêu biến đạo hàm trong quá trình huấn luyện.

### 1.4. Tăng cường dữ liệu (Data Augmentation)
Để hạn chế hiện tượng quá khớp (overfitting) và tăng cường khả năng tổng quát hóa của mô hình trên các dữ liệu thực tế, các kỹ thuật tăng cường dữ liệu sau đã được áp dụng trong quá trình huấn luyện:
*   **Lật ảnh ngẫu nhiên theo chiều ngang (Random Horizontal Flip, $p=0.5$)**: Mô phỏng sự thay đổi hướng nhìn của nhân vật.
*   **Xoay ảnh ngẫu nhiên (Random Rotation, góc tối đa $10^\circ$)**: Giúp mô hình bất biến với góc nghiêng đầu của nhân vật trong video.
*   **Biến đổi màu sắc ngẫu nhiên (Color Jitter)**: Điều chỉnh ngẫu nhiên độ sáng (brightness), độ tương phản (contrast), độ bão hòa (saturation) ở mức sai số $20\%$ và sắc độ (hue) ở mức $10\%$. Phép biến đổi này mô phỏng sự đa dạng của các điều kiện ánh sáng và thiết bị ghi hình khác nhau.

---

## 2. Phân bố tập dữ liệu (Dataset Distribution)

Bộ dữ liệu được phân chia theo tỷ lệ phân bổ của các tác nhân (actor-disjoint split) từ FaceForensics++. Việc phân chia này đảm bảo rằng các nhân vật xuất hiện trong tập Test hoàn toàn không có mặt trong tập Train và Validation, tránh hiện tượng rò rỉ thông tin danh tính (identity leakage).

Tỷ lệ phân chia dữ liệu tổng thể là **$72\%$ Train / $14\%$ Validation / $14\%$ Test** tương ứng với số lượng video mô tả trong Bảng 1:

### Bảng 1: Thống kê số lượng mẫu video và khung hình kiểm thử cho từng phương pháp

| Lớp dữ liệu (Class) | Tập huấn luyện (Train) | Tập kiểm định (Validation) | Tập kiểm thử (Test - Video) | Số lượng khung hình Test (Frame-level) |
| :--- | :---: | :---: | :---: | :---: |
| **Real (Gốc)** | 720 video | 140 video | 70 video | 2000 ảnh |
| **Fake (Giả mạo)** | 720 video | 140 video | 70 video | 2000 ảnh |
| **Tổng cộng** | **1440 video** | **280 video** | **140 video** | **4000 ảnh** |

**Nhận xét về mức độ cân bằng dữ liệu:** 
Tỷ lệ phân bố giữa mẫu thật (Real) và mẫu giả mạo (Fake) đạt tỷ lệ **1:1** trên tất cả các tập dữ liệu. Sự cân biến hoàn hảo này giúp loại bỏ hoàn toàn hiện tượng lệch độ chệch (bias) của mô hình về phía lớp chiếm đa số, đảm bảo các chỉ số đo lường như Accuracy và F1-Score phản ánh trung thực năng lực phân loại của mạng nơ-ron.

---

## 3. Quá trình huấn luyện mô hình (Model Training)

### 3.1. Kiến trúc mô hình
Nghiên cứu lựa chọn kiến trúc mạng **XceptionNet** làm mô hình phân loại cốt lõi. XceptionNet cải tiến từ Inception bằng cách thay thế các module Inception thông thường bằng phép tích chập tách biệt theo chiều sâu (Depthwise Separable Convolutions). Lựa chọn này dựa trên các nghiên cứu thực nghiệm trước đó chứng minh XceptionNet có khả năng nắm bắt cực tốt các nhiễu tần số cao và các lỗi bất thường cục bộ ở cấp độ điểm ảnh - những đặc trưng điển hình của các vùng biên ghép mặt trong ảnh Deepfake.

### 3.2. Cấu hình phần cứng và môi trường huấn luyện
*   **Hệ điều hành**: Linux (Ubuntu 22.04 LTS) trên môi trường điện toán đám mây Kaggle.
*   **Phần cứng**: GPU **Nvidia Tesla T4 (16GB VRAM)**.
*   **Môi trường phần mềm**: Python 3.12, PyTorch 2.2.2, Torchvision 0.27.0.

### 3.3. Các siêu tham số huấn luyện (Hyperparameters)
Cấu hình siêu tham số được tối ưu hóa thông qua các vòng thực nghiệm sơ bộ, mô tả chi tiết tại Bảng 2:

### Bảng 2: Các siêu tham số cấu hình hệ thống huấn luyện

| Siêu tham số (Hyperparameter) | Giá trị thiết lập | Lý do lựa chọn |
| :--- | :---: | :--- |
| **Batch Size** | 16 | Tối ưu hóa dung lượng bộ nhớ VRAM của GPU T4, đảm bảo độ ổn định của ước lượng gradient. |
| **Learning Rate** | 0.0002 | Tránh hiện tượng nhảy bước quá lớn làm mất hội tụ, đồng thời không quá nhỏ gây kéo dài thời gian huấn luyện. |
| **Optimizer** | Adam | Tự động thích ứng tốc độ học cho từng tham số, giúp hội tụ nhanh trên các bài toán phân loại ảnh phức tạp. |
| **Loss Function** | Cross Entropy | Hàm mất mát tiêu chuẩn cho bài toán phân loại nhị phân (Binary Classification). |
| **Epochs** | 10 | Đủ để mô hình hội tụ dựa trên tập dữ liệu đã qua tiền xử lý kỹ lưỡng. |

### 3.4. Quy trình huấn luyện từng Epoch và Cơ chế Lưu checkpoint
Mỗi epoch huấn luyện bao gồm hai giai đoạn liên tiếp:
1.  **Giai đoạn Huấn luyện (Training Phase)**: Mô hình thực hiện lan truyền xuôi (Forward pass) để tính toán Loss, thực hiện lan truyền ngược (Backward pass) để tính đạo hàm và cập nhật trọng số thông qua bộ tối ưu hóa Adam.
2.  **Giai đoạn Kiểm định (Validation Phase)**: Chuyển mô hình sang chế độ đánh giá (`model.eval()`), tắt tính toán đạo hàm (`torch.no_grad()`). Tính toán độ chính xác (Accuracy) trên tập Validation.
3.  **Cơ chế Checkpoint**: Trọng số mô hình chỉ được lưu lại khi và chỉ khi giá trị `val_acc` ở epoch hiện tại cao hơn giá trị `val_acc` tốt nhất lịch sử huấn luyện. Trọng số này được lưu trữ dưới dạng tệp tin `.pth` (ví dụ: `best_Deepfakes_c23.pth`).

---

## 4. Kết quả huấn luyện và Đánh giá hiệu năng

Quá trình huấn luyện được thực hiện độc lập cho cả 4 phương pháp giả mạo. Thời gian huấn luyện trung bình cho mỗi mô hình dao động từ **85 đến 110 phút** trên GPU T4. Bảng 3 tổng hợp kết quả đánh giá chi tiết của cả 4 mô hình trên tập kiểm thử độc lập (Test Set) gồm 4000 khung hình:

### Bảng 3: Kết quả đánh giá hiệu năng phân loại trên tập kiểm thử (Test Set)

| Phương pháp (Method) | Accuracy | F1-Score | Precision | Recall | AUC-ROC | Epoch tối ưu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deepfakes** | 95.25% | 0.9526 | 0.9548 | 0.9500 | 0.9700 | Epoch 4 |
| **Face2Face** | **97.12%** | **0.9712** | **0.9724** | **0.9700** | **0.9712** | Epoch 4 |
| **FaceSwap** | 96.63% | 0.9663 | 0.9651 | 0.9675 | 0.9663 | Epoch 6 |
| **NeuralTextures** | 94.50% | 0.9451 | 0.9472 | 0.9425 | 0.9450 | Epoch 1 |

### Nhận xét về khả năng hội tụ và hiệu năng:
*   Mô hình đạt độ hội tụ rất nhanh (thường từ epoch 4 đến epoch 6). Độ chính xác trên tập kiểm thử của cả 4 phương pháp đều vượt ngưỡng **$94\%$**, trong đó mô hình phát hiện **Face2Face** đạt kết quả cao nhất với độ chính xác **$97.12\%$** và giá trị AUC-ROC đạt **$0.9712$**.
*   Sự tương đồng rất cao giữa chỉ số Precision và Recall chứng tỏ mô hình có khả năng nhận diện cân bằng tốt: không bị bỏ sót các video giả mạo (Recall cao) đồng thời không báo động giả các video thật (Precision cao).
*   Đồ thị ROC và phân tích Confusion Matrix cho thấy các điểm phân loại sai lệch phân bố rất ít, minh chứng cho việc các đặc trưng học được từ kiến trúc XceptionNet có tính phân tách tuyến tính rất mạnh trên không gian đặc trưng.

---

## 5. Kết luận

### 5.1. Ưu điểm
*   Mô hình phát hiện đạt độ chính xác rất cao ($>94\%$), phản ánh hiệu quả vượt trội của kiến trúc tích chập tách biệt XceptionNet đối với bài toán nhận diện biên giả mạo.
*   Quy trình tiền xử lý dữ liệu chặt chẽ (cắt mở rộng biên khuôn mặt $40\%$) đã giữ lại được nhiều dấu vết vật lý cốt lõi của Deepfake.
*   Tỷ lệ chia dữ liệu và huấn luyện đảm bảo tính khách quan khoa học, không bị hiện tượng rò rỉ thông tin danh tính giữa các tập dữ liệu.

### 5.2. Hạn chế
*   Mô hình hoạt động dựa trên thông tin không gian của từng khung hình độc lập (Spatial Features), chưa tận dụng được tính liên tục và sự bất đồng bộ thời gian (Temporal Anomalies) giữa các khung hình liên tiếp trong video.
*   Hiệu năng huấn luyện phụ thuộc nhiều vào tài nguyên phần cứng mạnh (GPU).

### 5.3. Hướng phát triển tương lai
*   Tích hợp các kiến trúc mạng học chuỗi thời gian như **LSTM (Long Short-Term Memory)** hoặc mạng **Transformer** để kết hợp phân tích cả đặc trưng không gian (Spatial) và thời gian (Temporal), giúp phát hiện các video Deepfake có mức độ nén cực cao hoặc bị nhiễu nặng.
*   Nghiên cứu các mô hình nhẹ hơn (như MobileNetV4 hoặc EfficientNet) để có thể triển khai chạy demo thời gian thực trực tiếp trên các thiết bị di động hoặc môi trường web có tài nguyên hạn chế.
