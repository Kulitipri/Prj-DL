README: Phân loại Mã độc bằng Hình ảnh sử dụng CNN
1. Giới thiệu & Cơ sở Lý thuyết
Dự án này áp dụng phương pháp học sâu (Deep Learning) để phân loại các họ mã độc (malware families) khác nhau. Thay vì phân tích hành vi động hoặc tĩnh truyền thống trên mã nguồn, phương pháp này chuyển đổi các file mã độc nhị phân (binary) thành hình ảnh và sử dụng Mạng nơ-ron tích chập (CNN) để nhận diện.

Cơ sở lý thuyết:
Biểu diễn Mã độc dưới dạng Hình ảnh: Một file thực thi (binary) có thể được đọc dưới dạng một chuỗi các số nguyên 8-bit (0-255). Chuỗi này có thể được sắp xếp thành một ma trận 2 chiều và hiển thị như một hình ảnh thang độ xám (grayscale image).

Đặc trưng hình ảnh (Texture Analysis): Các biến thể khác nhau thuộc cùng một họ mã độc thường chia sẻ cấu trúc mã lệnh và dữ liệu tương tự nhau. Khi chuyển sang hình ảnh, chúng tạo ra các "vân" (textures) và bố cục hình ảnh rất đặc trưng và giống nhau.

Mạng Nơ-ron Tích chập (CNN): CNN là thuật toán cực kỳ mạnh mẽ trong việc nhận diện mẫu và đặc trưng trong hình ảnh. Do đó, CNN có thể học các đặc trưng vân của từng họ mã độc để phân loại chúng với độ chính xác cao.

2. Dataset (Bộ dữ liệu)
Dự án sử dụng bộ dữ liệu Malimg Dataset.

Nguồn: Tải về tự động thông qua thư viện kagglehub từ nguồn manmandes/malimg.

Số lượng: Hơn 9,000 hình ảnh.

Phân loại: Bao gồm 25 họ mã độc khác nhau (Ví dụ: Adialer.C, Agent.FYI, Allaple.A, Wannacry, v.v.).

Đặc điểm: Dữ liệu bị mất cân bằng (imbalanced), một số lớp có rất nhiều mẫu (như Allaple.A) trong khi các lớp khác có ít hơn. Code xử lý vấn đề này bằng cách sử dụng Class Weights.

3. Yêu cầu Hệ thống (Prerequisites)
Để chạy dự án, bạn cần cài đặt môi trường Python với các thư viện sau:

Bash

pip install tensorflow pandas numpy matplotlib seaborn opencv-python scikit-learn kagglehub
Các thư viện chính:

tensorflow & keras: Xây dựng và huấn luyện mô hình CNN.

kagglehub: Tải dataset tự động.

sklearn: Chia tập dữ liệu, tính toán class weights và các chỉ số đánh giá.

matplotlib & seaborn: Vẽ biểu đồ (Confusion Matrix, ROC curves).

cv2 (OpenCV): Xử lý hình ảnh.

4. Cấu trúc Mô hình CNN
Mô hình được xây dựng dạng chuỗi (Sequential) với các lớp như sau:

Input Layer: Nhận ảnh kích thước (256, 256, 3).

Convolutional Block 1: Conv2D (32 filters) + MaxPooling2D.

Convolutional Block 2: Conv2D (64 filters) + MaxPooling2D.

Convolutional Block 3: Conv2D (128 filters) + MaxPooling2D.

Flatten Layer: Duỗi dữ liệu 2D thành vector 1D.

Dense Layer (Fully Connected): 128 nơ-ron, hàm kích hoạt ReLU.

Dropout: Tỷ lệ 0.5 để giảm thiểu Overfitting.

Output Layer: 25 nơ-ron (tương ứng 25 lớp), hàm kích hoạt Softmax để đưa ra xác suất phân loại.

Optimizer: Adam (learning rate = 0.001) Loss Function: sparse_categorical_crossentropy

5. Cách vận hành Code (Workflow)
Quy trình thực thi trong Notebook đi qua các bước sau:

Bước 1: Tải và Chuẩn bị Dữ liệu
Code tự động tải dataset bằng kagglehub.

Kiểm tra cấu trúc thư mục (train/val/test).

Gom tất cả hình ảnh lại và chia lại (Split) theo tỷ lệ cấu hình (ví dụ: Train 60%, Val 20%, Test 20%) để đảm bảo tính ngẫu nhiên.

Bước 2: Tiền xử lý (Preprocessing)
Sử dụng tf.data.Dataset để tạo pipeline nạp dữ liệu hiệu quả.

Resize: Tất cả ảnh được đưa về kích thước cố định (256, 256).

Class Weights: Tính toán trọng số cho từng lớp để mô hình chú trọng hơn vào các lớp hiếm (do dữ liệu mất cân bằng).

Bước 3: Huấn luyện Lần 1 (Initial Training)
Huấn luyện mô hình cơ bản trong khoảng 10 epochs.

Sử dụng callback ModelCheckpoint để lưu lại model tốt nhất dựa trên val_loss.

Bước 4: Tái Huấn luyện (Retraining) - Tùy chọn nâng cao
Để tăng độ chính xác, code thực hiện gộp tập Train và Validation ban đầu lại.

Chia tách một phần nhỏ (10%) để làm validation mới cho các callbacks.

Huấn luyện lại (Retrain) mô hình với dữ liệu gộp này trong khoảng 15 epochs.

Sử dụng ReduceLROnPlateau để giảm tốc độ học nếu loss không giảm, và EarlyStopping để dừng sớm nếu mô hình không cải thiện.

Bước 5: Đánh giá (Evaluation)
Sau khi huấn luyện, mô hình được kiểm thử trên tập Test set (dữ liệu chưa từng thấy):

Accuracy: Tính độ chính xác tổng thể (thường đạt >98%).

Confusion Matrix: Vẽ biểu đồ nhiệt (heatmap) để xem sự nhầm lẫn giữa các lớp.

Classification Report: Chi tiết Precision, Recall, F1-Score cho từng dòng mã độc.

ROC Curves: Vẽ đường cong ROC cho từng lớp để đánh giá khả năng phân loại.

Misclassified Images: Hiển thị trực quan các hình ảnh mà mô hình dự đoán sai để phân tích.

6. Kết quả (Dự kiến)
Dựa trên log chạy:

Mô hình có khả năng đạt độ chính xác rất cao (~98% - 99%).

Các lớp phổ biến như Adialer.C, Allaple.A thường được nhận diện chính xác 100%.

Một số lớp biến thể khó (như Swizzor.gen) có thể có tỷ lệ nhận diện thấp hơn đôi chút.
