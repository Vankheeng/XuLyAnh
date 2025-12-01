# Nhận dạng chữ viết tay và hình dạng đơn giản bằng mạng nơ-ron nhân tạo

### Đề tài: Xây dựng ứng dụng Web nhận diện chữ viết và các hình dạng cơ bản sử dụng CNN

#### 1. Mục tiêu đề tài
Xây dựng một **ứng dụng web tương tác** cho phép người dùng:
- Vẽ chữ viết tay bằng chuột hoặc tải ảnh lên → hệ thống tự động nhận diện chữ cái.
- Vẽ hoặc tải lên hình tròn, hình vuông, hình tam giác → hệ thống dự đoán hình dạng.

Đề tài kết hợp hai bài toán kinh điển trong Xử lý ảnh và Thị giác máy tính:
- **Optical Character Recognition (OCR)** cho chữ viết tay
- **Image Classification** cho hình dạng đơn giản

#### 2. Ý nghĩa thực tiễn & khoa học
- Chứng minh khả năng triển khai thực tế (deployment) các mô hình Deep Learning chỉ bằng Python + Streamlits.
- Làm nền tảng để mở rộng sang các bài toán phức tạp hơn: nhận diện chữ tiếng Việt, biển số xe, chữ trên biên lai, v.v.

#### 3. Công nghệ & kiến trúc chính
| Thành phần              | Công nghệ sử dụng                          | Mô hình nổi bật          |
|-------------------------|---------------------------------------------|--------------------------|
| Mạng nhận diện chữ      | CRNN (CNN + RNN) + CTC Loss                 | CRNN                     |
| Mạng nhận diện hình     | Convolutional Neural Network (CNN)          | Custom CNN               |
| Giao diện người dùng    | Streamlit                                   | Web app tương tác        |
| Huấn luyện              | Kaggle Notebook + GPU miễn phí              | TensorFlow/Keras         |
| Triển khai              | Chạy local hoặc deploy online (Streamlit Cloud) | 100% Python           |

#### 4. Kết quả đạt được
- Model chữ viết tay: **~87% độ chính xác ký tự**.
- Model hình dạng: **> 99% độ chính xác**.
- Ứng dụng web giao diện thân thiện, hỗ trợ vẽ tay và tải ảnh.

#### 5. Thành viên thực hiện
| Họ tên                | Mã sinh viên       |
|-----------------------|--------------------|
|Trần Mai Hương         | B22DCCN424         |
|Nguyễn Thị Khánh Vân   | B22DCCN892         |
- Lớp: D22CNPM02
- Môn học: Xử lý ảnh 

---
