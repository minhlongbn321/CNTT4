# Hướng dẫn thực hiện

Đây là bản tham khảo để tự thực hành. Đọc file giải thích, tự chạy và đối chiếu kết quả trước khi trình bày; tuân thủ quy định lớp về công cụ hỗ trợ và nguồn tham khảo.

## Chạy trên máy cá nhân

Cài Python 3.11 hoặc 3.12. Giải nén và mở Terminal tại thư mục có app.py.

```text
python -m venv .venv
```

Windows kích hoạt bằng `.venv\Scripts\activate`; macOS/Linux dùng `source .venv/bin/activate`.

```text
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

Mở địa chỉ localhost được Terminal hiển thị. train.py lưu lại mô hình, ket_qua.json và vi_du_kiem_thu.csv. Giữ đúng phiên bản scikit-learn hoặc huấn luyện lại nếu đổi phiên bản.

## Kiểm tra

Dùng hai dòng trong vi_du_kiem_thu.csv để nhập các đầu vào vào app; đối chiếu Du_doan và Diem_mo_hinh. Không nhập Nhan_thuc_te, Du_doan và Diem_mo_hinh làm đặc trưng. Thử thiếu một chỉ số để kiểm tra thông báo nhập đủ, thử giá trị nghi vấn Glucose = 750 và HbA1C = 8025 để kiểm tra thông báo xác minh.

Huấn luyện và dự đoán đã được chạy kiểm tra. Giao diện mới được kiểm tra cú pháp, chưa chạy tương tác vì môi trường kiểm tra chưa có Streamlit. Cần chạy thử trên máy cá nhân trước khi nộp.

## Triển khai và nộp bài

Nếu cần link, đưa các tệp chương trình và mô hình lên kho mã được phép sử dụng rồi chọn app.py làm tệp chính trong Streamlit Community Cloud. Chỉ công bố dữ liệu khi được phép. Bộ này chưa được triển khai lên URL mới.

Đổi tên thư mục HoTenHV_KiemTra theo họ tên của người học. Phần giải thích và hướng dẫn đã lưu thành hai tệp riêng đúng yêu cầu đề. Nguồn tham khảo và giới hạn được ghi trong file giải thích.
