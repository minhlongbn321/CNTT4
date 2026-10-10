# Phân tích và câu trả lời tham khảo môn Trí tuệ nhân tạo

Tài liệu giải thích cách giải đề xây dựng phần mềm học máy dự đoán nguy cơ tiểu đường, dựa trên các tệp được cung cấp. Sản phẩm đi kèm là ứng dụng Streamlit, chương trình huấn luyện, mô hình đã huấn luyện và kết quả chạy lại. Đây là tài liệu tham khảo để người học tự thực hành và trình bày; không khẳng định người học đã tự thực hiện các thao tác trong tài liệu.

## 1 Xác định bài toán

Bài toán là phân loại nhị phân có giám sát. Đầu vào gồm 13 đặc trưng: Tuoi, GioiTinh, Glucose, HbA1C, Cholesterol, HDL, LDL, Triglycerid, Creatinine, Ure, AcidUric, AST và ALT. Đầu ra là NguyCoTieuDuong, mã hóa 0 là không có nguy cơ theo nhãn dữ liệu và 1 là có nguy cơ theo nhãn dữ liệu.

Mục tiêu của bài thực hành là huấn luyện mô hình dự đoán nhãn, so sánh chất lượng và xây dựng giao diện nhập dữ liệu. Không thể từ nhãn CSV kết luận đây là công cụ chẩn đoán hay dự báo bệnh trong tương lai, vì chưa có định nghĩa nhãn, thời điểm theo dõi và xác nhận lâm sàng.

## 2 Thu thập và khảo sát dữ liệu

Sử dụng CSDL_TieuDuong.csv của đề bài, gồm 4.707 bản ghi, 14 cột. Có 25 ô thiếu và không có dòng trùng hoàn toàn trong dữ liệu gốc. Giới tính xuất hiện dưới các cách ghi Nam, Nữ, Male, Female. Nhãn có TRUE, FALSE, 1, 0, Có nguy cơ và Không có nguy cơ.

Sau đồng bộ, có 3.610 bản ghi lớp 0 và 1.097 bản ghi lớp 1, tương ứng khoảng 76,7% và 23,3%. Dữ liệu mất cân bằng nên không đánh giá chỉ bằng Accuracy. Bộ CSV chưa kèm nguồn cơ sở y tế, đơn vị xét nghiệm, cách tạo nhãn hoặc thông tin đồng ý sử dụng; chỉ có thể gọi đây là dữ liệu được đề cung cấp, chưa đủ căn cứ khẳng định dữ liệu y tế thực tế đã được kiểm chứng.

## 3 Tiền xử lý dữ liệu

### 3.1 Làm sạch

Đồng bộ chuỗi bằng cách bỏ khoảng trắng và đổi chữ thường trước khi ánh xạ. Giới tính Male thành Nam, Female thành Nữ; nhãn TRUE, 1, Có nguy cơ thành 1, FALSE, 0, Không có nguy cơ thành 0. Không điền nhãn bằng giá trị dự đoán; dòng có nhãn không nhận diện được phải loại bỏ. Trong bộ hiện tại tất cả nhãn được nhận diện.

Các cột số được chuyển bằng pd.to_numeric với errors='coerce'. Giá trị không chuyển được, vô hạn hoặc âm được đưa về thiếu. Tuổi ngoài khoảng kiểm tra kỹ thuật 1–120 cũng đưa về thiếu. Đây là quy tắc kiểm tra của bài thực hành, không phải tiêu chuẩn lâm sàng.

Glucose = 750 và HbA1C = 8025 khác biệt lớn so với các giá trị còn lại nên được đánh dấu là thiếu, cần đối chiếu nguồn. Không tự sửa thành 7,50 hay 8,025 vì không có căn cứ về đơn vị hoặc lỗi nhập. Các giá trị cao khác được giữ lại: bất thường thống kê không mặc nhiên là sai y khoa. Do thiếu đơn vị, bài này chưa giải quyết hoàn toàn việc xác minh ngoại lai; cần trao đổi với nguồn dữ liệu trước khi dùng thực tế.

Sau làm sạch có 27 ô thiếu và vẫn giữ 4.707 dòng. Các cột số được điền bằng trung vị; giới tính được điền bằng giá trị phổ biến nhất. Bộ điền thiếu chỉ học trên tập train, tránh dùng thông tin của validation/test.

### 3.2 Biến đổi

OneHotEncoder mã hóa giới tính, handle_unknown='ignore' xử lý giá trị chưa gặp. StandardScaler chuẩn hóa cột số theo z = (x − trung bình train) / độ lệch chuẩn train. Logistic Regression hưởng lợi từ chuẩn hóa; Random Forest không bắt buộc, nhưng dùng chung cấu trúc pipeline để thuận tiện đối chiếu. ColumnTransformer xử lý riêng cột số và cột phân loại.

Pipeline gộp tiền xử lý và mô hình nên lúc dự đoán dữ liệu mới nhận đúng phép biến đổi đã học. Không fit bộ điền thiếu, chuẩn hóa hoặc mã hóa trên toàn bộ dữ liệu trước khi tách tập.

### 3.3 Phân chia dữ liệu

Tách 70% train và 30% tạm; chia tập tạm thành hai phần bằng nhau để có 15% validation và 15% test. Dùng random_state=42 và stratify theo nhãn ở cả hai lần tách. Số lượng thực tế do làm tròn là 3.294 train, 706 validation và 707 test.

Train dùng để học tham số; validation dùng để chọn mô hình; test chỉ dùng đánh giá mô hình đã chọn. Nếu bản ghi là nhiều lần khám của cùng một bệnh nhân, cần tách theo bệnh nhân; CSV hiện không có mã định danh để kiểm tra điều này.

## 4 Chọn mô hình

So sánh Logistic Regression và Random Forest. Đề liệt kê SVM cùng các lựa chọn khác, không bắt buộc triển khai mọi thuật toán. Logistic Regression là mô hình cơ sở dễ đối chiếu; Random Forest có thể biểu diễn quan hệ phi tuyến và tương tác giữa chỉ số.

Logistic Regression dùng max_iter=1000 và random_state=42. Random Forest dùng 100 cây, max_depth=10, random_state=42 và n_jobs=-1. Chọn mô hình có F1 cao nhất trên validation; nếu bằng nhau ưu tiên Recall. Với mục đích sàng lọc, Recall rất đáng quan tâm, nhưng bài thực hành này chưa tối ưu ngưỡng theo chi phí bỏ sót hoặc báo động giả.

## 5 Huấn luyện

Lệnh python train.py đọc CSV theo thư mục của chính tệp chương trình, làm sạch, phân chia dữ liệu rồi fit từng pipeline trên train. Sau đánh giá validation, chương trình chọn mô hình theo quy tắc đã đặt và lưu toàn bộ pipeline trong diabetes_model.joblib. Chương trình còn lưu ket_qua.json và hai ví dụ từ test trong vi_du_kiem_thu.csv.

Mô hình được lưu vẫn là mô hình đã học trên train 70%; không huấn luyện thêm trên validation sau khi báo cáo điểm test. Nhờ vậy kết quả báo cáo và mô hình lưu thống nhất. Tệp joblib gốc không được dùng để suy ra kết quả mới; phiên bản đi kèm được tạo lại từ chương trình đã sửa.

## 6 Đánh giá mô hình

Với lớp dương là 1, Accuracy là tỷ lệ dự đoán đúng; Precision là tỷ lệ đúng trong các dự đoán dương; Recall là tỷ lệ phát hiện được trong các nhãn dương thực tế; F1 là trung bình điều hòa của Precision và Recall. ROC AUC đo khả năng xếp hạng hai lớp qua nhiều ngưỡng.

### Kết quả validation

| Mô hình | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9632 | 1.0000 | 0.8415 | 0.9139 | 0.9809 |
| Random Forest | 0.9915 | 1.0000 | 0.9634 | 0.9814 | 1.0000 |

Random Forest có F1 và Recall cao hơn nên được chọn theo quy tắc validation. Đây là kết quả chạy lại bản sửa trên bộ dữ liệu đính kèm, không phải số liệu sao chép từ ứng dụng trực tuyến.

### Kết quả test của Random Forest

| Accuracy | Precision | Recall | F1 | ROC AUC |
| ---: | ---: | ---: | ---: | ---: |
| 0.9972 | 1.0000 | 0.9879 | 0.9939 | 1.0000 |

Ma trận nhầm lẫn có hàng là nhãn thực tế và cột là dự đoán:

| Nhãn thực tế | Dự đoán 0 | Dự đoán 1 |
| --- | ---: | ---: |
| 0 | 542 | 0 |
| 1 | 2 | 163 |

Trong 707 bản ghi test, mô hình dự đoán đúng 705, bỏ sót 2 nhãn dương và không có dương giả. Điểm rất cao chỉ phản ánh phép tách này. Cần kiểm tra nguồn nhãn, khả năng nhãn được tạo trực tiếp từ Glucose/HbA1C, dữ liệu lặp theo bệnh nhân và tính đại diện trước khi kết luận khả năng ứng dụng. Nhiều cột có trung vị và tứ phân vị trùng nhau cũng cần xem lại cách thu thập hoặc điền dữ liệu trước đó.

Chưa thực hiện kiểm định độc lập tại cơ sở khác, kiểm định chéo, hiệu chuẩn xác suất hoặc đánh giá theo nhóm tuổi/giới tính. Không nên diễn giải ROC AUC gần 1 là chứng minh hiệu quả y khoa gần tuyệt đối.

## 7 Giao diện và giải thích kết quả

app.py dùng Streamlit, nhập tuổi, giới tính và 11 chỉ số. Khi bấm Dự đoán, giao diện kiểm tra đủ dữ liệu rồi gửi một DataFrame tới pipeline. model.predict trả nhãn; model.predict_proba trả điểm của lớp 1. Giao diện hiển thị điểm, cảnh báo tương ứng và nhắc người dùng trao đổi với bác sĩ.

Phần giải thích mô tả các đầu vào và cơ chế tổng hợp 100 cây. Đây là giải thích cách vận hành chung, chưa phải giải thích riêng từng bệnh nhân bằng SHAP/LIME. Không khẳng định một xét nghiệm cụ thể gây ra kết quả nếu chưa tính đóng góp riêng.

Ứng dụng không lưu thông tin người bệnh, không đưa ra đơn thuốc. Do chưa xác định đơn vị trong CSV, không dùng trực tiếp cho xét nghiệm ngoài bộ bài tập. Điểm mô hình không phải xác suất mắc bệnh đã được hiệu chuẩn.

## 8 Đối chiếu mã nguồn ban đầu và bản sửa

| Điểm trong tệp ban đầu | Ảnh hưởng | Bản sửa |
| --- | --- | --- |
| Đường dẫn CSV cố định C:\Users\Admin\... | Không chạy ở máy khác | Path(__file__) và đường dẫn tương đối |
| Không ánh xạ Có nguy cơ | Có thể tạo nhãn NaN, gây lỗi stratify/fit | Bổ sung đầy đủ cách ghi |
| dropna trước khi chuyển kiểu, ánh xạ | Không xử lý thiếu phát sinh sau đó | Chuẩn hóa trước, dùng imputer trong pipeline |
| Chỉ kiểm tra, chưa xử lý giá trị bất thường | Mô hình có thể bị chi phối bởi lỗi nhập | Đánh dấu hai giá trị nghi vấn và nêu giới hạn |
| Chọn sẵn Random Forest | Thiếu quy tắc chọn rõ ràng | Chọn theo F1 validation |
| Lặp page_config, title, caption | Giao diện lặp nội dung; cấu hình phụ thuộc phiên bản | Chỉ cấu hình một lần |
| Hiển thị thư mục và đường dẫn mô hình | Lộ thông tin kỹ thuật không cần thiết | Bỏ nội dung debug |

## 9 Sản phẩm cụ thể và nguồn tham khảo

Thư mục đi kèm có app.py, train.py, requirements.txt, CSDL_TieuDuong.csv, diabetes_model.joblib, ket_qua.json, vi_du_kiem_thu.csv và file hướng dẫn riêng. Sản phẩm huấn luyện và dự đoán đã chạy thành công; app.py đã kiểm tra cú pháp. Môi trường hiện chưa có Streamlit nên chưa xác nhận tương tác trình duyệt hoặc triển khai trực tuyến.

Nguồn phân tích chính: đề KiemTra_TriTueNhanTao.docx và các tệp người dùng cung cấp. Hai nguồn trực tuyến được chỉ dẫn là https://github.com/tdhai-2batrung/CNTT4/tree/main/BTTriTueNhanTao và https://qwwyb5xkk7aaqdlh2cwgct.streamlit.app/ nhưng không truy cập được trong phiên kiểm tra này. Khi nộp, người học cần tự kiểm tra lại link và ghi nhận đúng nguồn mã tham khảo theo yêu cầu của môn học.

## 10 Các câu hỏi cần tự giải thích được

- Vì sao không chỉ dùng Accuracy? Vì lớp 0 chiếm khoảng 76,7%, mô hình luôn đoán 0 vẫn có Accuracy cao nhưng Recall lớp 1 bằng 0.
- Vì sao cần validation riêng? Để lựa chọn mô hình mà vẫn giữ test cho đánh giá cuối cùng.
- Vì sao không chuẩn hóa toàn bộ CSV trước? Vì trung bình, độ lệch chuẩn và trung vị của test sẽ ảnh hưởng phép biến đổi, gây rò rỉ dữ liệu.
- Vì sao không tự sửa 8025 thành 8,025? Vì chưa có chứng cứ xác nhận lỗi và đơn vị.
- Vì sao điểm 99,7% chưa đủ dùng trong bệnh viện? Vì đó là Accuracy trên một lần chia của bộ dữ liệu chưa được xác minh nguồn và chưa kiểm định độc lập.
- Tệp joblib chứa gì? Pipeline tiền xử lý cùng bộ phân loại, không chỉ riêng các cây.
