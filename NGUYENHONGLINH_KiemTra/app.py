from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Sàng lọc nguy cơ tiểu đường', page_icon='🩺')
st.title('Sàng lọc nguy cơ tiểu đường')
st.caption('Bản thực hành học máy. Không thay thế chẩn đoán y khoa.')
BASE=Path(__file__).resolve().parent
@st.cache_resource
def load_model():
    return joblib.load(BASE/'diabetes_model.joblib')
if not (BASE/'diabetes_model.joblib').is_file():
    st.error('Chưa có mô hình. Hãy chạy python train.py trước.'); st.stop()
model=load_model()
columns=['Glucose','HbA1C','Cholesterol','HDL','LDL','Triglycerid','Creatinine','Ure','AcidUric','AST','ALT']
st.info('Chỉ sử dụng số liệu có cùng đơn vị với dữ liệu huấn luyện. File CSV chưa cung cấp bảng mô tả đơn vị; cần xác minh trước khi sử dụng ngoài bài thực hành.')
with st.form('patient'):
    age=st.number_input('Tuổi',min_value=1,max_value=120,value=40)
    gender=st.selectbox('Giới tính',['Nam','Nữ'])
    values={c:st.number_input(c,min_value=0.0,value=None,format='%.3f',placeholder='Nhập chỉ số') for c in columns}
    submitted=st.form_submit_button('Dự đoán',type='primary')
if submitted:
    if any(v is None for v in values.values()):
        st.error('Vui lòng nhập đủ các chỉ số.')
    elif values['Glucose']==750 or values['HbA1C']==8025:
        st.error('Giá trị trùng với dữ liệu bất thường đã đánh dấu khi huấn luyện. Hãy kiểm tra lại kết quả và đơn vị.')
    else:
        x=pd.DataFrame([{'Tuoi':age,'GioiTinh':gender,**values}])
        pred=int(model.predict(x)[0]);score=float(model.predict_proba(x)[0,list(model.classes_).index(1)])
        st.metric('Điểm mô hình ước tính',f'{score:.1%}')
        if pred==1: st.warning('Mô hình xếp vào nhóm có nguy cơ theo nhãn dữ liệu. Cần trao đổi với bác sĩ để được đánh giá.')
        else: st.success('Mô hình xếp vào nhóm nguy cơ thấp hơn theo nhãn dữ liệu. Kết quả không loại trừ bệnh.')
        st.write('Pipeline xử lý tuổi, giới tính và 11 chỉ số xét nghiệm; Random Forest tổng hợp dự đoán từ 100 cây. Điểm hiển thị là đầu ra predict_proba, chưa phải xác suất mắc bệnh được hiệu chuẩn lâm sàng.')
        st.write('Không tự điều chỉnh thuốc hay trì hoãn khám bệnh dựa trên công cụ này.')
