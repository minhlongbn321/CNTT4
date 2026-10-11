from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Sàng lọc nguy cơ tiểu đường', page_icon='$$$')
st.title('THỰC HÀNH SÀNG LỌC NGUY CƠ TIỂU ĐƯỜNG - NGUYỄN HỒNG LĨNH')
st.caption('Bài thực hành. Không sử dụng thực tế.')
BASE=Path(__file__).resolve().parent
@st.cache_resource
def load_model():
    return joblib.load(BASE/'diabetes_model.joblib')
if not (BASE/'diabetes_model.joblib').is_file():
    st.error('Chưa có mô hình.'); st.stop()
model=load_model()
columns=['Glucose','HbA1C','Cholesterol','HDL','LDL','Triglycerid','Creatinine','Ure','AcidUric','AST','ALT']
st.info('Chỉ sử dụng để nghiên cứu, không sử dụng thật.')
with st.form('patient'):
    age=st.number_input('Tuổi',min_value=1,max_value=120,value=40)
    gender=st.selectbox('Giới tính',['Nam','Nữ'])
    values = {}
    for start in range(0, len(columns), 3):
        row = st.columns(3)
        for position, name in enumerate(columns[start:start + 3]):
            with row[position]:
                values[name] = st.number_input(
                    name, min_value=0.0, value=None,
                    format='%.3f', placeholder='Nhập chỉ số', key=name
                )
    submitted=st.form_submit_button('Dự đoán',type='primary')
if submitted:
    if any(v is None for v in values.values()):
        st.error('Vui lòng nhập đủ các chỉ số.')
    elif values['Glucose']==750 or values['HbA1C']==8025:
        st.error('Giá trị trùng. Hãy kiểm tra lại kết quả và đơn vị.')
    else:
        x=pd.DataFrame([{'Tuoi':age,'GioiTinh':gender,**values}])
        pred=int(model.predict(x)[0]);score=float(model.predict_proba(x)[0,list(model.classes_).index(1)])
        st.metric('Điểm mô hình ước tính',f'{score:.1%}')
        if pred==1: st.warning('Mô hình xếp vào nhóm có nguy cơ. Cần đi khám thật để có KQ chính xac.')
        else: st.success('Mô hình xếp vào nhóm nguy cơ thấp. Kết quả không loại trừ bệnh.')
        st.write('.')
        st.write('Không tự điều chỉnh thuốc dựa trên công cụ này. Chỉ dùng để tham khảo!!!')
