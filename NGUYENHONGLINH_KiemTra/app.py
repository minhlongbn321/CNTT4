from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Sàng lọc nguy cơ tiểu đường', page_icon='🩺')
st.title('Sàng lọc nguy cơ tiểu đường')
st.caption('Bài thự hành, không sử dụng thực tế')
BASE=Path(__file__).resolve().parent
@st.cache_resource
def load_model():
    return joblib.load(BASE/'diabetes_model.joblib')
if not (BASE/'diabetes_model.joblib').is_file():
    st.error('Chưa có mô hình.'); st.stop()
model=load_model()
columns=['Glucose','HbA1C','Cholesterol','HDL','LDL','Triglycerid','Creatinine','Ure','AcidUric','AST','ALT']
st.info('Bài thự hành, không sử dụng thực tế.')
with st.form('patient'):
    age=st.number_input('Tuổi',min_value=1,max_value=120,value=40)
    gender=st.selectbox('Giới tính',['Nam','Nữ'])
    values={c:st.number_input(c,min_value=0.0,value=None,format='%.3f',placeholder='Nhập chỉ số') for c in columns}
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
        if pred==1: st.warning('Mô hình xếp vào nhóm có nguy cơ. Cần trao đổi với bác sĩ để được đánh giá.')
        else: st.success('Mô hình xếp vào nhóm nguy cơ thấp. Kết quả không loại trừ bệnh.')
        st.write('.')
        st.write('Đây là bài thực hành. Không trì hoãn khám bệnh dựa trên công cụ này.')
