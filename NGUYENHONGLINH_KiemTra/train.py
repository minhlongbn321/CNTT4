from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE = Path(__file__).resolve().parent
NUM = ['Tuoi','Glucose','HbA1C','Cholesterol','HDL','LDL','Triglycerid','Creatinine','Ure','AcidUric','AST','ALT']
def clean(df):
    df = df.copy()
    df['GioiTinh'] = df['GioiTinh'].astype('string').str.strip().str.lower().map({'nam':'Nam','male':'Nam','m':'Nam','nữ':'Nữ','nu':'Nữ','female':'Nữ','f':'Nữ'})
    df['NguyCoTieuDuong'] = df['NguyCoTieuDuong'].astype('string').str.strip().str.lower().map({'true':1,'false':0,'1':1,'0':0,'có nguy cơ':1,'không có nguy cơ':0})
    for col in NUM:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf,-np.inf],np.nan)
        df.loc[df[col]<0,col] = np.nan
    # Hai giá trị cần xác minh với nguồn dữ liệu; không tự suy đoán giá trị đúng.
    df.loc[df['Glucose']==750, 'Glucose'] = np.nan
    df.loc[df['HbA1C']==8025, 'HbA1C'] = np.nan
    df.loc[(df['Tuoi']<1)|(df['Tuoi']>120),'Tuoi'] = np.nan
    return df.dropna(subset=['NguyCoTieuDuong']).drop_duplicates()
def metrics(model,X,y):
    p=model.predict(X); q=model.predict_proba(X)[:,list(model.classes_).index(1)]
    return dict(Accuracy=accuracy_score(y,p),Precision=precision_score(y,p,zero_division=0),Recall=recall_score(y,p,zero_division=0),F1=f1_score(y,p,zero_division=0),ROC_AUC=roc_auc_score(y,q),confusion_matrix=confusion_matrix(y,p,labels=[0,1]).tolist())
def main():
    raw=pd.read_csv(BASE/'CSDL_TieuDuong.csv'); df=clean(raw)
    X=df[NUM+['GioiTinh']]; y=df['NguyCoTieuDuong'].astype(int)
    a,b,c,d=train_test_split(X,y,test_size=.3,stratify=y,random_state=42)
    v,t,yv,yt=train_test_split(b,d,test_size=.5,stratify=d,random_state=42)
    models={}; validation={}
    for name, estimator in {'Logistic Regression':LogisticRegression(max_iter=1000,random_state=42),'Random Forest':RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42,n_jobs=-1)}.items():
        pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),['GioiTinh'])])
        model=Pipeline([('preprocessor',pre),('model',estimator)]).fit(a,c)
        models[name]=model; validation[name]=metrics(model,v,yv)
    selected=max(validation,key=lambda k:(validation[k]['F1'],validation[k]['Recall']))
    best=models[selected]
    report={'raw_rows':len(raw),'clean_rows':len(df),'missing_cells_before':int(raw.isna().sum().sum()),'missing_cells_after_cleaning':int(df.isna().sum().sum()),'class_counts':y.value_counts().sort_index().to_dict(),'split':{'train':len(a),'validation':len(v),'test':len(t)},'selection_rule':'Validation F1; tie break Recall','validation':validation,'selected':selected,'test':metrics(best,t,yt)}
    joblib.dump(best,BASE/'diabetes_model.joblib')
    (BASE/'ket_qua.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    cases=t.loc[[yt[yt==0].index[0],yt[yt==1].index[0]]].copy();cases['Nhan_thuc_te']=yt.loc[cases.index];cases['Du_doan']=best.predict(cases[NUM+['GioiTinh']]);cases['Diem_mo_hinh']=best.predict_proba(cases[NUM+['GioiTinh']])[:,1];cases.to_csv(BASE/'vi_du_kiem_thu.csv',index=False)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
