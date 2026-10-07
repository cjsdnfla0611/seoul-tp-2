import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="서울 기온 다항회귀 모델 평가", layout="wide")
st.title("수업용: 서울 연평균 기온 다항회귀(1차·3차·9차) 모델 평가")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"].str.strip())
    df["연도"] = df["날짜"].dt.year
    return df

df_raw = load_data()

df_filtered = df_raw[df_raw["연도"] <= 2025].copy()
yearly_counts = df_filtered.groupby("연도")["평균기온"].count()
valid_years = yearly_counts[yearly_counts >= 300].index

df_valid = df_filtered[df_filtered["연도"].isin(valid_years)]
df_yearly = df_valid.groupby("연도")["평균기온"].mean().reset_index()

BASE_YEAR = 1908
df_yearly["X"] = df_yearly["연도"] - BASE_YEAR

df_train = df_yearly[df_yearly["연도"] < 2005].copy()
df_test = df_yearly[df_yearly["연도"] >= 2005].copy()

st.subheader("📌 데이터셋 구성 정보")
col_train, col_test = st.columns(2)
col_train.metric("훈련용 데이터 (<2005년)", f"{df_train['연도'].min()}년 ~ {df_train['연도'].max()}년", f"{len(df_train)}개 연도")
col_test.metric("테스트용 데이터 (≥2005년)", f"{df_test['연도'].min()}년 ~ {df_test['연도'].max()}년", f"{len(df_test)}개 연도")

st.divider()

degrees = [1, 3, 9]
results = []
models = {}

X_train = df_train["X"].values
y_train = df_train["평균기온"].values
X_test = df_test["X"].values
y_test = df_test["평균기온"].values

X_2050 = np.array([2050 - BASE_YEAR])

for deg in degrees:
    coeffs = np.polyfit(X_train, y_train, deg)
    poly_func = np.poly1d(coeffs)
    models[deg] = poly_func
    
    y_pred_test
