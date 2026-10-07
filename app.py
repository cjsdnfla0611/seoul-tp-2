import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="서울 기온 회귀 모델 평가", layout="wide")

st.title("🌡️ 서울 연평균 기온 선형회귀 모델 학습 및 성능 평가")

# 1. 데이터 불러오기
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"].str.strip())
    df["연도"] = df["날짜"].dt.year
    return df


df_raw = load_data()

# 2. 데이터 전처리
# 2025년 이하 & 관측일 300일 이상 필터링
df_filtered = df_raw[df_raw["연도"] <= 2025].copy()
yearly_counts = df_filtered.groupby("연도")["평균기온"].count()
valid_years = yearly_counts[yearly_counts >= 300].index

df_valid = df_filtered[df_filtered["연도"].isin(valid_years)]
df_yearly = df_valid.groupby("연도")["평균기온"].mean().reset_index()

# 1908년 기준 경과 연수 X (독립변수)
df_yearly["X"] = df_yearly["연도"] - 1908

# 3. 데이터셋 분할
# 공통 테스트 데이터: 최근 20년 (2006~2025)
df_test = df_yearly[(df_yearly["연도"] >= 2006) & (df_yearly["연도"] <= 2025)].copy()

# 훈련 데이터 1: 최근 100년 구간 (1906~2005)
df_train_100 = df_yearly[(df_yearly["연도"] >= 1906) & (df_yearly["연도"] <= 2005)].copy()

# 훈련 데이터 2: 최근 50년 구간 (1956~2005)
df_train_50 = df_yearly[(df_yearly["연도"] >= 1956) & (df_yearly["연도"] <= 2005)].copy()


# 회귀 모델 학습 및 평가 함수
def fit_and_evaluate(df_train, df_test, label):
    # 모델 학습
    slope, intercept = np.polyfit(df_train["X"].values, df_train["평균기온"].values, 1)

    # 테스트 데이터 예측
    y_true = df_test["평균기온"].values
    y_pred = slope * df_test["X"].values + intercept

    # 평가지표 계산
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rate_100y = slope * 100

    return {
        "label": label,
        "slope": slope,
        "intercept": intercept,
        "rate_100y": rate_100y,
        "mae": mae,
        "mse": mse,
        "r2": r2,
