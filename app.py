import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="서울 기온 회귀 모델 평가", layout="wide")

st.title("수업용: 서울 연평균 기온 선형회귀 모델 평가")

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

df_yearly["X"] = df_yearly["연도"] - 1908

df_test = df_yearly[(df_yearly["연도"] >= 2006) & (df_yearly["연도"] <= 2025)].copy()
df_train_100 = df_yearly[(df_yearly["연도"] >= 1906) & (df_yearly["연도"] <= 2005)].copy()
df_train_50 = df_yearly[(df_yearly["연도"] >= 1956) & (df_yearly["연도"] <= 2005)].copy()


def fit_and_evaluate(df_train, df_test, label):
    slope, intercept = np.polyfit(df_train["X"].values, df_train["평균기온"].values, 1)

    y_true = df_test["평균기온"].values
    y_pred = slope * df_test["X"].values + intercept

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
        "train_start": int(df_train["연도"].min()),
        "train_end": int(df_train["연도"].max()),
        "train_count": len(df_train),
    }


slope_all, intercept_all = np.polyfit(df_yearly["X"].values, df_yearly["평균기온"].values, 1)
rate_all_100y = slope_all * 100

res_100 = fit_and_evaluate(df_train_100, df_test, "100년 학습 (1906~2005)")
res_50 = fit_and_evaluate(df_train_50, df_test, "50년 학습 (1956~2005)")

st.subheader("데이터셋 구성")
col_info1, col_info2, col_info3 = st.columns(3)
col_info1.metric("공통 테스트 데이터", "2006년 ~ 2025년", f"{len(df_test)}개 해")
col_info2.metric("학습 모델 A (100년 데이터)", "1906년 ~ 2005년", f"{res_100['train_count']}개 해")
col_info3.metric("학습 모델 B (50년 데이터)", "1956년 ~ 2005년",
