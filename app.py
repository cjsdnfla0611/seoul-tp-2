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
        "train_start": int(df_train["연도"].min()),
        "train_end": int(df_train["연도"].max()),
        "train_count": len(df_train),
    }


# 전체 데이터 회귀 (비교용)
slope_all, intercept_all = np.polyfit(df_yearly["X"].values, df_yearly["평균기온"].values, 1)
rate_all_100y = slope_all * 100

# 모델 fit & 평가 실행
res_100 = fit_and_evaluate(df_train_100, df_test, "100년 학습 (1906~2005)")
res_50 = fit_and_evaluate(df_train_50, df_test, "50년 학습 (1956~2005)")

# 4. 화면 출력 - 개요
st.subheader("📌 데이터셋 구성")
col_info1, col_info2, col_info3 = st.columns(3)
col_info1.metric("공통 테스트 데이터", "2006년 ~ 2025년", f"{len(df_test)}개 해")
col_info2.metric("학습 모델 A (100년 데이터)", "1906년 ~ 2005년", f"{res_100['train_count']}개 해")
col_info3.metric("학습 모델 B (50년 데이터)", "1956년 ~ 2005년", f"{res_50['train_count']}개 해")

st.divider()

# 5. 모델 비교 성능 지표 출력
st.subheader("📊 최근 20년(2006~2025) 테스트 데이터에 대한 예측 성능 비교")

metric_col1, metric_col2, metric_col3 = st.columns(3)

# 100년당 기울기 비교
metric_col1.metric(
    label="기울기 (100년당 상승 폭)",
    value=f"+{res_50['rate_100y']:.2f} °C / 100년",
    delta=f"100년 모델 대비 +{res_50['rate_100y'] - res_100['rate_100y']:.2f} °C 가속",
    help="50년 모델(1956~2005) vs 100년 모델(1906~2005)",
)

# MAE 비교 (낮을수록 좋음)
mae_diff = res_50["mae"] - res_100["mae"]
metric_col2.metric(
    label="MAE (평균 절대 오차)",
    value=f"{res_50['mae']:.4f} °C",
    delta=f"{mae_diff:+.4f} °C ({'개선' if mae_diff < 0 else '저하'})",
    delta_color="inverse",
)

# R² 비교 (높을수록 좋음)
r2_diff = res_50["r2"] - res_100["r2"]
metric_col3.metric(
    label="R² (결정계수)",
    value=f"{res_50['r2']:.4f}",
    delta=f"{r2_diff:+.4f} ({'개선' if r2_diff > 0 else '저하'})",
)

# 상세 비교 표
summary_df = pd.DataFrame(
    [
        {
            "모델 구분": "전체 데이터 분석 (참고)",
            "학습 기간": f"{int(df_yearly['연도'].min())}~{int(df_yearly['연도'].max())}",
            "기울기 (°C/100년)": f"+{rate_all_100y:.2f}",
            "MAE (°C)": "-",
            "MSE (°C²)": "-",
            "R² Score": "-",
        },
        {
            "모델 구분": "100년 학습 모델",
            "학습 기간": "1906~2005",
            "기울기 (°C/100년)": f"+{res_100['rate_100y']:.2f}",
            "MAE (°C)": f"{res_100['mae']:.4f}",
            "MSE (°C²)": f"{res_100['mse']:.4f}",
            "R² Score": f"{res_100['r2']:.4f}",
        },
        {
            "모델 구분": "50년 학습 모델",
            "학습 기간": "1956~2005",
            "기울기 (°C/100년)": f"+{res_50['rate_100y']:.2f}",
            "MAE (°C)": f"{res_50['mae']:.4f}",
            "MSE (°C²)": f"{res_50['mse']:.4f}",
            "R² Score": f"{res_50['r2']:.4f}",
        },
    ]
)

st.table(summary_df)

st.divider()

# 6. Plotly 시각화
st.subheader("📈 회귀 직선 및 테스트 데이터 비교 차트")

fig = go.Figure()

# 과거 관측 데이터 (학습용)
fig.add_trace(
    go.Scatter(
        x=df_yearly[df_yearly["연도"] < 2006]["연도"],
        y=df_yearly[df_yearly["연도"] < 2006]["평균기온"],
        mode="markers",
        name="과거 관측 데이터 (학습 구간)",
        marker=dict(size=7, color="#7f7f7f", opacity=0.5),
    )
)

# 최근 20년 테스트 데이터
fig.add_trace(
    go.Scatter(
        x=df_test["연도"],
        y=df_test["평균기온"],
        mode="markers",
        name="최근 20년 테스트 데이터 (2006~2025)",
        marker=dict(
