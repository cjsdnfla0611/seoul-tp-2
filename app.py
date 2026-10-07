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
col_info3.metric("학습 모델 B (50년 데이터)", "1956년 ~ 2005년", f"{res_50['train_count']}개 해")

st.divider()

st.subheader("최근 20년(2006~2025) 테스트 데이터 예측 성능 비교")

metric_col1, metric_col2, metric_col3 = st.columns(3)

rate_diff = res_50['rate_100y'] - res_100['rate_100y']
metric_col1.metric(
    label="기울기 (100년당 상승 폭)",
    value=f"+{res_50['rate_100y']:.2f} °C / 100년",
    delta=f"100년 모델 대비 +{rate_diff:.2f} °C 가속"
)

mae_diff = res_50["mae"] - res_100["mae"]
mae_label = "개선" if mae_diff < 0 else "저하"
metric_col2.metric(
    label="MAE (평균 절대 오차)",
    value=f"{res_50['mae']:.4f} °C",
    delta=f"{mae_diff:+.4f} °C ({mae_label})",
    delta_color="inverse"
)

r2_diff = res_50["r2"] - res_100["r2"]
r2_label = "개선" if r2_diff > 0 else "저하"
metric_col3.metric(
    label="R² (결정계수)",
    value=f"{res_50['r2']:.4f}",
    delta=f"{r2_diff:+.4f} ({r2_label})"
)

summary_data = [
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

summary_df = pd.DataFrame(summary_data)
st.table(summary_df)

st.divider()

st.subheader("회귀 직선 및 테스트 데이터 비교 차트")

fig = go.Figure()

trace_past = go.Scatter(
    x=df_yearly[df_yearly["연도"] < 2006]["연도"],
    y=df_yearly[df_yearly["연도"] < 2006]["평균기온"],
    mode="markers",
    name="과거 관측 데이터 (학습 구간)",
    marker=dict(size=7, color="#7f7f7f", opacity=0.5)
)

trace_test = go.Scatter(
    x=df_test["연도"],
    y=df_test["평균기온"],
    mode="markers",
    name="최근 20년 테스트 데이터 (2006~2025)",
    marker=dict(size=9, color="#d62728", symbol="diamond")
)

x_range = np.arange(1900, 2031)

y_pred_100 = res_100["slope"] * (x_range - 1908) + res_100["intercept"]
trace_100 = go.Scatter(
    x=x_range,
    y=y_pred_100,
    mode="lines",
    name=f"100년 학습 회귀선 (+{res_100['rate_100y']:.2f}°C/100년)",
    line=dict(color="#1f77b4", width=2.5)
)

y_pred_50 = res_50["slope"] * (x_range - 1908) + res_50["intercept"]
trace_50 = go.Scatter(
    x=x_range,
    y=y_pred_50,
    mode="lines",
    name=f"50년 학습 회귀선 (+{res_50['rate_100y']:.2f}°C/100년)",
    line=dict(color="#ff7f0e", width=2.5, dash="dash")
)

fig.add_trace(trace_past)
fig.add_trace(trace_test)
fig.add_trace(trace_100)
fig.add_trace(trace_50)

fig.update_layout(
    title="학습 기간별 회귀선의 최근 20년 기온 예측 성능 비교",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    xaxis=dict(tickformat="d"),
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"""
### 주요 분석 결과
1. **기울기 비교**: 50년 학습 모델(1956~2005)의 100년당 기온 상승량(+{res_50['rate_100y']:.2f}°C)이 100년 학습 모델(1906~2005, +{res_100['rate_100y']:.2f}°C)보다 높게 나타납니다.
2. **예측 성능 비교**: 최근 20년(2006~2025) 실제 기온 데이터에 대해 최근 50년 데이터로 학습한 회귀 모델의 오차(MAE)가 더 적고 결정계수($R^2$)가 높아, 가속화되는 기온 상승 추세를 더 잘 설명함을 확인할 수 있습니다.
"""
)
