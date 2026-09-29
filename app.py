import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연도별 평균기온 예측기")

# 1. 데이터 불러오기
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"].str.strip())
    df["연도"] = df["날짜"].dt.year
    return df


df_raw = load_data()

# 2. 전처리 & 정제
# - 2025년 이하 데이터만 필터링
# - 연도별 관측일 300일 이상 필터링
df_filtered = df_raw[df_raw["연도"] <= 2025].copy()

yearly_counts = df_filtered.groupby("연도")["평균기온"].count()
valid_years = yearly_counts[yearly_counts >= 300].index

df_valid = df_filtered[df_filtered["연도"].isin(valid_years)]

# 연도별 평균기온 계산
df_yearly = df_valid.groupby("연도")["평균기온"].mean().reset_index()

# 3. 회귀분석 (독립변수: 1908년부터 경과한 연수)
df_yearly["X"] = df_yearly["연도"] - 1908
X = df_yearly["X"].values
Y = df_yearly["평균기온"].values

# 선형 회귀 계수 계산 (1차 다항식)
slope, intercept = np.polyfit(X, Y, 1)

# 상관계수 계산
corr = np.corrcoef(df_yearly["연도"], df_yearly["평균기온"])[0, 1]

# 정보 추출
num_years = len(df_yearly)
start_year = int(df_yearly["연도"].min())
end_year = int(df_yearly["연도"].max())

# 4. 화면 UI - 기본 정보 표시
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 연도 개수", f"{num_years}개 해")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수 (r)", f"{corr:.4f}")

st.divider()

# 5. 연도 선택 슬라이더 & 예상 기온 출력
selected_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2025, step=1)

# 회귀 방정식을 이용한 예상 기온 계산
predicted_temp = slope * (selected_year - 1908) + intercept

st.markdown(
    f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 25px;">
        <h3 style="margin: 0; color: #333;">{selected_year}년 서울 예상 평균기온</h3>
        <h1 style="margin: 10px 0 0 0; color: #ff4b4b; font-size: 3rem;">{predicted_temp:.2f} °C</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# 6. Plotly 시각화
fig = go.Figure()

# 실제 관측 데이터 산점도
fig.add_trace(
    go.Scatter(
        x=df_yearly["연도"],
        y=df_yearly["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=8, color="#1f77b4"),
    )
)

# 회귀 직선 (1900년 ~ 2100년 전체 구간)
x_range = np.arange(1900, 2101)
y_pred_range = slope * (x_range - 1908) + intercept

fig.add_trace(
    go.Scatter(
        x=x_range,
        y=y_pred_range,
        mode="lines",
        name="선형 회귀선",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
    )
)

# 선택한 연도 강조 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"선택한 해 ({selected_year}년)",
        marker=dict(size=14, color="#d62728", symbol="star"),
    )
)

fig.update_layout(
    title=f"서울 연도별 평균기온 및 회귀 직선 (1908년 기준 경과 연수 분석)",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    xaxis=dict(tickformat="d"),
    hovermode="x unified",
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)
