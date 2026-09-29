import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연도별 평균기온 및 온난화 속도 비교 예측기")

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

# 3. 회귀분석 1: 전체 기간 (독립변수: 1908년부터 경과한 연수)
df_yearly["X"] = df_yearly["연도"] - 1908
X_all = df_yearly["X"].values
Y_all = df_yearly["평균기온"].values

slope_all, intercept_all = np.polyfit(X_all, Y_all, 1)

# 4. 회귀분석 2: 최근 20년 기간
recent_20_years = df_yearly[
    df_yearly["연도"] >= (df_yearly["연도"].max() - 19)
]
X_recent = recent_20_years["X"].values
Y_recent = recent_20_years["평균기온"].values

slope_recent, intercept_recent = np.polyfit(X_recent, Y_recent, 1)

# 상관계수 (전체)
corr = np.corrcoef(df_yearly["연도"], df_yearly["평균기온"])[0, 1]

# 정보 추출
num_years = len(df_yearly)
start_year = int(df_yearly["연도"].min())
end_year = int(df_yearly["연도"].max())
recent_start_year = int(recent_20_years["연도"].min())

# 100년당 기온 상승량 (°C / 100년)
rate_all_100y = slope_all * 100
rate_recent_100y = slope_recent * 100

# 5. 화면 UI - 기본 정보 표시
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 연도 개수", f"{num_years}개 해")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수 (전체)", f"{corr:.4f}")

st.divider()

# 6. 기울기(100년당 상승 폭) 나란히 비교
st.subheader("🔥 100년당 기온 상승 속도 비교")
metric_col1, metric_col2 = st.columns(2)

metric_col1.metric(
    label=f"전체 기간 ({start_year}~{end_year}년)",
    value=f"+{rate_all_100y:.2f} °C / 100년",
    help="전체 관측 데이터로 계산한 100년당 기온 상승량입니다.",
)

delta_recent = rate_recent_100y - rate_all_100y
metric_col2.metric(
    label=f"최근 20년 ({recent_start_year}~{end_year}년)",
    value=f"+{rate_recent_100y:.2f} °C / 100년",
    delta=f"전체 평균 대비 +{delta_recent:.2f} °C 가속",
    delta_color="normal",
    help="최근 20년간의 데이터만으로 추산한 100년당 기온 상승 추세입니다.",
)

st.divider()

# 7. 연도 선택 슬라이더 & 예상 기온 출력 (전체 추세 vs 최근 추세)
selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

# 예측값 계산
pred_all = slope_all * (selected_year - 1908) + intercept_all
pred_recent = slope_recent * (selected_year - 1908) + intercept_recent

res_col1, res_col2 = st.columns(2)

with res_col1:
    st.markdown(
        f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h4 style="margin: 0; color: #555;">전체 추세 기준 {selected_year}년 예상 기온</h4>
            <h1 style="margin: 10px 0 0 0; color: #ff7f0e; font-size: 2.5rem;">{pred_all:.2f} °C</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

with res_col2:
    st.markdown(
        f"""
        <div style="background-color: #ffebeb; padding: 20px; border-radius: 10px; text-align: center;">
            <h4 style="margin: 0; color: #555;">최근 20년 추세 기준 {selected_year}년 예상 기온</h4>
            <h1 style="margin: 10px 0 0 0; color: #d62728; font-size: 2.5rem;">{pred_recent:.2f} °C</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# 8. Plotly 시각화 (두 회귀선 함께 표시)
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

# 전체 회귀 직선 (1900년 ~ 2100년)
x_range = np.arange(1900, 2101)
y_pred_all_range = slope_all * (x_range - 1908) + intercept_all
y_pred_recent_range = slope_recent * (x_range - 1908) + intercept_recent

fig.add_trace(
    go.Scatter(
        x=x_range,
        y=y_pred_all_range,
        mode="lines",
        name=f"전체 기간 회귀선 (+{rate_all_100y:.2f}°C/100년)",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
    )
)

# 최근 20년 회귀 직선 (1900년 ~ 2100년)
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=y_pred_recent_range,
        mode="lines",
        name=f"최근 20년 회귀선 (+{rate_recent_100y:.2f}°C/100년)",
        line=dict(color="#d62728", width=2, dash="dot"),
    )
)

# 선택한 연도 강조 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year, selected_year],
        y=[pred_all, pred_recent],
        mode="markers+text",
        name=f"선택 연도 ({selected_year}년)",
        marker=dict(size=12, color=["#ff7f0e", "#d62728"], symbol="star"),
        text=[f"{pred_all:.2f}°C", f"{pred_recent:.2f}°C"],
        textposition="top center",
    )
)

fig.update_layout(
    title="서울 연도별 평균기온 및 회귀 직선 비교 (전체 vs 최근 20년)",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    xaxis=dict(tickformat="d"),
    hovermode="x unified",
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)
