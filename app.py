import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연도별 평균기온 & 10년 단위 구간별 기울기 비교")

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
df_filtered = df_raw[df_raw["연도"] <= 2025].copy()

yearly_counts = df_filtered.groupby("연도")["평균기온"].count()
valid_years = yearly_counts[yearly_counts >= 300].index

df_valid = df_filtered[df_filtered["연도"].isin(valid_years)]
df_yearly = df_valid.groupby("연도")["평균기온"].mean().reset_index()

# 회귀분석용 기준 연수 (1908년 기준)
df_yearly["X"] = df_yearly["연도"] - 1908

max_year = int(df_yearly["연도"].max())
start_year = int(df_yearly["연도"].min())

# 3. 10년 차이 구간 설정 (최근 10년, 20년, 30년, 40년, 50년)
spans = [10, 20, 30, 40, 50]
colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]  # 빨강, 주황, 초록, 보라, 갈색

reg_results = []

for span, color in zip(spans, colors):
    sub_df = df_yearly[df_yearly["연도"] >= (max_year - span + 1)]
    
    # 해당 구간 데이터로 회귀 계산
    slope, intercept = np.polyfit(sub_df["X"].values, sub_df["평균기온"].values, 1)
    rate_100y = slope * 100
    sub_start = int(sub_df["연도"].min())
    
    reg_results.append({
        "span": span,
        "label": f"최근 {span}년 ({sub_start}~{max_year}년)",
        "slope": slope,
        "intercept": intercept,
        "rate_100y": rate_100y,
        "color": color
    })

# 전체 기간 기준 회귀도 함께 계산
slope_all, intercept_all = np.polyfit(df_yearly["X"].values, df_yearly["평균기온"].values, 1)
rate_all_100y = slope_all * 100
corr = np.corrcoef(df_yearly["연도"], df_yearly["평균기온"])[0, 1]

# 4. 상단 요약 정보
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 연도 개수", f"{len(df_yearly)}개 해")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{max_year}년")
col4.metric("상관계수 (전체)", f"{corr:.4f}")

st.divider()

# 5. 5개 구간 기울기(100년당 상승 폭) 지표 표시
st.subheader("🔥 최근 구간별 100년당 기온 상승 속도 (5개 구간 비교)")

cols = st.columns(5)
for i, res in enumerate(reg_results):
    cols[i].metric(
        label=res["label"],
        value=f"+{res['rate_100y']:.2f} °C / 100년",
        delta=f"전체 평균 대비 +{res['rate_100y'] - rate_all_100y:.2f} °C",
        delta_color="normal"
    )

st.caption(f"💡 전체 기간({start_year}~{max_year}년) 평균 상승 속도: +{rate_all_100y:.2f} °C / 100년")

st.divider()

# 6. 연도 선택 슬라이더 및 예측값 출력
selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

st.write(f"### 🎯 {selected_year}년 예상 평균기온 (구간별 추세선 기준)")

pred_cols = st.columns(5)
for i, res in enumerate(reg_results):
    pred_val = res["slope"] * (selected_year - 1908) + res["intercept"]
    pred_cols[i].markdown(
        f"""
        <div style="background-color: #f9f9f9; padding: 12px; border-radius: 8px; text-align: center; border-top: 4px solid {res['color']};">
            <span style="font-size: 0.85rem; color: #555;">{res['span']}년 추세</span><br>
            <strong style="font-size: 1.4rem; color: {res['color']};">{pred_val:.2f} °C</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# 7. Plotly 시각화 (5개 회귀선 표시)
fig = go.Figure()

# 실제 관측 데이터 산점도
fig.add_trace(
    go.Scatter(
        x=df_yearly["연도"],
        y=df_yearly["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7, color="#7f7f7f", opacity=0.6),
    )
)

x_range = np.arange(1900, 2101)

# 5개 구간 회귀선 추가
for res in reg_results:
    y_pred = res["slope"] * (x_range - 1908) + res["intercept"]
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_pred,
            mode="lines",
            name=f"{res['span']}년 추세 (+{res['rate_100y']:.2f}°C/100년)",
            line=dict(color=res["color"], width=2)
        )
    )

# 선택한 연도에서의 각 회귀선 상의 예측점 표시
for res in reg_results:
    pred_y = res["slope"] * (selected_year - 1908) + res["intercept"]
    fig.add_trace(
        go.Scatter(
            x=[selected_year],
            y=[pred_y],
            mode="markers",
            showlegend=False,
            marker=dict(size=10, color=res["color"], symbol="star")
        )
    )

fig.update_layout(
    title="서울 연도별 평균기온 및 10년 차이 5개 구간 회귀선 비교",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    xaxis=dict(tickformat="d"),
    hovermode="x unified",
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)
