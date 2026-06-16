"""
⚡ 변압기 화재 위험도 예측 대시보드
TransFireRisk | 날씨 빅데이터 콘테스트 2026
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

st.set_page_config(
    page_title="TransFireRisk — 변압기 화재 위험도 예측",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")

# ── 데이터 / 모델 로드 ────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(MODEL_DIR, "full_predictions.csv"))
    fi = pd.read_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index_col=0, header=0)
    fi.columns = ['중요도']
    return df, fi.sort_values('중요도', ascending=False)

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(MODEL_DIR, "xgb_transformer_fire.pkl"))

df, fi = load_data()
model  = load_model()

SIDO_LIST   = sorted(df['시도'].unique())
RISK_COLOR  = {'낮음':'#4CAF50','보통':'#FFC107','높음':'#FF9800','매우높음':'#F44336'}
SEASON_CODE = {12:4,1:4,2:4, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}

def risk_grade(v):
    if   v < 0.3: return "낮음"
    elif v < 0.7: return "보통"
    elif v < 1.2: return "높음"
    else:         return "매우높음"

FEATURES = [
    '월최고기온','월평균기온','월최저기온','월평균습도','월강수합계','월최대풍속',
    '월평균일교차','강수일수','월전3일평균기온','월전7일강수합계','월연속고온일수',
    '월','계절코드','시도코드','전년전기화재건수','전년변압기화재건수',
]

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ TransFireRisk")
    st.markdown("**변압기 화재 위험도 예측**")
    st.markdown("날씨 빅데이터 콘테스트 2026")
    st.markdown("---")
    st.markdown("""
    **데이터**
    - 소방청 화재발생정보 (2020~2024)
    - Open-Meteo 기상 데이터 (17개 시도)

    **모델**
    - XGBoost Regressor
    - 1,020행 (17시도×5년×12월)
    - 피처 16개

    **분석 단위**: 시도 × 월
    """)
    st.markdown("---")
    page = st.radio("페이지", ["📊 현황 분석", "🌦️ 날씨-화재 상관", "🔮 위험도 예측"])

# ═══════════════════════════════════════════════════════════
# 페이지 1: 현황 분석
# ═══════════════════════════════════════════════════════════
if page == "📊 현황 분석":
    st.title("📊 변압기 화재 발생 현황 (2020~2024)")

    # KPI 카드
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("5년 총 화재",     f"{int(df['변압기화재건수'].sum())}건")
    c2.metric("연평균",          f"{df.groupby('연도')['변압기화재건수'].sum().mean():.0f}건/년")
    c3.metric("최다 지역",       df.groupby('시도')['변압기화재건수'].sum().idxmax())
    c4.metric("최다 발생월",     f"{df.groupby('월')['변압기화재건수'].sum().idxmax()}월")
    c5.metric("2025 예상",       "68건 ↑", delta="전년 대비 +13%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        yearly = df.groupby('연도').agg(
            실제=('변압기화재건수','sum'), 예측=('예측건수','sum')
        ).reset_index()
        fig = go.Figure()
        fig.add_bar(x=yearly['연도'], y=yearly['실제'], name='실제 화재건수',
                    marker_color='#EF5350', opacity=0.85,
                    text=yearly['실제'], textposition='outside')
        fig.add_scatter(x=yearly['연도'], y=yearly['예측'], name='모델 예측',
                        mode='lines+markers', line=dict(color='#1565C0', width=2.5),
                        marker=dict(size=9))
        fig.update_layout(title='연도별 변압기 화재 건수 추이',
                          xaxis_title='연도', yaxis_title='건수',
                          height=360, legend=dict(x=0.01,y=0.99),
                          xaxis=dict(tickvals=yearly['연도']))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sido_sum = df.groupby('시도')['변압기화재건수'].sum().reset_index()
        sido_sum = sido_sum.sort_values('변압기화재건수')
        fig = px.bar(sido_sum, x='변압기화재건수', y='시도',
                     orientation='h', color='변압기화재건수',
                     color_continuous_scale='Reds',
                     title='시도별 변압기 화재 발생건수 (5년 합계)',
                     text='변압기화재건수')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # 히트맵
    st.markdown("#### 🗺️ 시도 × 월별 화재 발생 히트맵")
    pivot = df.groupby(['시도','월'])['변압기화재건수'].sum().unstack().fillna(0)
    fig = px.imshow(pivot, color_continuous_scale='YlOrRd', aspect='auto',
                    labels=dict(x='월', y='시도', color='화재건수'),
                    title='시도 × 월별 변압기 화재 (2020~2024 합계)')
    fig.update_xaxes(tickvals=list(range(1,13)),
                     ticktext=[f'{i}월' for i in range(1,13)])
    fig.update_layout(height=430)
    st.plotly_chart(fig, use_container_width=True)

    # 발화요인 분석
    st.markdown("#### ⚡ 발화요인별 발생 현황 (소방청 원본)")
    cause_data = {
        '미확인단락':13, '절연열화에 의한 단락':12, '과부하/과전류':12,
        '트래킹에 의한 단락':8, '접촉불량에 의한 단락':6, '노후':6,
        '기타(전기적요인)':5, '누전,지락':4, '기타(기계적요인)':3,
    }
    cdf = pd.DataFrame({'발화요인':list(cause_data.keys()), '건수':list(cause_data.values())})
    fig = px.bar(cdf.sort_values('건수'), x='건수', y='발화요인', orientation='h',
                 color='건수', color_continuous_scale='Oranges',
                 title='변압기 화재 발화요인 소분류 (2020~2024, 86건)')
    fig.update_layout(height=380, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# 페이지 2: 날씨-화재 상관
# ═══════════════════════════════════════════════════════════
elif page == "🌦️ 날씨-화재 상관":
    st.title("🌦️ 날씨 × 변압기 화재 상관 분석")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 변수 중요도 (XGBoost)")
        fi_df = fi.reset_index()
        fi_df.columns = ['변수','중요도']
        fig = px.bar(fi_df, x='중요도', y='변수', orientation='h',
                     color='중요도', color_continuous_scale='Blues',
                     title='예측 모델의 변수 중요도')
        fig.update_layout(height=430, yaxis={'categoryorder':'total ascending'},
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🔥 발화요인별 날씨 조건")
        cause_df = pd.DataFrame({
            '발화요인':  ['과부하/과전류','절연열화·단락','트래킹·단락','누전·지락','자연재해'],
            '평균기온(℃)':[13.5, 10.4, 16.7, 21.8, 23.4],
            '평균습도(%)':[ 74,   76,   82,   81,   89],
            '강수량(mm)': [0.6,  6.8, 11.6, 13.7, 17.2],
        })
        fig = go.Figure()
        fig.add_bar(name='평균기온(℃)', x=cause_df['발화요인'], y=cause_df['평균기온(℃)'], marker_color='#EF5350')
        fig.add_bar(name='평균습도(%)', x=cause_df['발화요인'], y=cause_df['평균습도(%)'], marker_color='#42A5F5')
        fig.add_bar(name='강수량(mm)',  x=cause_df['발화요인'], y=cause_df['강수량(mm)'],  marker_color='#66BB6A')
        fig.update_layout(barmode='group', height=430, xaxis_tickangle=-15,
                          title='발화요인별 화재 당일 평균 날씨 조건')
        st.plotly_chart(fig, use_container_width=True)

    # 산점도
    st.markdown("---")
    st.subheader("🔍 날씨 변수별 화재 발생 분포")
    col_a, col_b = st.columns([1,3])
    with col_a:
        sel_var = st.selectbox("날씨 변수", [
            '월최고기온','월평균습도','월강수합계','월평균일교차','강수일수','월전7일강수합계'
        ])
    with col_b:
        fig = px.scatter(df, x=sel_var, y='변압기화재건수',
                         color='계절', size_max=10, opacity=0.6,
                         hover_data=['시도','연도','월'],
                         color_discrete_map={'봄':'#66BB6A','여름':'#EF5350',
                                             '가을':'#FFA726','겨울':'#42A5F5'},
                         title=f'{sel_var} vs 변압기 화재 건수')
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    # 계절 분석
    col1, col2 = st.columns(2)
    with col1:
        season_df = df.groupby('계절')['변압기화재건수'].sum().reset_index()
        fig = px.pie(season_df, values='변압기화재건수', names='계절',
                     color='계절',
                     color_discrete_map={'봄':'#66BB6A','여름':'#EF5350',
                                         '가을':'#FFA726','겨울':'#42A5F5'},
                     title='계절별 변압기 화재 비율')
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        month_df = df.groupby('월')['변압기화재건수'].sum().reset_index()
        fig = px.bar(month_df, x='월', y='변압기화재건수',
                     color='변압기화재건수', color_continuous_scale='Reds',
                     title='월별 변압기 화재 발생건수')
        fig.update_xaxes(tickvals=list(range(1,13)), ticktext=[f'{i}월' for i in range(1,13)])
        fig.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # 날씨 3축 요약 박스
    st.markdown("---")
    st.markdown("### 🌡️ 날씨가 변압기를 공격하는 3가지 방식")
    b1, b2, b3 = st.columns(3)
    b1.info("**① 열 (고온)**\n\n여름 폭염 → 에어컨 수요 급증\n→ 변압기 **과부하/과전류**\n\n관련: 최고기온, 연속고온일수")
    b2.warning("**② 습기 (고습·강수)**\n\n비·습기 → 절연체 오염\n→ **트래킹·누전** 단락\n\n관련: 평균습도, 강수합계")
    b3.error("**③ 온도 변화 (일교차)**\n\n낮밤 기온차 → 팽창·수축 반복\n→ **절연열화** 균열\n\n관련: 기온일교차")

# ═══════════════════════════════════════════════════════════
# 페이지 3: 위험도 예측
# ═══════════════════════════════════════════════════════════
elif page == "🔮 위험도 예측":
    st.title("🔮 변압기 화재 위험도 실시간 예측")
    st.info("날씨 조건을 입력하면 해당 시도의 이번 달 변압기 화재 위험도를 예측합니다.")

    col_inp, col_out = st.columns([1,1])

    with col_inp:
        st.subheader("📥 조건 입력")
        pred_sido    = st.selectbox("시도 선택", SIDO_LIST)
        pred_month   = st.slider("월", 1, 12, 7)
        st.markdown("**기상 조건**")
        pred_maxtemp = st.slider("월 최고기온 (℃)",  -10.0, 42.0, 33.0, 0.5)
        pred_avgtemp = st.slider("월 평균기온 (℃)",  -15.0, 35.0, 27.0, 0.5)
        pred_mintemp = st.slider("월 최저기온 (℃)",  -20.0, 30.0, 22.0, 0.5)
        pred_humid   = st.slider("월 평균습도 (%)",    30.0,100.0, 82.0, 1.0)
        pred_rain    = st.slider("월 강수합계 (mm)",    0.0,600.0,150.0, 5.0)
        pred_wind    = st.slider("월 최대풍속 (km/h)",  0.0, 80.0, 20.0, 1.0)
        pred_range   = st.slider("월 기온일교차 (℃)",   0.0, 20.0,  6.0, 0.5)
        pred_rdays   = st.slider("강수일수 (일)",        0,   31,    12)

        # 지역 이력 (데이터 평균으로 자동 설정)
        elec_h = df[df['시도']==pred_sido]['전년전기화재건수'].mean()
        tr_h   = df[df['시도']==pred_sido]['전년변압기화재건수'].mean()
        sido_c = SIDO_LIST.index(pred_sido)

        X_input = pd.DataFrame([{
            '월최고기온':pred_maxtemp,'월평균기온':pred_avgtemp,'월최저기온':pred_mintemp,
            '월평균습도':pred_humid,'월강수합계':pred_rain,'월최대풍속':pred_wind,
            '월평균일교차':pred_range,'강수일수':pred_rdays,
            '월전3일평균기온':pred_avgtemp*0.95,'월전7일강수합계':pred_rain*0.23,
            '월연속고온일수':max(0,(pred_maxtemp-33)*3) if pred_maxtemp>33 else 0,
            '월':pred_month,'계절코드':SEASON_CODE.get(pred_month,1),
            '시도코드':sido_c,'전년전기화재건수':elec_h,'전년변압기화재건수':tr_h,
        }])

    with col_out:
        st.subheader("📊 예측 결과")
        pred_val = float(model.predict(X_input)[0])
        pred_val = max(0, pred_val)
        grade    = risk_grade(pred_val)
        color    = RISK_COLOR[grade]

        # 결과 카드
        st.markdown(f"""
        <div style='text-align:center;padding:28px;border-radius:14px;
                    background:{color}22;border:3px solid {color};margin-bottom:16px'>
            <div style='font-size:3.2rem;color:{color};font-weight:bold'>{grade}</div>
            <div style='font-size:1.3rem;color:#333;margin:6px 0'>
                예측 화재건수 <b>{pred_val:.2f}건</b>
            </div>
            <div style='color:#666'>{pred_sido} · {pred_month}월</div>
        </div>
        """, unsafe_allow_html=True)

        # 게이지
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_val,
            delta={'reference':
                   df[(df['시도']==pred_sido)&(df['월']==pred_month)]['예측건수'].mean()},
            title={'text':'위험도 (예측 화재건수)', 'font':{'size':14}},
            gauge={
                'axis':{'range':[0,2.5],'tickwidth':1},
                'bar':{'color':color},
                'steps':[
                    {'range':[0,0.3], 'color':'#E8F5E9'},
                    {'range':[0.3,0.7],'color':'#FFFDE7'},
                    {'range':[0.7,1.2],'color':'#FFF3E0'},
                    {'range':[1.2,2.5],'color':'#FFEBEE'},
                ],
                'threshold':{'line':{'color':'red','width':3},'value':pred_val}
            }
        ))
        fig.update_layout(height=270)
        st.plotly_chart(fig, use_container_width=True)

        # 위험 요인 해석
        st.subheader("💡 위험 요인")
        reasons = []
        if pred_maxtemp >= 33:
            reasons.append(f"🌡️ 최고기온 **{pred_maxtemp}℃** — 변압기 과부하 위험 구간")
        if pred_humid >= 80:
            reasons.append(f"💧 평균습도 **{pred_humid:.0f}%** — 트래킹·절연열화 발생 환경")
        if pred_rain >= 100:
            reasons.append(f"🌧️ 강수합계 **{pred_rain:.0f}mm** — 누전·지락 발생 환경")
        if pred_range <= 5:
            reasons.append(f"📊 기온일교차 **{pred_range}℃** — 지속 열 누적, 절연 피로")
        if pred_month in [7, 8]:
            reasons.append("📅 **7·8월** — 연중 최고 위험 시기 (전체 화재의 29%)")
        if not reasons:
            reasons.append("✅ 특이 위험 요인 없음 — 안전한 기상 조건")
        for r in reasons:
            st.markdown(f"- {r}")

    # 전국 비교
    st.markdown("---")
    st.subheader(f"🗺️ {pred_month}월 전국 시도별 위험도 비교")
    month_cmp = df[df['월']==pred_month].groupby('시도').agg(
        평균예측=('예측건수','mean')
    ).reset_index()
    month_cmp['등급'] = month_cmp['평균예측'].apply(risk_grade)
    month_cmp = month_cmp.sort_values('평균예측', ascending=False)

    # 현재 선택 시도 강조
    month_cmp['강조'] = month_cmp['시도'].apply(lambda x: '★ ' + x if x == pred_sido else x)

    fig = px.bar(month_cmp, x='시도', y='평균예측',
                 color='등급',
                 color_discrete_map=RISK_COLOR,
                 category_orders={'등급':['매우높음','높음','보통','낮음']},
                 title=f'{pred_month}월 시도별 변압기 화재 위험도 (2020~2024 평균)')
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

# ── 공통 푸터 ──────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "📌 **데이터**: 소방청 화재발생정보(2020~2024) · Open-Meteo 기상 API · 국가화재정보시스템  |  "
    "**모델**: XGBoost Regressor  |  **프로젝트**: TransFireRisk — 날씨 빅데이터 콘테스트 2026"
)
