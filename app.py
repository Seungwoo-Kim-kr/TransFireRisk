"""
TransFireRisk IMS v7.0
변압기 화재 위험 통합관리 시스템 | Transformer Fire Risk IMS
==============================================
모델 v3.0 (ensemble_v3.pkl):
  - XGB + RandomForest + LogisticRegression 소프트 보팅
  - 28개 피처 | Ensemble: XGB + RF + LR, 28 features
  - 평가 지표: PR-AUC / F2(β=2) / MCC  [F1 대신 불균형 특화]
  - 검증 ROC-AUC=0.640, PR-AUC=0.122, Recall@0.20=0.562
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, os, requests, math
from datetime import datetime, date, timedelta
from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                             f1_score, roc_curve, fbeta_score,
                             average_precision_score, matthews_corrcoef,
                             precision_recall_curve)

try:
    from openai import OpenAI
    _OPENAI_PKG = True
except ImportError:
    _OPENAI_PKG = False

# ── .env 로드 (python-dotenv) ─────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv 없어도 os.environ 은 동작함

_KMA_KEY    = os.environ.get("KMA_API_KEY",    "")
_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Plotly 호환 색상 변환 ──────────────────────────────────────
def _hex_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """'#RRGGBB' → 'rgba(R,G,B,alpha)' — scatterpolar 등 fillcolor 에 사용"""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

# ══════════════════════════════════════════════════════════════
#  다국어 지원 (i18n)
# ══════════════════════════════════════════════════════════════
_TRANS: dict = {
    # 탭
    'tab_dashboard': {'ko':'🏠 종합 현황판',    'en':'🏠 Dashboard'},
    'tab_region':    {'ko':'🗺️ 지역 상세',      'en':'🗺️ Regional Detail'},
    'tab_forecast':  {'ko':'📡 기상 예보',       'en':'📡 Weather Forecast'},
    'tab_risk':      {'ko':'🔬 복합위험 분석',   'en':'🔬 Risk Analysis'},
    'tab_insp':      {'ko':'📋 점검 관리',       'en':'📋 Inspection Mgmt'},
    'tab_model':     {'ko':'📊 이력·모델',       'en':'📊 History & Model'},
    # 사이드바
    'sidebar_title': {'ko':'경보 현황',          'en':'Alert Status'},
    'p1_label':      {'ko':'P1 즉시 점검',       'en':'P1 Immediate'},
    'p2_label':      {'ko':'P2 주의 지역',       'en':'P2 Caution'},
    'normal_label':  {'ko':'전 지역 정상',        'en':'All regions normal'},
    'view_month':    {'ko':'현황 기준 월',        'en':'Reference Month'},
    'risk_legend':   {'ko':'위험도 기준 (발화 확률)',    'en':'Risk Thresholds (Fire Prob.)'},
    'grade_vh':      {'ko':'매우높음 ≥ 40%',     'en':'Very High ≥ 40%'},
    'grade_h':       {'ko':'높음 25~40%',        'en':'High 25–40%'},
    'grade_m':       {'ko':'보통 15~25%',        'en':'Moderate 15–25%'},
    'grade_l':       {'ko':'낮음 < 15%',         'en':'Low < 15%'},
    # 현황판
    'dashboard_title': {'ko':'전국 변압기 화재 위험 현황',   'en':'National Transformer Fire Risk Status'},
    'data_basis':    {'ko':'2024년 기상 기반 모델 예측',     'en':'Based on 2024 weather data'},
    'avg_risk':      {'ko':'전국 평균 위험도',    'en':'National Avg Risk'},
    'p1_count':      {'ko':'P1 즉시 점검',       'en':'P1 Immediate'},
    'p2_count':      {'ko':'P2 주의 지역',       'en':'P2 Caution'},
    'est_fire':      {'ko':'고위험 예상 화재',    'en':'Est. High-Risk Fires'},
    'top_region':    {'ko':'최고 위험 지역',      'en':'Highest Risk Region'},
    'region_grid':   {'ko':'전국 발화 확률 현황', 'en':'National Fire Probability Map'},
    'risk_ranking':  {'ko':'종합위험 순위',       'en':'Risk Ranking'},
    'ml_vs_tfri':    {'ko':'ML 발화확률 vs TFRI 비교', 'en':'ML Prob. vs TFRI Comparison'},
    'weather_factors':{'ko':'현재 기상 위험 요인', 'en':'Current Weather Risk Factors'},
    # AI 브리핑
    'ai_briefing':   {'ko':'📋 AI 운영 브리핑',  'en':'📋 AI Operational Briefing'},
    'ai_gen':        {'ko':'🤖 브리핑 생성',     'en':'🤖 Generate Briefing'},
    'ai_refresh':    {'ko':'🔄 새로 생성',       'en':'🔄 Refresh'},
    'ai_no_key':     {'ko':'(사이드바에 GPT API Key를 입력하면 AI 브리핑이 자동 생성됩니다)',
                      'en':'(Enter GPT API Key in sidebar to auto-generate AI briefing)'},
    # 지역 상세
    'region_title':  {'ko':'지역 상세 조회',      'en':'Regional Detail'},
    'region_sel':    {'ko':'지역',               'en':'Region'},
    'month_sel':     {'ko':'월',                 'en':'Month'},
    # 점검 관리
    'insp_title':    {'ko':'점검 관리',           'en':'Inspection Management'},
    'insp_dl':       {'ko':'점검 계획표 CSV 다운로드', 'en':'Download Inspection Plan CSV'},
    # 모델 정보
    'model_title':   {'ko':'이력 분석 및 모델 정보', 'en':'History & Model Info'},
    'metric_note':   {
        'ko': '**왜 F1이 아닌 F2·MCC·PR-AUC를 사용하나요?**\n\n'
              '- 데이터 불균형(발화율 7.8%)에서 F1은 정밀도·재현율 동등 가중 → 탐지보다 오탐 감소에 유리\n'
              '- **F2 (β=2)**: 재현율을 2배 가중 — 화재를 놓치는 비용 > 오탐 비용\n'
              '- **MCC**: -1~+1 범위, 불균형 무관 가장 신뢰할 수 있는 단일 지표\n'
              '- **PR-AUC**: 임계값 없이 전체 Precision-Recall 성능 요약',
        'en': '**Why F2 / MCC / PR-AUC instead of F1?**\n\n'
              '- F1 equally weights precision & recall — penalizes false alarms equally\n'
              '- **F2 (β=2)**: Recall weighted 2× — missing a fire costs more than false alarm\n'
              '- **MCC**: Range −1 to +1, most robust metric for skewed class distributions\n'
              '- **PR-AUC**: Summarises precision-recall without threshold selection',
    },
    'model_compare': {'ko':'모델 성능 비교 (v1 기준 vs v3 개선)', 'en':'Model Performance: v1 Baseline vs v3 Improved'},
    'fi_title':      {'ko':'피처 중요도 (상위 15개)',  'en':'Feature Importance (Top 15)'},
    'roc_title':     {'ko':'ROC Curve (검증셋 2023~2024)', 'en':'ROC Curve (Validation 2023–2024)'},
}

def T(key: str) -> str:
    """현재 언어 기준으로 번역된 문자열 반환"""
    entry = _TRANS.get(key, {})
    lang_key = 'en' if st.session_state.get('lang','한국어') == 'English' else 'ko'
    return entry.get(lang_key, entry.get('ko', key))

def _is_en() -> bool:
    return st.session_state.get('lang','한국어') == 'English'

# ── 지역명 / 등급 / 계절 / 월 번역 ────────────────────────────
_SIDO_EN: dict = {
    '서울':'Seoul',  '부산':'Busan',    '대구':'Daegu',    '인천':'Incheon',
    '광주':'Gwangju','대전':'Daejeon',  '울산':'Ulsan',    '세종':'Sejong',
    '경기':'Gyeonggi','강원':'Gangwon', '충북':'Chungbuk', '충남':'Chungnam',
    '전북':'Jeonbuk','전남':'Jeonnam',  '경북':'Gyeongbuk','경남':'Gyeongnam',
    '제주':'Jeju',
}
_GRADE_EN: dict = {
    '낮음':'Low','보통':'Moderate','높음':'High','매우높음':'Very High',
}
_SEASON_EN: dict = {
    '봄':'Spring','여름':'Summer','가을':'Fall','겨울':'Winter',
}
_MONTH_SHORT: dict = {
    1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
    7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec',
}

def S(sido: str) -> str:
    """지역명 번역 (English 모드에서 영문명 반환)"""
    return _SIDO_EN.get(sido, sido) if _is_en() else sido

def G(grade: str) -> str:
    """위험도 등급 번역"""
    return _GRADE_EN.get(grade, grade) if _is_en() else grade

def Mn(month: int) -> str:
    """월 표시 (EN: Jan / KO: 1월)"""
    return _MONTH_SHORT.get(month, str(month)) if _is_en() else f"{month}월"

# ══════════════════════════════════════════════════════════════
#  AI 운영 브리핑
# ══════════════════════════════════════════════════════════════
def get_ai_briefing(baseline, month, year, api_key: str) -> str | None:
    """전국 현황을 GPT로 요약해 운영 브리핑 생성"""
    if not _OPENAI_PKG or not api_key:
        return None
    lang_key = 'en' if st.session_state.get('lang','한국어') == 'English' else 'ko'
    top3  = baseline.head(3)
    vh    = baseline[baseline['등급']=='매우높음']['시도'].tolist()
    hi    = baseline[baseline['등급']=='높음']['시도'].tolist()
    avg   = baseline['종합위험'].mean()
    top3_str = ', '.join(f"{r['시도']} {r['종합위험']:.0f}%" for _,r in top3.iterrows())

    if lang_key == 'en':
        prompt = f"""You are a power facility fire risk analyst (KEPCO).
Write a concise operational briefing (≤120 words) for {datetime(year,month,1).strftime('%B %Y')}.

Data:
- National avg combined risk: {avg:.1f}%
- P1 Immediate inspection: {', '.join(vh) if vh else 'None'}
- P2 Caution: {', '.join(hi) if hi else 'None'}
- Top 3 regions: {top3_str}

Format: 3 short paragraphs — ① overall risk level, ② priority regions & reason, ③ top 2 action items."""
    else:
        prompt = f"""당신은 KEPCO 전력설비 화재 위험 분석 전문가입니다.
{year}년 {month}월 운영 브리핑을 120자 이내로 작성하세요.

데이터:
- 전국 평균 종합위험도: {avg:.1f}%
- P1 즉시 점검 필요: {', '.join(vh) if vh else '없음'}
- P2 주의: {', '.join(hi) if hi else '없음'}
- 상위 3개 지역: {top3_str}

3개 단락으로 작성: ① 전반적 위험 수준, ② 우선 대응 지역과 이유, ③ 핵심 조치 2가지."""

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=250, temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception:
        return None

def rule_briefing(baseline, month) -> str:
    """GPT 없을 때 규칙 기반 요약"""
    lang_key = 'en' if st.session_state.get('lang','한국어') == 'English' else 'ko'
    vh = baseline[baseline['등급']=='매우높음']['시도'].tolist()
    hi = baseline[baseline['등급']=='높음']['시도'].tolist()
    avg = baseline['종합위험'].mean()
    if lang_key == 'en':
        level = "HIGH" if avg>=30 else "MODERATE" if avg>=20 else "LOW"
        parts = [f"**{datetime(2026,month,1).strftime('%B')} Risk Level: {level}** (avg {avg:.1f}%)."]
        if vh: parts.append(f"Immediate inspection required: **{', '.join(vh)}**.")
        if hi: parts.append(f"Monitor closely: {', '.join(hi)}.")
        if not vh and not hi: parts.append("No high-risk regions detected.")
        parts.append("Recommendation: Inspect cooling systems & insulation on high-load transformers.")
    else:
        level = "높음" if avg>=30 else "보통" if avg>=20 else "낮음"
        parts = [f"**{month}월 전국 위험 수준: {level}** (평균 {avg:.1f}%)."]
        if vh: parts.append(f"즉시 점검 권고: **{', '.join(vh)}**.")
        if hi: parts.append(f"주의 지역: {', '.join(hi)}.")
        if not vh and not hi: parts.append("고위험 지역 없음.")
        parts.append("권고: 고부하 변압기 냉각 설비 및 절연 상태 점검.")
    return "  \n".join(parts)

st.set_page_config(
    page_title="TransFireRisk IMS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""<style>
[data-testid="stMetricValue"]{font-size:1.55rem!important;font-weight:700!important}
[data-testid="stMetricLabel"]{font-size:0.8rem!important;color:#555!important}
.ims-header{background:linear-gradient(135deg,#0D1B4B 0%,#1565C0 100%);
  padding:18px 28px;border-radius:10px;margin-bottom:18px;
  display:flex;justify-content:space-between;align-items:center}
.alert-p1{background:#FFEBEE;border-left:5px solid #F44336;
  border-radius:6px;padding:10px 16px;margin:3px 0;color:#C62828;font-weight:600}
.alert-p2{background:#FFF3E0;border-left:5px solid #FF9800;
  border-radius:6px;padding:10px 16px;margin:3px 0;color:#E65100;font-weight:600}
.method-box{background:#E8EAF6;border-radius:8px;padding:14px;
  border-left:4px solid #3F51B5;margin:8px 0;font-size:0.87rem}
.ai-box{background:#F0F7FF;border-radius:10px;padding:16px;
  border-left:4px solid #1565C0;margin-top:10px}
.perf-delta-good{color:#2E7D32;font-weight:700}
.perf-delta-bad{color:#C62828;font-weight:700}
</style>""", unsafe_allow_html=True)

# ── 경로·날짜 ──────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")
TODAY     = date.today()
NOW       = datetime.now()
CUR_MONTH = TODAY.month
CUR_YEAR  = TODAY.year
MONTH_KR  = {i: f"{i}월" for i in range(1, 13)}

# ── 상수 ──────────────────────────────────────────────────────
SIDO_LIST = ['서울','부산','대구','인천','광주','대전','울산','세종',
             '경기','강원','충북','충남','전북','전남','경북','경남','제주']
SIDO_NX_NY = {
    '서울':(60,127),'부산':(98,76),'대구':(89,90),'인천':(55,124),
    '광주':(58,74), '대전':(67,100),'울산':(102,84),'세종':(66,103),
    '경기':(60,120),'강원':(73,134),'충북':(69,107),'충남':(68,100),
    '전북':(63,89), '전남':(51,67), '경북':(89,106),'경남':(91,77),'제주':(52,38),
}
SIDO_COORDS = {
    '서울':(37.5665,126.9780),'부산':(35.1796,129.0756),
    '대구':(35.8714,128.6014),'인천':(37.4563,126.7052),
    '광주':(35.1595,126.8526),'대전':(36.3504,127.3845),
    '울산':(35.5384,129.3114),'세종':(36.4800,127.2890),
    '경기':(37.2636,127.0286),'강원':(37.8813,127.7298),
    '충북':(36.6424,127.4890),'충남':(36.6588,126.6728),
    '전북':(35.8242,127.1480),'전남':(34.8161,126.4630),
    '경북':(36.5760,128.5056),'경남':(35.2373,128.6925),
    '제주':(33.4996,126.5312),
}
RISK_COLOR = {'낮음':'#4CAF50','보통':'#FFC107','높음':'#FF9800','매우높음':'#F44336'}
RISK_BG    = {'낮음':'#E8F5E9','보통':'#FFFDE7','높음':'#FFF3E0','매우높음':'#FFEBEE'}
RISK_TEXT  = {'낮음':'#2E7D32','보통':'#F57F17','높음':'#E65100','매우높음':'#C62828'}
RISK_EMOJI = {'낮음':'🟢','보통':'🟡','높음':'🟠','매우높음':'🔴'}
SEASON_CODE= {12:4,1:4,2:4,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}

# ── 위험도 등급 (확률 기반 v3.0) ─────────────────────────────
#   낮음<15% / 보통 15-25% / 높음 25-40% / 매우높음≥40%
def risk_grade(pct: float) -> str:
    if pct < 15:  return "낮음"
    elif pct < 25: return "보통"
    elif pct < 40: return "높음"
    else:          return "매우높음"

# ── 피처 엔지니어링 (모델 학습과 동일) ───────────────────────
TARGET = '변압기화재건수'
ORIG_FEATURES = [
    '월최고기온','월평균기온','월최저기온','월평균습도','월강수합계','월최대풍속',
    '월평균일교차','강수일수','월전3일평균기온','월전7일강수합계','월연속고온일수',
    '월','계절코드','시도코드','전년전기화재건수','전년변압기화재건수',
]
NEW_FEATURES = ORIG_FEATURES + [
    '열지수','열습도스트레스','강수고온지수','폭염습도',
    '과부하스트레스','강수습도','기온편차','습도편차','강수편차',
    '지역발화율','월_sin','월_cos',
]

def engineer_features(df_in: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    d['열지수']      = d['월최고기온'] * d['월평균습도'] / 100.0
    d['열습도스트레스'] = np.maximum(0, d['월최고기온']-30) * np.maximum(0,(d['월평균습도']-60)/40)
    d['강수고온지수']  = d['월강수합계'] * np.maximum(0, d['월평균기온']-15) / 100.0
    d['폭염습도']     = d['월연속고온일수'] * d['월평균습도'] / 100.0
    d['과부하스트레스'] = np.maximum(0, d['월최고기온']-33) ** 2
    d['강수습도']     = d['강수일수'] * d['월평균습도'] / 100.0

    clim = ref_df.groupby(['시도코드','월'])[
        ['월평균기온','월평균습도','월강수합계']].mean().reset_index()
    clim.columns = ['시도코드','월','clim_T','clim_RH','clim_Rain']
    d = d.merge(clim, on=['시도코드','월'], how='left')
    d['기온편차'] = d['월평균기온'] - d['clim_T'].fillna(d['월평균기온'].mean())
    d['습도편차'] = d['월평균습도'] - d['clim_RH'].fillna(d['월평균습도'].mean())
    d['강수편차'] = d['월강수합계'] - d['clim_Rain'].fillna(d['월강수합계'].mean())
    d.drop(columns=['clim_T','clim_RH','clim_Rain'], inplace=True, errors='ignore')

    fire_rate = ref_df.groupby('시도코드')[TARGET].apply(lambda x:(x>0).mean())
    d['지역발화율'] = d['시도코드'].map(fire_rate).fillna(fire_rate.mean())
    d['월_sin'] = np.sin(2*np.pi*d['월']/12)
    d['월_cos'] = np.cos(2*np.pi*d['월']/12)
    return d

# ── 데이터 로드 ───────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(MODEL_DIR, "full_predictions_final.csv"))
    ref = pd.read_csv(os.path.join(MODEL_DIR, "model_input.csv"))
    ref = ref[ref['연도'] <= 2022]          # 학습셋만 참조
    return df, ref

@st.cache_resource
def load_bundle():
    path = os.path.join(MODEL_DIR, "ensemble_v3.pkl")
    return joblib.load(path)

df, ref_df = load_data()
bundle     = load_bundle()

# ── 앙상블 예측 함수 ─────────────────────────────────────────
def predict_prob(row_dict: dict) -> float:
    """단일 행 딕셔너리 → 발화 확률(%) 반환"""
    row_df = pd.DataFrame([row_dict])
    row_eng = engineer_features(row_df, ref_df)
    X = row_eng[NEW_FEATURES].values
    X_s = bundle['scaler_lr'].transform(X)
    w = bundle['weights']
    prob = (w[0]*bundle['model_xgb'].predict_proba(X)[:,1] +
            w[1]*bundle['model_rf'].predict_proba(X)[:,1]  +
            w[2]*bundle['model_lr'].predict_proba(X_s)[:,1])
    return round(float(prob[0]) * 100, 1)

def row_from_weather(sido, month, maxT, avgT, minT, humid, rain,
                     wind, trange, rdays):
    elec_h = df[df['시도']==sido]['전년전기화재건수'].mean()
    tr_h   = df[df['시도']==sido]['전년변압기화재건수'].mean()
    return {
        '월최고기온':maxT,'월평균기온':avgT,'월최저기온':minT,
        '월평균습도':humid,'월강수합계':rain,'월최대풍속':wind,
        '월평균일교차':trange,'강수일수':rdays,
        '월전3일평균기온':avgT*0.95,'월전7일강수합계':rain*0.23,
        '월연속고온일수':max(0,(maxT-33)*2) if maxT>33 else 0,
        '월':month,'계절코드':SEASON_CODE.get(month,1),
        '시도코드':SIDO_LIST.index(sido),
        '전년전기화재건수':elec_h,'전년변압기화재건수':tr_h,
        TARGET:0,
    }

# ── TFRI 복합위험지수 ─────────────────────────────────────────
def compute_whi(maxT, humid, rain, trange):
    theta = max(0.0,(maxT-30)/10)
    phi   = max(0.0,(humid-70)/30)**1.5
    R     = min(math.log1p(rain/50)/math.log1p(200/50),1.0)
    fat   = max(0.0,1.0-trange/15)
    return round(min((0.32*theta+0.28*phi+0.25*R+0.15*fat)*100,100),1)

def compute_ida(avgT, maxT, humid):
    theta_hs = avgT+50.0+30.0*(maxT/40.0)**2
    try:   V = math.exp(15000/(273+98)-15000/(273+theta_hs))
    except: V = 1.0
    km = 1.0+2.0*max(0.0,(humid-70)/30)**2
    return round(min(V*km*20.0,100.0),1)

def compute_hri(sido, month):
    hist    = df[(df['시도']==sido)&(df['월']==month)]['변압기화재건수'].sum()
    nat_avg = df[df['월']==month].groupby('시도')['변압기화재건수'].sum().mean()
    n_yrs   = df['연도'].nunique()
    return round(min((hist/n_yrs)/(nat_avg+1e-6)*50,100),1)

def compute_tfri(sido, month, maxT, avgT, humid, rain, trange):
    whi = compute_whi(maxT, humid, rain, trange)
    ida = compute_ida(avgT, maxT, humid)
    hri = compute_hri(sido, month)
    return round(0.45*whi+0.35*ida+0.20*hri,1), whi, ida, hri

# ── 현황 기준 데이터 ─────────────────────────────────────────
def get_baseline(month: int):
    base = df[(df['연도']==2024)&(df['월']==month)].copy()
    if len(base) < 17:
        base = df[df['월']==month].groupby('시도').agg(
            발화확률=('발화확률','mean'), 변압기화재건수=('변압기화재건수','mean'),
            월최고기온=('월최고기온','mean'), 월평균기온=('월평균기온','mean'),
            월평균습도=('월평균습도','mean'), 월강수합계=('월강수합계','mean'),
            월평균일교차=('월평균일교차','mean'),
        ).reset_index()
    base['등급'] = base['발화확률'].apply(risk_grade)
    tfri_rows = []
    for _, r in base.iterrows():
        t,wh,id_,hr = compute_tfri(
            r['시도'], month, r.get('월최고기온',20), r.get('월평균기온',15),
            r.get('월평균습도',70), r.get('월강수합계',50), r.get('월평균일교차',8))
        tfri_rows.append({'시도':r['시도'],'TFRI':t,'WHI':wh,'IDA':id_,'HRI':hr})
    tfri_df = pd.DataFrame(tfri_rows)
    base = base.merge(tfri_df, on='시도', how='left')
    base['종합위험'] = ((base['발화확률'] + base['TFRI']) / 2).round(1)
    return base.sort_values('종합위험', ascending=False).reset_index(drop=True)

# ── KMA / Open-Meteo 예보 ─────────────────────────────────────
def _kma_base_time():
    h = NOW.hour; avail=[2,5,8,11,14,17,20,23]; past=[x for x in avail if x+1<=h]
    if not past: return (NOW-timedelta(days=1)).strftime('%Y%m%d'),'2300'
    return NOW.strftime('%Y%m%d'), f'{past[-1]:02d}00'

def _parse_pcp(v):
    s = str(v)
    if s in ('강수없음','','nan'): return 0.0
    if '미만' in s: return 0.5
    if '~' in s:
        try: a,b=s.replace('mm','').split('~'); return (float(a)+float(b))/2
        except: return 1.0
    try: return float(s.replace('mm','').strip())
    except: return 0.0

@st.cache_data(ttl=1800)
def fetch_kma(sido, api_key):
    if not api_key: return None,"KMA API KEY 없음"
    nx,ny = SIDO_NX_NY.get(sido,(60,127))
    bd,bt = _kma_base_time()
    try:
        # serviceKey는 URL에 직접 삽입해 이중인코딩 방지
        from urllib.parse import unquote
        key_dec = unquote(api_key)
        base_url = ("https://apis.data.go.kr/1360000"
                    "/VilageFcstInfoService2.0/getVilageFcst")
        full_url = (f"{base_url}?serviceKey={key_dec}"
                    f"&pageNo=1&numOfRows=1000&dataType=JSON"
                    f"&base_date={bd}&base_time={bt}&nx={nx}&ny={ny}")
        r = requests.get(full_url, timeout=12)
        if r.status_code == 500:
            return None, ("인증 실패 (HTTP 500) — 공공데이터포털에서 "
                          "'기상청 단기예보 조회서비스' 신청 및 활성화 필요")
        body = r.json().get('response',{})
        rc = body.get('header',{}).get('resultCode','')
        if rc not in ('00', ''):
            return None, f"API 오류 [{rc}] {body.get('header',{}).get('resultMsg','')}"
        body = body.get('body',{})
        if not body.get('totalCount',0): return None,f"데이터 없음({bd} {bt})"
        raw = pd.DataFrame(body['items']['item'])
        raw['날짜'] = pd.to_datetime(raw['fcstDate'])
        result = []
        for d_,grp in raw.groupby('날짜'):
            def g(cat): return pd.to_numeric(grp[grp['category']==cat]['fcstValue'],errors='coerce').dropna()
            t=g('TMP'); pcp=grp[grp['category']=='PCP']['fcstValue'].apply(_parse_pcp).sum()
            reh=g('REH'); wsd=g('WSD')
            tx=g('TMX'); tn=g('TMN')
            result.append({'날짜':d_,
                '최고기온':float(tx.iloc[0]) if len(tx) else (t.max() if len(t) else 20),
                '최저기온':float(tn.iloc[0]) if len(tn) else (t.min() if len(t) else 10),
                '평균기온':t.mean() if len(t) else 15,
                '강수량':pcp,'최대풍속':wsd.max() if len(wsd) else 0,
                '평균습도':reh.mean() if len(reh) else 70,'출처':'기상청(KMA)'})
        fc=pd.DataFrame(result); fc['일교차']=fc['최고기온']-fc['최저기온']
        return fc.head(3),None
    except Exception as e: return None,str(e)

@st.cache_data(ttl=3600)
def fetch_openmeteo(sido):
    lat,lon=SIDO_COORDS.get(sido,(37.5665,126.9780))
    try:
        r=requests.get("https://api.open-meteo.com/v1/forecast",
            params=dict(latitude=lat,longitude=lon,
                daily="temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                      "precipitation_sum,wind_speed_10m_max",
                hourly="relative_humidity_2m",forecast_days=14,timezone="Asia/Seoul"),
            timeout=10); r.raise_for_status()
        data=r.json()
        daily=pd.DataFrame({'날짜':pd.to_datetime(data['daily']['time']),
            '최고기온':data['daily']['temperature_2m_max'],
            '최저기온':data['daily']['temperature_2m_min'],
            '평균기온':data['daily']['temperature_2m_mean'],
            '강수량':data['daily']['precipitation_sum'],
            '최대풍속':data['daily']['wind_speed_10m_max']})
        daily['일교차']=daily['최고기온']-daily['최저기온']
        hr=pd.DataFrame({'dt':pd.to_datetime(data['hourly']['time']),
                         'hum':data['hourly']['relative_humidity_2m']})
        hr['date']=hr['dt'].dt.date
        dh=hr.groupby('date')['hum'].mean().reset_index()
        dh['날짜']=pd.to_datetime(dh['date'])
        daily=daily.merge(dh[['날짜','hum']],on='날짜',how='left')
        daily.rename(columns={'hum':'평균습도'},inplace=True)
        daily['평균습도']=daily['평균습도'].fillna(70)
        daily['출처']='Open-Meteo'
        return daily,None
    except Exception as e: return None,str(e)

def fetch_forecast(sido, kma_key=""):
    if kma_key:
        kdf,kerr=fetch_kma(sido,kma_key)
        if kdf is not None and len(kdf):
            om,_=fetch_openmeteo(sido)
            if om is not None:
                extra=om[om['날짜']>kdf['날짜'].max()].copy()
                extra['출처']='Open-Meteo'
                return pd.concat([kdf,extra],ignore_index=True),None,"기상청(3일)+Open-Meteo(4~14일)"
            return kdf,None,"기상청(KMA)"
    om,err=fetch_openmeteo(sido)
    return om,err,"Open-Meteo"

def compute_forecast_risk(fc_df, sido):
    """14일 예보 → 일별 발화확률(%) 및 TFRI"""
    scale=30/max(len(fc_df),1)
    rows=[]
    for i in range(len(fc_df)):
        row=fc_df.iloc[i]
        w7=fc_df.iloc[max(0,i-6):i+1]; w3=fc_df.iloc[max(0,i-2):i+1]
        rdict=row_from_weather(
            sido, row['날짜'].month,
            fc_df['최고기온'].max(), fc_df['평균기온'].mean(),
            fc_df['최저기온'].min(), fc_df['평균습도'].mean(),
            fc_df['강수량'].sum()*scale, fc_df['최대풍속'].max(),
            fc_df['일교차'].mean(), int((fc_df['강수량']>1).sum()*scale))
        rdict['월전3일평균기온']=w3['평균기온'].mean()
        rdict['월전7일강수합계']=w7['강수량'].sum()
        rdict['월연속고온일수']=max(0,(row['최고기온']-33)*2) if row['최고기온']>33 else 0
        rdict['월']=row['날짜'].month
        rdict['계절코드']=SEASON_CODE.get(row['날짜'].month,1)
        ml_prob=predict_prob(rdict)
        tfri_v,whi,ida,hri=compute_tfri(sido,row['날짜'].month,
            row['최고기온'],row['평균기온'],row['평균습도'],row['강수량'],row['일교차'])
        composite=round((ml_prob+tfri_v)/2,1)
        rows.append({'날짜':row['날짜'],'ML발화확률(%)':ml_prob,'TFRI(%)':tfri_v,
                     '종합위험(%)':composite,'WHI':whi,'IDA':ida,'HRI':hri,
                     '등급':risk_grade(composite),'최고기온':row['최고기온'],
                     '강수량':row['강수량'],'평균습도':row['평균습도'],
                     '출처':row.get('출처','')})
    return pd.DataFrame(rows)

# ── GPT 가이드 ────────────────────────────────────────────────
def get_ai_guide(sido, month, grade, ml_pct, tfri_pct, whi, ida, hri, reasons, api_key):
    if not _OPENAI_PKG or not api_key: return None,"API KEY 없음"
    try:
        client=OpenAI(api_key=api_key)
        prompt=f"""당신은 KEPCO 전력설비 화재 예방 전문가 (IEC 60076-7·CIGRE 기준 정통)입니다.

지역={sido}, {month}월, 위험등급={grade}
ML 발화확률={ml_pct:.1f}%  TFRI={tfri_pct:.1f}%  종합={(ml_pct+tfri_pct)/2:.1f}%
WHI={whi:.1f} / IDA={ida:.1f} / HRI={hri:.1f}
주요 위험 요인: {', '.join(reasons) if reasons else '없음'}

아래 4항목을 각각 2~3줄 실무 중심으로 작성하세요:
### 🔍 위험 메커니즘
### ✅ 즉시 점검 항목 (bullet 3~4개, 기준값 포함)
### 🔧 이번 달 정비 계획
### 🌦️ 기상 대응 조치"""
        resp=client.chat.completions.create(model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],max_tokens=600,temperature=0.25)
        return resp.choices[0].message.content,None
    except Exception as e: return None,str(e)

def rule_guide(grade, whi, ida):
    base={'낮음':'정기 점검 주기를 유지하세요.',
          '보통':'취약 설비를 집중 모니터링하세요.',
          '높음':'부하율 높은 변압기를 즉시 점검하고 비상 대응 체계를 가동하세요.',
          '매우높음':'즉각 특별 점검 및 24시간 감시 체계를 구축하세요.'}.get(grade,'')
    items=['절연저항 측정(기준:≥1GΩ@1kV)','부스바·단자 발열 점검(IR카메라)']
    if whi>60: items+=['냉각팬·방열기 작동 점검','부하 분산 및 과부하 차단기 검토']
    if ida>50: items+=['절연유 내전압 측정(기준:≥30kV/2.5mm)','흡습 브리더·실리카겔 교체']
    if whi>40: items+=['방수 패킹·케이블 관통부 실링 점검']
    return f"**{base}**\n\n**권장 점검 항목:**\n"+"\n".join(f"- {i}" for i in items[:5])

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    # 숨기기 버튼 (JS로 Streamlit 네이티브 collapse 트리거)
    st.markdown("""
<button onclick="
  var btn=window.parent.document.querySelector('[data-testid=collapsedControl]');
  if(btn)btn.click();
" style="width:100%;cursor:pointer;padding:6px;background:#f0f2f6;
  border:1px solid #d0d3d9;border-radius:6px;font-size:0.8rem;
  color:#555;margin-bottom:6px">◀ 사이드바 숨기기</button>
""", unsafe_allow_html=True)

    st.markdown(f"### ⚡ TransFireRisk IMS")
    st.markdown(f"`{CUR_YEAR}.{CUR_MONTH:02d}.{TODAY.day:02d}` · `{NOW.strftime('%H:%M')} KST`")
    lang_sel = st.radio("🌐", ["한국어", "English"], horizontal=True, key='lang')
    st.markdown("---")
    view_month=st.selectbox(f"📅 {T('view_month')}",list(range(1,13)),
        index=CUR_MONTH-1,format_func=lambda x:MONTH_KR[x])

    baseline=get_baseline(view_month)
    vh=baseline[baseline['등급']=='매우높음']['시도'].tolist()
    hi=baseline[baseline['등급']=='높음']['시도'].tolist()
    st.markdown(f"**🚨 {T('sidebar_title')}**")
    if vh: st.error(f"🔴 {T('p1_label')} ({len(vh)}): {', '.join(S(s) for s in vh)}")
    if hi: st.warning(f"🟠 {T('p2_label')} ({len(hi)}): {', '.join(S(s) for s in hi)}")
    if not vh and not hi: st.success(f"✅ {T('normal_label')}")
    st.markdown("---")
    # API 연결 상태 (한 줄 요약)
    _kma_dot = '🟢' if _KMA_KEY else '⚪'
    _ai_dot  = '🟢' if _OPENAI_KEY else '⚪'
    st.markdown(f"**🔑 API** &nbsp; {_kma_dot} KMA &nbsp;|&nbsp; {_ai_dot} AI")

# ── 전역 API 키 변수 (sidebar 이후에도 사용) ─────────────────
kma_key    = _KMA_KEY
openai_key = _OPENAI_KEY

# ── IMS 헤더 ─────────────────────────────────────────────────
st.markdown(f"""
<div class="ims-header">
  <div>
    <div style="color:#90CAF9;font-size:0.75rem;letter-spacing:3px">INTEGRATED MANAGEMENT SYSTEM</div>
    <div style="color:#fff;font-size:1.7rem;font-weight:800;line-height:1.1">⚡ TransFireRisk IMS</div>
    <div style="color:#BBDEFB;font-size:0.82rem;margin-top:2px">
      변압기 화재 위험 통합관리 시스템 · 발화 확률 모델 v3.0 · 날씨 빅데이터 콘테스트 2026
    </div>
  </div>
  <div style="text-align:right;color:#90CAF9">
    <div style="font-size:1.1rem;font-weight:600;color:#fff">{CUR_YEAR}년 {Mn(view_month)} 기준</div>
    <div style="font-size:0.8rem">앙상블(XGB+RF+LR) · 28 피처 · AUC 0.640</div>
    <div style="font-size:0.78rem;margin-top:2px">
      {'🟢 KMA 연결됨' if kma_key else '⚪ KMA 미설정'}
      {'  |  🤖 GPT 연결됨' if openai_key else ''}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ── 탭 (언어에 따라 동적 레이블) ─────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6=st.tabs([
    T('tab_dashboard'), T('tab_region'),  T('tab_forecast'),
    T('tab_risk'),      T('tab_insp'),    T('tab_model'),
])

# ═══════════════════════════════════════════════════════════════
# 탭 1  종합 현황판
# ═══════════════════════════════════════════════════════════════
with tab1:
    _p1_lbl = "Immediate Inspection" if _is_en() else "즉시 점검"
    _p2_lbl = "Caution"              if _is_en() else "주의"
    if vh: st.markdown(f'<div class="alert-p1">🔴 [P1] {_p1_lbl} — {"  ·  ".join(S(s) for s in vh)}</div>',
                       unsafe_allow_html=True)
    if hi: st.markdown(f'<div class="alert-p2">🟠 [P2] {_p2_lbl} — {"  ·  ".join(S(s) for s in hi)}</div>',
                       unsafe_allow_html=True)

    # ── AI 운영 브리핑 ─────────────────────────────────────────
    _brief_key = f"briefing_{view_month}_{st.session_state.get('lang','ko')}"
    with st.expander(f"**{T('ai_briefing')}**", expanded=True):
        col_b1, col_b2 = st.columns([4,1])
        with col_b2:
            if st.button(T('ai_refresh'), key="brief_refresh", use_container_width=True):
                st.session_state.pop(_brief_key, None)
        with col_b1:
            if _brief_key not in st.session_state:
                if openai_key:
                    with st.spinner("AI generating briefing..."):
                        brief = get_ai_briefing(baseline, view_month, CUR_YEAR, openai_key)
                    st.session_state[_brief_key] = brief or rule_briefing(baseline, view_month)
                else:
                    st.session_state[_brief_key] = rule_briefing(baseline, view_month)
            st.markdown(st.session_state[_brief_key])
            if not openai_key:
                st.caption(T('ai_no_key'))
    st.markdown("---")

    prev_base=get_baseline(view_month-1 if view_month>1 else 12)
    avg_now =baseline['종합위험'].mean(); avg_prev=prev_base['종합위험'].mean()
    exp_fire=baseline['발화확률'].apply(lambda x: 1 if x>=40 else 0.5 if x>=25 else 0).sum()

    k1,k2,k3,k4,k5=st.columns(5)
    k1.metric(T('avg_risk'),  f"{avg_now:.1f}%",
              delta=f"{avg_now-avg_prev:+.1f}%p {'MoM' if T('avg_risk')=='National Avg Risk' else '전월比'}")
    k2.metric(T('p1_count'),  f"{len(vh)} {'regions' if T('p1_count')=='P1 Immediate' else '개 지역'}",
              delta=', '.join(S(s) for s in vh) if vh else ("N/A" if _is_en() else "해당 없음"),
              delta_color="inverse" if vh else "off")
    k3.metric(T('p2_count'),  f"{len(hi)} {'regions' if T('p2_count')=='P2 Caution' else '개'}")
    k4.metric(T('est_fire'),  f"{exp_fire:.0f} {'fires' if T('est_fire')=='Est. High-Risk Fires' else '건'}",
              help="≥40%: 1건, 25~40%: 0.5건 기대값")
    k5.metric(T('top_region'), baseline.iloc[0]['시도'],
              delta=f"{baseline.iloc[0]['종합위험']:.0f}%")

    st.markdown("---")
    st.markdown(f"### 📍 {Mn(view_month)} {T('region_grid')} (ML + TFRI)")

    rows_5=[baseline.iloc[i:i+5] for i in range(0,len(baseline),5)]
    for grp in rows_5:
        cols=st.columns(5)
        for ci,(_,reg) in enumerate(grp.iterrows()):
            g=reg['등급']; ml=reg['발화확률']; tf=reg['TFRI']; v=reg['종합위험']
            with cols[ci]:
                st.markdown(f"""
                <div style="background:{RISK_BG[g]};border:2px solid {RISK_COLOR[g]};
                  border-radius:10px;padding:12px 6px;text-align:center;min-height:110px">
                  <div style="font-size:0.95rem;font-weight:700;color:#333">{S(reg['시도'])}</div>
                  <div style="font-size:2.0rem;font-weight:900;color:{RISK_TEXT[g]};
                    line-height:1.1;margin:2px 0">{v:.0f}%</div>
                  <div style="font-size:0.68rem;color:#888">ML {ml:.0f}%  TFRI {tf:.0f}%</div>
                  <div style="font-size:0.75rem;color:{RISK_TEXT[g]}">{RISK_EMOJI[g]} {G(g)}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a,col_b=st.columns(2)
    with col_a:
        st.markdown("#### 📊 종합위험 순위")
        fig=px.bar(baseline,x='종합위험',y='시도',orientation='h',
            color='등급',color_discrete_map=RISK_COLOR,
            category_orders={'등급':['매우높음','높음','보통','낮음']},
            text=baseline['종합위험'].apply(lambda x:f"{x:.0f}%"))
        fig.update_traces(textposition='outside')
        fig.update_layout(height=440,yaxis={'categoryorder':'total ascending'},
            xaxis_title='종합위험도 (%)',yaxis_title='',
            margin=dict(l=0,r=60,t=20,b=20))
        st.plotly_chart(fig,use_container_width=True)

    with col_b:
        st.markdown("#### 📈 ML 발화확률 vs TFRI 비교")
        comp=baseline[['시도','발화확률','TFRI','종합위험']].sort_values('종합위험',ascending=True)
        fig=go.Figure()
        fig.add_bar(x=comp['발화확률'],y=comp['시도'],name='ML 발화확률',
            marker_color='#1565C0',opacity=0.75,orientation='h')
        fig.add_bar(x=comp['TFRI'],y=comp['시도'],name='TFRI 복합지수',
            marker_color='#E91E63',opacity=0.75,orientation='h')
        fig.add_scatter(x=comp['종합위험'],y=comp['시도'],mode='markers',
            marker=dict(color='#333',size=8,symbol='diamond'),name='종합위험',orientation='h')
        fig.update_layout(barmode='group',height=440,xaxis_title='위험도(%)',
            yaxis_title='',margin=dict(l=0,r=10,t=20,b=20))
        st.plotly_chart(fig,use_container_width=True)

    if '월최고기온' in baseline.columns:
        st.markdown("---")
        w1,w2,w3,w4=st.columns(4)
        w1.metric("🌡️ 고온 지역 (≥33℃)",f"{(baseline['월최고기온']>=33).sum()}개")
        w2.metric("💧 고습 지역 (≥80%)", f"{(baseline['월평균습도']>=80).sum()}개")
        w3.metric("🌧️ 강수 지역 (≥100mm)",f"{(baseline['월강수합계']>=100).sum()}개")
        w4.metric("🔴 TFRI 최고",f"{baseline.iloc[0]['시도']} {baseline.iloc[0]['TFRI']:.0f}%")

# ═══════════════════════════════════════════════════════════════
# 탭 2  지역 상세
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🗺️ " + ("Regional Detail" if _is_en() else "지역 상세 조회"))
    c1,c2=st.columns(2)
    with c1: sel_sido=st.selectbox("지역" if not _is_en() else "Region", SIDO_LIST)
    with c2: sel_month=st.selectbox("월" if not _is_en() else "Month", list(range(1,13)),
        index=CUR_MONTH-1,format_func=lambda x:MONTH_KR[x])

    row=df[(df['시도']==sel_sido)&(df['연도']==2024)&(df['월']==sel_month)]
    if len(row)==0: row=df[(df['시도']==sel_sido)&(df['월']==sel_month)].tail(1)
    r=row.iloc[0] if len(row) else pd.Series()
    ml_pct=float(r.get('발화확률',20))
    t,whi,ida,hri=compute_tfri(sel_sido,sel_month,
        r.get('월최고기온',20),r.get('월평균기온',15),
        r.get('월평균습도',70),r.get('월강수합계',50),r.get('월평균일교차',8))
    composite=round((ml_pct+t)/2,1)
    grade=risk_grade(composite)
    color=RISK_COLOR[grade]

    # ── AI 분석 — 상단 자동 생성 ──────────────────────────────
    _ai_det_key = f"ai_det_{sel_sido}_{sel_month}_{st.session_state.get('lang','ko')}"
    _ai_det_ttl = "🤖 AI 분석" if not _is_en() else "🤖 AI Analysis"
    with st.expander(f"**{_ai_det_ttl}**", expanded=True):
        _col_det1, _col_det2 = st.columns([4,1])
        with _col_det2:
            if st.button("🔄" + (" Refresh" if _is_en() else " 새로 생성"),
                         key="det_ai_refresh", use_container_width=True):
                st.session_state.pop(_ai_det_key, None)
        with _col_det1:
            if _ai_det_key not in st.session_state:
                _reasons_det = []
                if r.get('월최고기온',0) >= 33: _reasons_det.append(f"최고기온 {r['월최고기온']:.1f}℃")
                if r.get('월평균습도',0) >= 80:  _reasons_det.append(f"평균습도 {r['월평균습도']:.0f}%")
                if r.get('월강수합계',0) >= 100: _reasons_det.append(f"강수합계 {r['월강수합계']:.0f}mm")
                if sel_month in [7,8]:           _reasons_det.append("7·8월 고위험 시기")
                if openai_key:
                    with st.spinner("AI 분석 생성 중..." if not _is_en() else "Generating..."):
                        _at, _ae = get_ai_guide(sel_sido, sel_month, grade, ml_pct, t,
                                                whi, ida, hri, _reasons_det, openai_key)
                    st.session_state[_ai_det_key] = _at if not _ae else rule_guide(grade,whi,ida)
                else:
                    st.session_state[_ai_det_key] = rule_guide(grade, whi, ida)
            st.markdown(f'<div class="ai-box">{st.session_state[_ai_det_key]}</div>',
                        unsafe_allow_html=True)
            if not openai_key: st.caption(T('ai_no_key'))
    st.markdown("---")

    col_big,col_right=st.columns([1,1])
    with col_big:
        st.markdown(f"""
        <div style="background:{RISK_BG[grade]};border:3px solid {color};
          border-radius:14px;padding:28px;text-align:center">
          <div style="color:#555;font-size:0.9rem">{S(sel_sido)} · {Mn(sel_month)} (2024)</div>
          <div style="font-size:4.5rem;font-weight:900;color:{RISK_TEXT[grade]};
            line-height:1.0;margin:4px 0">{composite:.0f}%</div>
          <div style="font-size:1.4rem;font-weight:700;color:{RISK_TEXT[grade]}">
            {RISK_EMOJI[grade]} {grade}
          </div>
          <div style="margin-top:10px;display:flex;justify-content:center;gap:24px">
            <span style="font-size:0.85rem;color:#555">발화확률 <b style="color:{color}">{ml_pct:.0f}%</b></span>
            <span style="font-size:0.85rem;color:#555">TFRI <b style="color:{color}">{t:.0f}%</b></span>
          </div>
        </div>""",unsafe_allow_html=True)

        fig=go.Figure(go.Scatterpolar(
            r=[whi,ida,hri,whi],
            theta=['WHI<br>기상위험','IDA<br>절연열화','HRI<br>이력위험','WHI<br>기상위험'],
            fill='toself',fillcolor=_hex_rgba(color,0.2),line=dict(color=color,width=2.5)))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),
            height=270,margin=dict(l=30,r=30,t=20,b=20),showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

        if len(row):
            m1,m2,m3=st.columns(3)
            m1.metric("최고기온",f"{r.get('월최고기온','-'):.1f}℃")
            m2.metric("평균습도",f"{r.get('월평균습도','-'):.0f}%")
            m3.metric("강수합계",f"{r.get('월강수합계','-'):.0f}mm")

    with col_right:
        monthly=df[(df['시도']==sel_sido)&(df['연도']==2024)].groupby('월').agg(
            ML=('발화확률','mean'), 실제=('변압기화재건수','mean'),
            maxT=('월최고기온','mean'),avgT=('월평균기온','mean'),
            humid=('월평균습도','mean'),rain=('월강수합계','mean'),
            trange=('월평균일교차','mean')).reset_index()
        monthly['TFRI_m']=monthly.apply(
            lambda x:compute_tfri(sel_sido,int(x['월']),x['maxT'],x['avgT'],
                                   x['humid'],x['rain'],x['trange'])[0],axis=1)
        monthly['종합']  =((monthly['ML']+monthly['TFRI_m'])/2).round(1)

        fig=go.Figure()
        fig.add_hrect(y0=0,y1=15,fillcolor='#E8F5E9',opacity=0.3,line_width=0)
        fig.add_hrect(y0=15,y1=25,fillcolor='#FFFDE7',opacity=0.3,line_width=0)
        fig.add_hrect(y0=25,y1=115,fillcolor='#FFEBEE',opacity=0.3,line_width=0)
        fig.add_bar(x=monthly['월'],y=monthly['ML'],marker_color='#90CAF9',
            opacity=0.7,name='ML 발화확률',width=0.35,offset=-0.2)
        fig.add_bar(x=monthly['월'],y=monthly['TFRI_m'],marker_color='#F48FB1',
            opacity=0.7,name='TFRI',width=0.35,offset=0.15)
        fig.add_scatter(x=monthly['월'],y=monthly['종합'],mode='lines+markers',
            line=dict(color='#1565C0',width=2.5),marker=dict(size=7),name='종합')
        fig.add_scatter(x=[sel_month],y=[composite],mode='markers',
            marker=dict(size=14,color='black',symbol='star'),name='현재')
        fm=monthly[monthly['실제']>0]
        if len(fm):
            fig.add_scatter(x=fm['월'],y=fm['종합']+5,mode='markers+text',
                text='🔥',textfont=dict(size=14),marker=dict(size=1,color='red'),
                name='과거 화재')
        fig.update_layout(title=f'{S(sel_sido)} ' + ('Monthly Risk (2024)' if _is_en() else '월별 위험도 (2024)'),barmode='overlay',
            xaxis=dict(tickvals=list(range(1,13)),
                       ticktext=[f'{i}월' for i in range(1,13)]),
            yaxis=dict(title='발화 확률 (%)',range=[0,120]),
            height=380,margin=dict(l=0,r=10,t=40,b=30))
        st.plotly_chart(fig,use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# 탭 3  기상 예보
# ═══════════════════════════════════════════════════════════════
with tab3:
    _t3_title = "Weather Forecast Risk Prediction" if _is_en() else "기상 예보 기반 위험도 예측"
    st.markdown(f"## 📡 {_t3_title}")
    st.caption("🏛️ KMA (기상청) · 🌐 Open-Meteo — " +
               ("side-by-side source comparison · ML Ensemble v3 + TFRI" if _is_en()
                else "두 예보 소스 비교 · ML 앙상블 v3 + TFRI 복합지수"))

    cf1,cf2=st.columns([1,3])
    with cf1:
        fc_sido=st.selectbox("지역" if not _is_en() else "Region", SIDO_LIST, key="fc_sido")
        fc_btn=st.button("🔄 " + ("Load Forecast" if _is_en() else "예보 불러오기"),
                         use_container_width=True)

    # ── 두 소스 독립 fetch ────────────────────────────────────────
    _need_reload = (fc_btn or
                    'fc_kma' not in st.session_state or
                    st.session_state.get('_fc_sido') != fc_sido)
    if _need_reload:
        with st.spinner(f"{S(fc_sido)} " + ("forecast loading..." if _is_en() else "예보 로딩...")):
            kma_df, kma_err = fetch_kma(fc_sido, kma_key)
            om_df,  om_err  = fetch_openmeteo(fc_sido)
            st.session_state.update({
                'fc_kma': kma_df, 'fc_kma_err': kma_err,
                'fc_om':  om_df,  'fc_om_err':  om_err,
                '_fc_sido': fc_sido,
            })

    kma_df  = st.session_state.get('fc_kma')
    kma_err = st.session_state.get('fc_kma_err')
    om_df   = st.session_state.get('fc_om')
    om_err  = st.session_state.get('fc_om_err')

    if kma_df is None and om_df is None:
        st.error("예보 로드 실패 — KMA: " + str(kma_err) + " / Open-Meteo: " + str(om_err))
    else:
        # ── 소스별 위험도 계산 ─────────────────────────────────────
        risk_kma = compute_forecast_risk(kma_df, fc_sido) if kma_df is not None else None
        risk_om  = compute_forecast_risk(om_df,  fc_sido) if om_df  is not None else None

        # ── 상단 알림 배너 (두 소스 중 높은 쪽 기준) ──────────────
        _all_risk = pd.concat([r for r in [risk_kma, risk_om] if r is not None])
        _max_any  = _all_risk['종합위험(%)'].max()
        _max_day  = _all_risk.loc[_all_risk['종합위험(%)'].idxmax(), '날짜'].strftime('%m/%d')
        if _max_any >= 40:
            st.markdown(f'<div class="alert-p1">🔴 {S(fc_sido)} — '
                        + ("Very High risk expected in forecast period "
                           if _is_en() else "예보 기간 내 매우높음 예상 ")
                        + f"(max {_max_any:.0f}%, {_max_day})</div>", unsafe_allow_html=True)
        elif _max_any >= 25:
            st.markdown(f'<div class="alert-p2">🟠 {S(fc_sido)} — '
                        + ("High risk zone in forecast period "
                           if _is_en() else "예보 기간 내 높음 구간 진입 ")
                        + f"(max {_max_any:.0f}%, {_max_day})</div>", unsafe_allow_html=True)

        # ── AI 분석 — 배너 직후 자동 생성 ────────────────────────
        _base_risk = risk_om if risk_om is not None else risk_kma
        _base_df   = om_df   if om_df   is not None else kma_df
        _ai_fc_key = f"ai_fc_{fc_sido}_{st.session_state.get('lang','ko')}"
        _ai_title  = "🤖 AI 분석" if not _is_en() else "🤖 AI Analysis"
        with st.expander(f"**{_ai_title}**", expanded=True):
            _col_fc1, _col_fc2 = st.columns([4,1])
            with _col_fc2:
                if st.button("🔄" + (" Refresh" if _is_en() else " 새로 생성"),
                             key="fc_ai_refresh", use_container_width=True):
                    st.session_state.pop(_ai_fc_key, None)
            with _col_fc1:
                if _base_risk is not None and _ai_fc_key not in st.session_state:
                    _peak = _base_risk.loc[_base_risk['종합위험(%)'].idxmax()]
                    _fc_r = ([f"최고기온 {_base_df['최고기온'].max():.1f}℃"]
                             if _base_df['최고기온'].max() >= 33 else [])
                    if _base_df['평균습도'].mean() >= 80:
                        _fc_r.append(f"평균습도 {_base_df['평균습도'].mean():.0f}%")
                    if _base_df['강수량'].sum() >= 50:
                        _fc_r.append(f"강수합계 {_base_df['강수량'].sum():.0f}mm")
                    _pk_month = int(_peak['날짜'].month if hasattr(_peak['날짜'], 'month')
                                    else pd.to_datetime(_peak['날짜']).month)
                    if openai_key:
                        with st.spinner("AI 분석 생성 중..." if not _is_en() else "Generating..."):
                            _at, _ae = get_ai_guide(fc_sido, _pk_month, _peak['등급'],
                                _peak['ML발화확률(%)'], _peak['TFRI(%)'],
                                _peak['WHI'], _peak['IDA'], _peak['HRI'], _fc_r, openai_key)
                        st.session_state[_ai_fc_key] = (
                            _at if not _ae
                            else rule_guide(_peak['등급'], _peak['WHI'], _peak['IDA']))
                    else:
                        st.session_state[_ai_fc_key] = rule_guide(
                            _peak['등급'], _peak['WHI'], _peak['IDA'])
                if _ai_fc_key in st.session_state:
                    st.markdown(f'<div class="ai-box">{st.session_state[_ai_fc_key]}</div>',
                                unsafe_allow_html=True)
                    if not openai_key: st.caption(T('ai_no_key'))
                else:
                    st.info("지역을 선택하고 **예보 불러오기** 를 클릭하세요."
                            if not _is_en() else "Select a region and click **Load Forecast**.")

        st.markdown("---")

        # ══════════════════════════════════════════════════════════
        # 두 패널 나란히
        # ══════════════════════════════════════════════════════════
        def _risk_panel(src_label, icon, fc_df_src, risk_src, color_main, color_tfri):
            """공통 위험도 패널 렌더러"""
            if risk_src is None or fc_df_src is None:
                st.warning(f"{icon} {src_label} — " +
                           ("API key not set" if "KMA" in src_label and not kma_key
                            else "fetch failed"))
                return
            mx = risk_src['종합위험(%)'].max()
            mx_d = risk_src.loc[risk_src['종합위험(%)'].idxmax(),'날짜'].strftime('%m/%d')
            hi_d = (risk_src['종합위험(%)'] >= 25).sum()
            tod  = risk_src.iloc[0]

            st.markdown(f"#### {icon} {src_label}")
            n_days = len(fc_df_src)
            _lbl_days = f"{n_days}-day" if _is_en() else f"{n_days}일"
            st.caption(f"{_lbl_days} forecast")

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Today" if _is_en() else "오늘", f"{tod['종합위험(%)']:.1f}%")
            m2.metric("Peak" if _is_en() else "최고", f"{mx:.1f}%", help=mx_d)
            m3.metric("Avg" if _is_en() else "평균", f"{risk_src['종합위험(%)'].mean():.1f}%")
            m4.metric("High days" if _is_en() else "고위험일",
                      f"{hi_d}" + (" days" if _is_en() else "일"))

            # 위험도 차트
            fig = go.Figure()
            fig.add_hrect(y0=0,  y1=15,  fillcolor='#E8F5E9', opacity=0.25, line_width=0)
            fig.add_hrect(y0=15, y1=25,  fillcolor='#FFFDE7', opacity=0.25, line_width=0)
            fig.add_hrect(y0=25, y1=115, fillcolor='#FFEBEE', opacity=0.25, line_width=0)
            fig.add_bar(x=risk_src['날짜'], y=risk_src['ML발화확률(%)'], name='ML',
                        marker_color=color_main, opacity=0.55, width=0.35, offset=-0.2)
            fig.add_bar(x=risk_src['날짜'], y=risk_src['TFRI(%)'], name='TFRI',
                        marker_color=color_tfri, opacity=0.55, width=0.35, offset=0.15)
            fig.add_scatter(x=risk_src['날짜'], y=risk_src['종합위험(%)'],
                            mode='lines+markers', line=dict(color=color_main, width=2.5),
                            marker=dict(size=7, color=[RISK_COLOR[g] for g in risk_src['등급']]),
                            name='종합위험')
            fig.update_layout(height=280, barmode='overlay',
                              yaxis=dict(range=[0,115], title='%'),
                              title=('Combined Risk' if _is_en() else '종합 위험도'),
                              margin=dict(l=0,r=10,t=36,b=20), showlegend=True,
                              legend=dict(orientation='h',y=-0.15))
            st.plotly_chart(fig, use_container_width=True)

            # 기상 차트
            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                subplot_titles=['기온(℃)' if not _is_en() else 'Temp(℃)',
                                '강수(mm)' if not _is_en() else 'Rain(mm)'],
                vertical_spacing=0.18)
            fig2.add_scatter(x=fc_df_src['날짜'], y=fc_df_src['최고기온'], name='최고',
                line=dict(color='#EF5350',width=2), mode='lines+markers', row=1, col=1)
            fig2.add_scatter(x=fc_df_src['날짜'], y=fc_df_src['평균기온'], name='평균',
                line=dict(color='#FF9800',dash='dot'), mode='lines', row=1, col=1)
            fig2.add_scatter(x=fc_df_src['날짜'], y=fc_df_src['최저기온'], name='최저',
                line=dict(color='#42A5F5',width=2), mode='lines+markers', row=1, col=1)
            fig2.add_bar(x=fc_df_src['날짜'], y=fc_df_src['강수량'],
                marker_color='#42A5F5', opacity=0.7, name='강수', row=2, col=1)
            fig2.update_layout(height=260, margin=dict(l=0,r=10,t=30,b=20), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        col_kma, col_om = st.columns(2)
        with col_kma:
            if not kma_key:
                st.markdown("#### 🏛️ KMA (기상청) 3일 예보")
                st.info("⚪ KMA API 키 미설정\n`.env`에 `KMA_API_KEY`를 추가하면 활성화됩니다.")
            else:
                _risk_panel("KMA (기상청) 3일", "🏛️", kma_df, risk_kma,
                            '#1565C0', '#E91E63')
        with col_om:
            _risk_panel("Open-Meteo 14일", "🌐", om_df, risk_om,
                        '#2E7D32', '#F57F17')

        # ══════════════════════════════════════════════════════════
        # 비교 차트 (중복 기간)
        # ══════════════════════════════════════════════════════════
        if risk_kma is not None and risk_om is not None:
            st.markdown("---")
            _cmp_title = "Source Comparison — Overlapping Period" if _is_en() else "소스 비교 — 중복 기간"
            st.markdown(f"#### 🔀 {_cmp_title}")
            overlap_dates = set(risk_kma['날짜'].dt.date) & set(risk_om['날짜'].dt.date)
            if overlap_dates:
                ck = risk_kma[risk_kma['날짜'].dt.date.isin(overlap_dates)].copy()
                co = risk_om [risk_om ['날짜'].dt.date.isin(overlap_dates)].copy()
                fig_cmp = go.Figure()
                fig_cmp.add_scatter(x=ck['날짜'], y=ck['종합위험(%)'], mode='lines+markers',
                    name='KMA', line=dict(color='#1565C0',width=2.5),
                    marker=dict(size=8,symbol='circle'))
                fig_cmp.add_scatter(x=co['날짜'], y=co['종합위험(%)'], mode='lines+markers',
                    name='Open-Meteo', line=dict(color='#2E7D32',width=2.5,dash='dash'),
                    marker=dict(size=8,symbol='diamond'))
                fig_cmp.add_hrect(y0=25,y1=115,fillcolor='#FFEBEE',opacity=0.18,line_width=0)
                fig_cmp.update_layout(height=260,
                    yaxis=dict(range=[0,max(ck['종합위험(%)'].max(),co['종합위험(%)'].max())+15,],
                               title='종합위험도 (%)'),
                    margin=dict(l=0,r=10,t=10,b=20),
                    legend=dict(orientation='h',y=1.12))
                st.plotly_chart(fig_cmp, use_container_width=True)

                diff = (ck['종합위험(%)'].values - co['종합위험(%)'].values)
                _diff_lbl = "KMA vs Open-Meteo avg gap" if _is_en() else "KMA vs Open-Meteo 평균 차이"
                st.caption(f"{_diff_lbl}: **{diff.mean():+.1f}%p**  "
                           f"(max gap {abs(diff).max():.1f}%p on "
                           f"{ck.iloc[abs(diff).argmax()]['날짜'].strftime('%m/%d')})")
            else:
                st.info("중복 예보 기간 없음 (KMA 3일 예보 로드 필요)")

        # ── 상세 테이블 ────────────────────────────────────────────
        st.markdown("---")
        if _base_risk is not None:
            disp = _base_risk[['날짜','종합위험(%)','ML발화확률(%)','TFRI(%)','등급',
                                '최고기온','강수량','평균습도','출처']].copy()
            disp['날짜'] = disp['날짜'].dt.strftime('%m/%d(%a)')
            disp.index  = range(1, len(disp)+1)
            st.dataframe(disp, use_container_width=True, height=280)

# ═══════════════════════════════════════════════════════════════
# 탭 4  복합위험 분석
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🔬 복합위험 분석")
    st.markdown("""<div class="method-box">
<b>종합위험도 = (ML 발화확률 + TFRI) / 2</b><br>
• <b>ML 발화확률</b>: 앙상블(XGB+RF+LR) 이진 분류 — P(화재 발생) × 100%<br>
• <b>TFRI</b>: WHI(45%) + IDA(35%) + HRI(20%) — IEC 60076-7 / CIGRE WG A2.49<br>
두 방법의 약점을 서로 보완: ML은 패턴 기반, TFRI는 물리 법칙 기반
</div>""",unsafe_allow_html=True)

    ca1,ca2=st.columns(2)
    with ca1: an_sido=st.selectbox("분석 지역",SIDO_LIST,key="an_sido")
    with ca2: an_month=st.selectbox("분석 월",list(range(1,13)),
        index=CUR_MONTH-1,format_func=lambda x:MONTH_KR[x])

    an_row=df[(df['시도']==an_sido)&(df['연도']==2024)&(df['월']==an_month)]
    if len(an_row)==0: an_row=df[(df['시도']==an_sido)&(df['월']==an_month)].tail(1)
    an_r=an_row.iloc[0] if len(an_row) else pd.Series()
    an_ml=float(an_r.get('발화확률',20))
    tfri_v,whi_v,ida_v,hri_v=compute_tfri(an_sido,an_month,
        an_r.get('월최고기온',20),an_r.get('월평균기온',15),
        an_r.get('월평균습도',70),an_r.get('월강수합계',50),an_r.get('월평균일교차',8))
    comp_v=round((an_ml+tfri_v)/2,1)

    r1,r2,r3=st.columns(3)
    r1.metric("ML 발화확률",f"{an_ml:.1f}%",help="앙상블 이진분류 (v3.0)")
    r2.metric("TFRI 복합지수",f"{tfri_v:.1f}%",help="물리·통계 기반")
    r3.metric("종합위험도",f"{comp_v:.1f}%",help="두 값의 평균")

    cl,cr=st.columns(2)
    with cl:
        comp_df=pd.DataFrame({
            '성분':['WHI (기상위험)','IDA (절연열화)','HRI (이력위험)','TFRI 종합','ML 발화확률','종합위험도'],
            '값':  [whi_v, ida_v, hri_v, tfri_v, an_ml, comp_v],
        })
        clrs=['#1976D2','#D32F2F','#388E3C','#7B1FA2','#0288D1','#000000']
        fig=px.bar(comp_df,x='값',y='성분',orientation='h',
            color='성분',color_discrete_sequence=clrs,
            text=comp_df['값'].apply(lambda x:f"{x:.1f}"),
            title='위험 성분 분해')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=340,xaxis=dict(range=[0,120]),
            showlegend=False,margin=dict(l=0,r=60,t=40,b=20))
        st.plotly_chart(fig,use_container_width=True)
    with cr:
        fig=go.Figure(go.Scatterpolar(
            r=[whi_v,ida_v,hri_v,an_ml,comp_v],
            theta=['WHI<br>기상','IDA<br>절연열화','HRI<br>이력','ML<br>발화확률','종합<br>위험'],
            fill='toself',fillcolor='rgba(21,101,192,0.15)',
            line=dict(color='#1565C0',width=2.5)))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),
            title='위험 성분 레이더',height=340,
            margin=dict(l=10,r=10,t=50,b=10),showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("---")
    st.markdown(f"#### 전국 {Mn(an_month)} TFRI 성분 비교")
    all_t=[]
    for s in SIDO_LIST:
        sr=df[(df['시도']==s)&(df['연도']==2024)&(df['월']==an_month)]
        if len(sr)==0: sr=df[(df['시도']==s)&(df['월']==an_month)].tail(1)
        if len(sr)==0: continue
        rr=sr.iloc[0]
        tv,wh,id_,hr=compute_tfri(s,an_month,rr.get('월최고기온',20),rr.get('월평균기온',15),
            rr.get('월평균습도',70),rr.get('월강수합계',50),rr.get('월평균일교차',8))
        ml_v=float(rr.get('발화확률',20))
        all_t.append({'시도':s,'WHI':wh,'IDA':id_,'HRI':hr,'TFRI':tv,'ML발화확률':ml_v,
                      '종합':(tv+ml_v)/2})
    all_t_df=pd.DataFrame(all_t).sort_values('종합',ascending=False)
    fig=px.bar(all_t_df,x='시도',y=['WHI','IDA','HRI'],barmode='stack',
        color_discrete_map={'WHI':'#1976D2','IDA':'#D32F2F','HRI':'#388E3C'},
        title=f'{Mn(an_month)} 전국 TFRI 성분 스택')
    fig.update_layout(height=340,margin=dict(l=0,r=10,t=40,b=30),
        xaxis_tickangle=-30,yaxis_title='지수값')
    st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# 탭 5  점검 관리
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 📋 점검 관리")
    if 'insp_status' not in st.session_state:
        st.session_state.insp_status={s:'대기' for s in SIDO_LIST}
    if 'insp_memo' not in st.session_state:
        st.session_state.insp_memo={s:'' for s in SIDO_LIST}

    insp_base=get_baseline(view_month)
    _P1 = 'P1 Immediate' if _is_en() else 'P1 즉시'
    _P2 = 'P2 Scheduled' if _is_en() else 'P2 계획'
    _P3 = 'P3 Routine'   if _is_en() else 'P3 정기'
    insp_base['우선순위']=insp_base['종합위험'].apply(
        lambda x:_P1 if x>=40 else _P2 if x>=25 else _P3)
    insp_base['예상공수(h)']=insp_base['종합위험'].apply(
        lambda x:8 if x>=40 else 4 if x>=25 else 2)

    p1=(insp_base['우선순위']==_P1).sum()
    p2=(insp_base['우선순위']==_P2).sum()
    _done_label = 'Completed' if _is_en() else '완료'
    done=sum(1 for v in st.session_state.insp_status.values() if v in ('완료','Completed'))

    k1,k2,k3,k4,k5=st.columns(5)
    k1.metric(_P1,f"{p1} {'rgns' if _is_en() else '개'}")
    k2.metric(_P2,f"{p2} {'rgns' if _is_en() else '개'}")
    k3.metric(_P3,f"{17-p1-p2} {'rgns' if _is_en() else '개'}")
    k4.metric('Completed' if _is_en() else '완료',f"{done}/17")
    k5.metric('Est. Man-hours' if _is_en() else '총 예상 공수',f"{insp_base['예상공수(h)'].sum()}h")
    _prog_txt = f"Completion: {done/17*100:.0f}%" if _is_en() else f"점검 완료율: {done/17*100:.0f}%"
    st.progress(done/17,text=_prog_txt)
    st.markdown("---")

    for _,row in insp_base.iterrows():
        s=row['시도']; risk=row['종합위험']; g=row['등급']
        prio=row['우선순위']; exp_h=int(row['예상공수(h)'])
        ca,cb,cc,cd,ce=st.columns([1.5,1,1,2,3])
        _status_opts = (['Pending','In Progress','Completed','On Hold']
                        if _is_en() else ['대기','점검중','완료','보류'])
        _cur_status = st.session_state.insp_status.get(s,'대기')
        # normalize stored value for index lookup
        _cur_idx = 0
        for _i,_o in enumerate(_status_opts):
            if _cur_status in (_o, ['대기','점검중','완료','보류'][_i],
                               ['Pending','In Progress','Completed','On Hold'][_i]):
                _cur_idx = _i; break
        with ca:
            st.markdown(f"""<div style="background:{RISK_BG[g]};border-left:4px solid {RISK_COLOR[g]};
              border-radius:6px;padding:8px 12px;margin:2px 0"><b>{S(s)}</b>
              <span style="float:right;font-size:1.1rem;font-weight:900;color:{RISK_TEXT[g]}">
              {risk:.0f}%</span></div>""",unsafe_allow_html=True)
        with cb:
            pc={_P1:'#F44336',_P2:'#FF9800',_P3:'#4CAF50'}.get(prio,'#999')
            st.markdown(f"<span style='color:{pc};font-weight:700;font-size:0.85rem'>{prio}</span>",
                unsafe_allow_html=True)
        with cc:
            _h_lbl = f"Est. {exp_h}h" if _is_en() else f"예상 {exp_h}h"
            st.markdown(f"<span style='font-size:0.85rem;color:#555'>{_h_lbl}</span>",
                unsafe_allow_html=True)
        with cd:
            ns=st.selectbox("Status",_status_opts,index=_cur_idx,
                key=f"st_{s}",label_visibility="collapsed")
            st.session_state.insp_status[s]=ns
        with ce:
            _memo_ph = "Engineer / Notes..." if _is_en() else "담당자·메모..."
            nm=st.text_input("Memo",value=st.session_state.insp_memo.get(s,''),
                key=f"mo_{s}",label_visibility="collapsed",placeholder=_memo_ph)
            st.session_state.insp_memo[s]=nm

    st.markdown("---")
    plan_df=insp_base[['시도','종합위험','발화확률','TFRI','등급','우선순위','예상공수(h)']].copy()
    plan_df['점검상태']=plan_df['시도'].map(st.session_state.insp_status)
    plan_df['메모']=plan_df['시도'].map(st.session_state.insp_memo)
    plan_df.columns=['시도','종합위험(%)','ML발화확률(%)','TFRI(%)','위험등급','우선순위','예상공수(h)','점검상태','메모']
    st.download_button("📥 점검 계획표 CSV 다운로드",
        data=plan_df.to_csv(index=False,encoding='utf-8-sig'),
        file_name=f"TransFireRisk_점검계획_{CUR_YEAR}{view_month:02d}.csv",
        mime='text/csv')

# ═══════════════════════════════════════════════════════════════
# 탭 6  이력·모델 정보
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## 📊 이력 분석 및 모델 정보")

    col1,col2=st.columns(2)
    with col1:
        yearly=df.groupby('연도').agg(실제=('변압기화재건수','sum')).reset_index()
        yearly['고위험(≥40%)']=df[df['발화확률']>=40].groupby('연도').size().reindex(yearly['연도'],fill_value=0).values
        fig=go.Figure()
        fig.add_bar(x=yearly['연도'],y=yearly['실제'],name='실제 화재',
            marker_color='#EF5350',opacity=0.85,text=yearly['실제'],textposition='outside')
        fig.add_scatter(x=yearly['연도'],y=yearly['고위험(≥40%)'],name='발화확률≥40% 예측',
            mode='lines+markers',line=dict(color='#1565C0',width=2.5))
        fig.update_layout(title='연도별 화재건수 vs 고위험 예측 건수',
            xaxis=dict(tickvals=yearly['연도']),height=290,
            margin=dict(l=0,r=10,t=40,b=20))
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        pivot=df.groupby(['시도','월'])['변압기화재건수'].sum().unstack().fillna(0)
        fig=px.imshow(pivot,color_continuous_scale='YlOrRd',aspect='auto',
            title='시도 × 월별 화재 누계 히트맵 (2020~2024)')
        fig.update_xaxes(tickvals=list(range(1,13)),
            ticktext=[f'{i}월' for i in range(1,13)])
        fig.update_layout(height=290,margin=dict(l=0,r=10,t=40,b=20))
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("---")
    st.markdown(f"#### ⚙️ {T('model_compare')}")

    # ── 왜 F1이 아닌가? ────────────────────────────────────────
    with st.expander("📐 " + ("Why F2 / MCC / PR-AUC?" if T('avg_risk')=='National Avg Risk'
                              else "왜 F2 · MCC · PR-AUC인가?"), expanded=False):
        st.markdown(T('metric_note'))

    # ── 계산된 지표 ────────────────────────────────────────────
    te_v3 = df[df['연도']>=2023]
    yb_v3 = (te_v3['변압기화재건수']>0).astype(int)
    prob_v3 = te_v3['발화확률']/100
    # v3 metrics at thr=0.25 (best F1)
    yp_v3 = (prob_v3 >= 0.25).astype(int)
    f1_v3  = f1_score(yb_v3, yp_v3, zero_division=0)
    f2_v3  = fbeta_score(yb_v3, yp_v3, beta=2, zero_division=0)
    mcc_v3 = matthews_corrcoef(yb_v3, yp_v3)
    prauc_v3 = average_precision_score(yb_v3, prob_v3)
    rocauc_v3= roc_auc_score(yb_v3, prob_v3)

    if T('avg_risk') == 'National Avg Risk':  # EN
        perf_data = {
            'Metric':     ['ROC-AUC','PR-AUC','F2 β=2 (thr=0.25)','F1 (thr=0.25)','MCC','Recall (thr=0.20)','Fire Events Caught'],
            'v1 Baseline':['0.612',  '~0.08',  '-',                 '0.239',        '-',  '0.219',           '13/32'],
            'v3 Ensemble':[ f'{rocauc_v3:.3f}', f'{prauc_v3:.3f}',
                            f'{f2_v3:.3f}', f'{f1_v3:.3f}', f'{mcc_v3:.3f}',
                            '0.562','18/32'],
            'Change':     ['↑+0.028','↑↑','New','→','New','↑+0.343','↑+5'],
        }
        note = "*(F2 β=2 weights recall 2× — fire miss > false alarm. MCC most robust for 7.8% imbalance.)*"
    else:  # KO
        perf_data = {
            '지표':        ['ROC-AUC','PR-AUC','F2 β=2 (thr=0.25)','F1 (thr=0.25)','MCC','Recall (thr=0.20)','화재 탐지'],
            'v1 기준':     ['0.612',  '~0.08',  '-',                 '0.239',        '-',  '0.219',           '13/32건'],
            'v3 앙상블':   [ f'{rocauc_v3:.3f}', f'{prauc_v3:.3f}',
                            f'{f2_v3:.3f}', f'{f1_v3:.3f}', f'{mcc_v3:.3f}',
                            '0.562','18/32건'],
            '변화':        ['↑+0.028','↑↑','신규','→','신규','↑+0.343','↑+5건'],
        }
        note = "*(F2 β=2: 재현율 2배 가중 — 화재 미탐지 비용 > 오탐 비용. MCC: 불균형 데이터 가장 신뢰할 수 있는 단일 지표.)*"

    perf_df=pd.DataFrame(perf_data)
    st.dataframe(perf_df,use_container_width=True,hide_index=True)
    st.caption(note)

    col3,col4=st.columns(2)
    with col3:
        st.markdown("**피처 중요도 (XGB+FE 기준, 상위 15개)**")
        feat_imp_data={
            '피처':['지역발화율★','월연속고온일수','월전3일평균기온','월평균습도','기온편차★',
                   '월평균기온','열습도스트레스★','월_sin★','강수습도★','월_cos★',
                   '월강수합계','과부하스트레스★','월최고기온','강수일수','전년변압기화재건수'],
            '중요도':[0.115,0.098,0.047,0.044,0.043,0.043,0.042,0.035,0.035,0.034,
                     0.031,0.030,0.028,0.027,0.026]
        }
        fi_df=pd.DataFrame(feat_imp_data)
        fig=px.bar(fi_df,x='중요도',y='피처',orientation='h',
            color=['#E91E63' if '★' in p else '#1565C0' for p in fi_df['피처']],
            text=fi_df['중요도'].apply(lambda x:f"{x:.3f}"),
            title='★ = 신규 추가 피처')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=430,yaxis={'categoryorder':'total ascending'},
            showlegend=False,margin=dict(l=0,r=60,t=40,b=20))
        st.plotly_chart(fig,use_container_width=True)

    with col4:
        te=df[df['연도']>=2023]
        yb=(te['변압기화재건수']>0).astype(int)
        prob=te['발화확률']/100
        fpr,tpr,_=roc_curve(yb,prob)
        auc=roc_auc_score(yb,prob)
        fig=go.Figure()
        fig.add_scatter(x=fpr,y=tpr,mode='lines',name=f'v3 앙상블 (AUC={auc:.3f})',
            line=dict(color='#1565C0',width=2.5))
        fig.add_scatter(x=[0,1],y=[0,1],mode='lines',name='랜덤(AUC=0.5)',
            line=dict(color='gray',dash='dot'))
        fig.update_layout(title='ROC Curve (검증셋 2023~2024)',
            xaxis_title='False Positive Rate',yaxis_title='True Positive Rate',
            height=430,margin=dict(l=0,r=10,t=40,b=30))
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("---")
    c5,c6=st.columns(2)
    with c5:
        st.markdown("""**모델 구성**
| 항목 | 내용 |
|---|---|
| 알고리즘 | XGB + RandomForest + LogisticReg |
| 보팅 | 소프트 보팅 (0.6·0.3·0.1) |
| 목적함수 | 이진분류 P(화재 발생) |
| 피처 수 | 28개 (기존 16 + 신규 12) |
| 임계값 | 0.20 (균형) / 0.15 (Recall 우선) |
""")
    with c6:
        st.markdown("""**개선 핵심 요약**
| 문제 | 해결책 |
|---|---|
| 회귀 → 희소 카운트 불안정 | 이진 분류로 전환 |
| 클래스 불균형 (8%) | scale_pos_weight + balanced RF |
| 피처 표현력 부족 | 상호작용·편차·주기 12개 추가 |
| 단일 모델 불안정 | 3모델 소프트 보팅 앙상블 |
""")

st.markdown("---")
st.caption("⚡ **TransFireRisk IMS v7.0**  |  모델: 앙상블 v3 (XGB+RF+LR) · 28피처  |  "
           "TFRI: IEC 60076-7 · CIGRE WG A2.49  |  기상: 기상청 API + Open-Meteo  |  "
           "날씨 빅데이터 콘테스트 2026")
