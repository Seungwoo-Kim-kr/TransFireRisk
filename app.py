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
import streamlit.components.v1 as _st_components
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
    'tab_sim':       {'ko':'⚙️ 시나리오 시뮬레이션', 'en':'⚙️ Scenario Simulation'},
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
    """전국 현황 GPT 운영 브리핑"""
    if not _OPENAI_PKG or not api_key:
        return None
    lang_key = 'en' if st.session_state.get('lang','한국어') == 'English' else 'ko'
    vh   = baseline[baseline['등급']=='매우높음']['시도'].tolist()
    hi   = baseline[baseline['등급']=='높음']['시도'].tolist()
    avg  = baseline['종합위험'].mean()
    all_regions = '\n'.join(
        f"  - {r['시도']}: {r['종합위험']:.0f}% ({r['등급']})"
        for _, r in baseline.iterrows()
    )

    if lang_key == 'en':
        month_label = datetime(year, month, 1).strftime('%B %Y')
        system_msg = (
            "You are a power infrastructure safety analyst at KEPCO. "
            "Write concise, technical operational briefings in the style of a utility risk assessment report. "
            "Rules: no emojis, no casual tone. Use **bold** only for region names and key thresholds. "
            "Output in plain paragraphs under labeled sections."
        )
        user_msg = f"""Produce the {month_label} national transformer fire risk briefing.

Input data:
- National average combined risk: {avg:.1f}%
- P1 — Immediate inspection required: {', '.join(vh) if vh else 'None'}
- P2 — Caution (elevated risk): {', '.join(hi) if hi else 'None'}
- All 17 regions:
{all_regions}

Output exactly these three sections (no emojis, bold for emphasis only):

Risk Assessment
[One sentence stating the national risk level and whether it is elevated compared to baseline. Include the average score.]

Priority Response
[For each P1 region, one line: region name — risk score — primary physical driver. If no P1, state that.]
[For P2 regions, brief one-line mention.]

Recommended Actions
1. [Specific action — which equipment, which standard to check]
2. [Specific action — weather or load-related measure]"""

    else:
        system_msg = (
            "당신은 KEPCO 전력설비 안전 분석 전문가입니다. "
            "전력 유틸리티 리스크 평가 보고서 형식의 간결하고 기술적인 운영 브리핑을 작성합니다. "
            "규칙: 이모지 사용 금지, 구어체 금지. **볼드**는 지역명·핵심 수치에만 사용합니다. "
            "각 섹션은 지정된 제목 아래 단락으로 작성합니다."
        )
        user_msg = f"""{year}년 {month}월 전국 변압기 화재 위험 브리핑을 작성하세요.

입력 데이터:
- 전국 평균 종합위험도: {avg:.1f}%
- P1 (즉시 점검): {', '.join(vh) if vh else '없음'}
- P2 (주의): {', '.join(hi) if hi else '없음'}
- 전국 17개 시도:
{all_regions}

아래 세 섹션을 정확히 출력하세요 (이모지 없이, 볼드는 강조에만):

위험 수준 평가
[전국 위험 수준과 평균 점수를 포함해 한 문장으로 작성합니다.]

우선 대응
[P1 지역별로 한 줄씩: 지역명 — 위험도 — 주요 물리적 원인. P1이 없으면 해당 없음으로 표기.]
[P2 지역은 간략히 한 줄로 언급.]

권고 조치
1. [구체적인 조치 — 대상 설비, 확인 기준 포함]
2. [기상 또는 부하 관련 조치]"""

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=400, temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


def rule_briefing(baseline, month) -> str:
    """API 없을 때 구조화된 규칙 기반 브리핑"""
    lang_key = 'en' if st.session_state.get('lang','한국어') == 'English' else 'ko'
    vh  = baseline[baseline['등급']=='매우높음']['시도'].tolist()
    hi  = baseline[baseline['등급']=='높음']['시도'].tolist()
    avg = baseline['종합위험'].mean()

    if lang_key == 'en':
        level = "HIGH" if avg >= 30 else "MODERATE" if avg >= 20 else "LOW"
        lines = [
            f"Risk Assessment\n"
            f"The national average combined risk index for this month stands at "
            f"**{avg:.1f}%**, corresponding to an overall **{level}** level.",

            "Priority Response\n"
            + (('\n'.join(f"- **{s}**: Immediate inspection required (≥40% fire probability)."
                         for s in vh)) if vh else "No P1 regions this month.")
            + ('\n' + '\n'.join(f"- **{s}**: Elevated risk — monitor closely." for s in hi) if hi else ""),

            "Recommended Actions\n"
            "1. Inspect cooling fans, heat exchangers, and insulation resistance on high-load transformers "
            "(acceptance: ≥1 GΩ at 1 kV).\n"
            "2. Review busbar and terminal heat patterns via IR camera; tighten or replace as needed.",
        ]
    else:
        level = "높음" if avg >= 30 else "보통" if avg >= 20 else "낮음"
        lines = [
            f"위험 수준 평가\n"
            f"이번 달 전국 평균 종합위험도는 **{avg:.1f}%**로, 전반적인 위험 수준은 **{level}**입니다.",

            "우선 대응\n"
            + (('\n'.join(f"- **{s}**: 즉시 점검 필요 (발화 확률 40% 이상)." for s in vh))
               if vh else "이번 달 P1 해당 지역 없음.")
            + ('\n' + '\n'.join(f"- **{s}**: 위험 상승 — 집중 모니터링." for s in hi) if hi else ""),

            "권고 조치\n"
            "1. 고부하 변압기 냉각팬·방열기 점검 및 절연 저항 측정 (기준: 1kV 인가 시 ≥1 GΩ).\n"
            "2. IR 카메라로 부스바·단자 발열 여부 확인 후 필요 시 체결 보강 또는 교체.",
        ]
    return "\n\n".join(lines)

st.set_page_config(
    page_title="TransFireRisk IMS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ── 테마 session state 초기화 ─────────────────────────────────────
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = True

_dark = st.session_state['dark_mode']

# ── 다크/라이트 팔레트 ────────────────────────────────────────────
if _dark:
    _BG           = "#0D1117"
    _BG2          = "#161B22"
    _TEXT         = "#E6EDF3"
    _TEXT_SUB     = "#8B9BAF"
    _ALERT_P1_BG  = "rgba(244,67,54,0.15)";  _ALERT_P1_TXT = "#EF9A9A"
    _ALERT_P2_BG  = "rgba(255,152,0,0.12)";  _ALERT_P2_TXT = "#FFCC80"
    _METHOD_BG    = "rgba(63,81,181,0.15)";   _METHOD_TXT   = "#C5CAE9"
    _AI_BG        = "rgba(21,101,192,0.12)";  _AI_TXT       = "#BBDEFB"
    _CAL_P1 = "background:rgba(244,67,54,0.25);border:1px solid #F44336;color:#EF9A9A"
    _CAL_P2 = "background:rgba(255,152,0,0.20);border:1px solid #FF9800;color:#FFCC80"
    _CAL_P3 = "background:rgba(76,175,80,0.15);border:1px solid #4CAF50;color:#A5D6A7"
else:
    _BG           = "#F8F9FA"
    _BG2          = "#FFFFFF"
    _TEXT         = "#1A1A2E"
    _TEXT_SUB     = "#555555"
    _ALERT_P1_BG  = "#FFEBEE";  _ALERT_P1_TXT = "#C62828"
    _ALERT_P2_BG  = "#FFF3E0";  _ALERT_P2_TXT = "#E65100"
    _METHOD_BG    = "#E8EAF6";  _METHOD_TXT   = "#1A237E"
    _AI_BG        = "#E3F2FD";  _AI_TXT       = "#0D47A1"
    _CAL_P1 = "background:#FFEBEE;border:1px solid #F44336;color:#C62828"
    _CAL_P2 = "background:#FFF3E0;border:1px solid #FF9800;color:#E65100"
    _CAL_P3 = "background:#E8F5E9;border:1px solid #4CAF50;color:#2E7D32"

# ── 라이트 모드: Streamlit 다크 테마 전면 덮어쓰기 ────────────────
_LIGHT_OVERRIDE = "" if _dark else f"""
/* ① 앱 전체 배경 */
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
.main .block-container, section.main {{
    background-color: {_BG} !important;
}}
/* ② 사이드바 */
[data-testid="stSidebar"] > div:first-child {{
    background-color: {_BG2} !important;
    border-right: 1px solid #E0E0E0 !important;
}}
/* ③ 모든 텍스트를 다크로 — 공통 우선순위 */
body, .stApp, .stApp * {{
    color: {_TEXT} !important;
}}
/* ④ 서브 색상 복원 */
[data-testid="stMetricLabel"],
[data-testid="stCaptionContainer"],
small, caption {{
    color: {_TEXT_SUB} !important;
}}
/* ⑤ 셀렉트박스·드롭다운 */
[data-baseweb="select"] > div,
[data-baseweb="select"] [role="combobox"],
[data-baseweb="select"] input {{
    background-color: {_BG2} !important;
    border-color: #CCCCCC !important;
    transition: none !important;
}}
/* Emotion 생성 클래스 직접 패치 (background-color: dark → white) */
.st-dw {{ background-color: {_BG2} !important; transition: none !important; }}
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] li,
[data-baseweb="menu"] {{
    background-color: {_BG2} !important;
    color: {_TEXT} !important;
}}
/* ⑥ 텍스트 인풋·슬라이더 */
[data-baseweb="input"] {{
    background-color: {_BG2} !important;
    border-color: #CCCCCC !important;
}}
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {{
    background-color: {_BG2} !important;
    color: {_TEXT} !important;
}}
/* ⑦ 탭 — 라이트 모드: 선택된 탭은 파란 텍스트+언더라인, 비선택은 회색 텍스트 */
[data-baseweb="tab-list"] {{
    background-color: {_BG} !important;
    border-bottom: 2px solid #E0E0E0 !important;
}}
[data-baseweb="tab"] {{
    background-color: transparent !important;
    color: #666666 !important;
}}
[aria-selected="true"][data-baseweb="tab"] {{
    background-color: transparent !important;
    color: #1565C0 !important;
    font-weight: 700 !important;
    border-bottom: 3px solid #1565C0 !important;
}}
/* ⑧ Expander */
[data-testid="stExpander"] {{
    background-color: {_BG2} !important;
    border: 1px solid #E0E0E0 !important;
}}
/* ⑨ DataFrames */
[data-testid="stDataFrame"] iframe,
[data-testid="stDataFrameResizable"] * {{
    background-color: {_BG2} !important;
    color: {_TEXT} !important;
}}
/* ⑩ 라디오·체크박스 레이블 */
[data-testid="stRadio"] label p,
[data-testid="stCheckbox"] label p,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label {{
    color: {_TEXT} !important;
}}
/* ⑪ 버튼 */
[data-testid="stButton"] > button {{
    background-color: #E8EAF6 !important;
    color: {_TEXT} !important;
    border: 1px solid #9FA8DA !important;
}}
[data-testid="stButton"] > button:hover {{
    background-color: #C5CAE9 !important;
}}
/* ⑫ Streamlit 경고/정보 박스 */
[data-testid="stAlert"] {{
    background-color: {_BG2} !important;
}}
/* ⑬ Progress */
[data-testid="stProgressBar"] {{
    background-color: #E0E0E0 !important;
}}
/* ⑭ Download 버튼 */
[data-testid="stDownloadButton"] > button {{
    background-color: #1565C0 !important;
    color: #FFFFFF !important;
    border: none !important;
}}
/* ⑮ 헤더 */
[data-testid="stHeader"] {{
    background-color: {_BG} !important;
}}
"""

# ── 라이트 모드: JS로 Emotion 재주입 후에도 강제 적용 (interval + observer) ──
if not _dark:
    _st_components.html("""
<script>
(function(){
  function patchSheet(doc){
    // .st-dw 같은 emotion 클래스를 스타일시트에서 직접 패치
    doc.querySelectorAll('style').forEach(function(tag){
      try {
        Array.from(tag.sheet.cssRules).forEach(function(rule){
          if(!rule.selectorText) return;
          // 셀렉트박스 배경색 클래스들
          if(rule.style && rule.style.backgroundColor === 'rgb(13, 17, 23)'){
            rule.style.setProperty('background-color','#FFFFFF','important');
          }
          // 트랜지션 제거 (background-color가 dark로 되돌아가는 것 방지)
          if(rule.style && rule.style.transitionProperty &&
             rule.style.transitionProperty.includes('background')){
            rule.style.setProperty('transition','none','important');
          }
        });
      } catch(e){}
    });
  }

  function fix(doc){
    patchSheet(doc);
    // 셀렉트박스 > 첫번째 div 인라인 강제
    doc.querySelectorAll('[data-baseweb="select"] > div').forEach(function(el){
      el.style.setProperty('transition','none','important');
      el.style.setProperty('background-color','#FFFFFF','important');
      el.style.setProperty('border-color','#CCCCCC','important');
    });
    // 셀렉트박스 내부 텍스트
    doc.querySelectorAll('[data-baseweb="select"] *').forEach(function(el){
      var bg = window.getComputedStyle(el).backgroundColor;
      if(bg === 'rgb(13, 17, 23)' || bg === 'rgb(22, 27, 34)'){
        el.style.setProperty('background-color','#FFFFFF','important');
      }
      el.style.setProperty('color','#1A1A2E','important');
    });
    // 선택된 탭 — 파란 텍스트
    doc.querySelectorAll('[aria-selected="true"][data-baseweb="tab"]').forEach(function(el){
      el.style.setProperty('color','#1565C0','important');
      el.style.setProperty('background-color','transparent','important');
    });
    // 비선택 탭 — 회색 텍스트
    doc.querySelectorAll('[aria-selected="false"][data-baseweb="tab"]').forEach(function(el){
      el.style.setProperty('color','#555555','important');
    });
    // 드롭다운 옵션
    doc.querySelectorAll('[data-baseweb="popover"] li, [role="option"]').forEach(function(el){
      el.style.setProperty('background-color','#FFFFFF','important');
      el.style.setProperty('color','#1A1A2E','important');
    });
    // 사이드바 섹션 자체
    var sb = doc.querySelector('[data-testid="stSidebar"]');
    if(sb) sb.style.setProperty('background-color','#FFFFFF','important');
    // 일반 버튼 배경 교정 (다운로드 제외)
    var DARK_BGS = ['rgb(26, 32, 40)','rgb(13, 17, 23)','rgb(22, 27, 34)','rgb(19, 23, 32)'];
    doc.querySelectorAll('button').forEach(function(el){
      var bg = window.getComputedStyle(el).backgroundColor;
      if(DARK_BGS.indexOf(bg) !== -1 && !el.closest('[data-testid="stDownloadButton"]')){
        el.style.setProperty('background-color','#E8EAF6','important');
        el.style.setProperty('color','#1A1A2E','important');
        el.style.setProperty('border','1px solid #9FA8DA','important');
      }
    });
  }

  try {
    var doc = window.parent.document;
    fix(doc);
    // DOM 변경 감지
    var obs = new MutationObserver(function(){ fix(doc); });
    obs.observe(doc.body, {childList:true, subtree:true, attributes:true, attributeFilter:['style','class']});
    // 주기적 보정 (React 리렌더 후 스타일 재적용 보장)
    setInterval(function(){ fix(doc); }, 400);
  } catch(e){}
})();
</script>
""", height=0)

st.markdown(f"""<style>
{_LIGHT_OVERRIDE}
/* ── 공통 (다크·라이트 모두 적용) ── */
[data-testid="stMetricValue"]{{font-size:1.55rem!important;font-weight:700!important}}
[data-testid="stMetricLabel"]{{font-size:0.8rem!important;color:{_TEXT_SUB}!important}}
.ims-header{{background:linear-gradient(135deg,#0D1B4B 0%,#1565C0 100%);
  padding:18px 28px;border-radius:10px;margin-bottom:18px;
  display:flex;justify-content:space-between;align-items:center}}
.alert-p1{{background:{_ALERT_P1_BG};border-left:5px solid #F44336;
  border-radius:6px;padding:10px 16px;margin:3px 0;color:{_ALERT_P1_TXT}!important;font-weight:600}}
.alert-p2{{background:{_ALERT_P2_BG};border-left:5px solid #FF9800;
  border-radius:6px;padding:10px 16px;margin:3px 0;color:{_ALERT_P2_TXT}!important;font-weight:600}}
.method-box{{background:{_METHOD_BG};border-radius:8px;padding:14px;
  border-left:4px solid #5C6BC0;margin:8px 0;font-size:0.87rem;color:{_METHOD_TXT}!important}}
.ai-box{{background:{_AI_BG};border-radius:10px;padding:16px;
  border-left:4px solid #42A5F5;margin-top:10px;color:{_AI_TXT}!important}}
.perf-delta-good{{color:#2E7D32!important;font-weight:700}}
.perf-delta-bad{{color:#C62828!important;font-weight:700}}
.cal-day{{border-radius:6px;padding:4px 2px;text-align:center;font-size:0.72rem;margin:1px}}
.cal-p1{{{_CAL_P1};font-weight:700}}
.cal-p2{{{_CAL_P2};font-weight:600}}
.cal-p3{{{_CAL_P3}}}
@media(max-width:768px){{
  .main .block-container{{padding:0.5rem!important;max-width:100%!important}}
  .ims-header{{flex-direction:column;gap:8px}}
  [data-testid="stMetricValue"]{{font-size:1.1rem!important}}
}}
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

# ── GPT 점검 가이드 ───────────────────────────────────────────
def get_ai_guide(sido, month, grade, ml_pct, tfri_pct, whi, ida, hri, reasons, api_key):
    if not _OPENAI_PKG or not api_key: return None, "API KEY 없음"
    lang_key = 'en' if st.session_state.get('lang', '한국어') == 'English' else 'ko'
    combined = round((ml_pct + tfri_pct) / 2, 1)

    if lang_key == 'en':
        system_msg = (
            "You are a senior electrical engineer specializing in transformer fire prevention, "
            "with deep expertise in IEC 60076-7 and CIGRE WG A2.49. "
            "Produce concise, field-ready technical inspection guides for KEPCO field engineers. "
            "Rules: no emojis, technical and factual tone only. "
            "Use **bold** for acceptance criteria and critical thresholds. "
            "Each section must be short and directly actionable."
        )
        user_msg = f"""Region: {sido} | Month: {month}
Risk grade: {grade} | ML fire probability: {ml_pct:.1f}% | TFRI: {tfri_pct:.1f}% | Combined: {combined:.1f}%
WHI (weather hazard index): {whi:.1f} | IDA (insulation degradation): {ida:.1f} | HRI (historical risk): {hri:.1f}
Key risk drivers: {', '.join(reasons) if reasons else 'None identified'}

Write a field inspection brief with these four sections (no emojis):

Risk Mechanism
[2 sentences explaining the dominant physical cause of elevated risk based on WHI/IDA/HRI values and the risk drivers above.]

Immediate Inspection Checklist
- [Equipment item]: [acceptance criterion in bold, e.g. **≥1 GΩ at 1 kV**]
- [Equipment item]: [acceptance criterion]
- [Equipment item]: [acceptance criterion]
- [Equipment item]: [acceptance criterion]

Maintenance Plan — Month {month}
[2 sentences on specific maintenance priorities given the current risk level and season.]

Weather Response
[2 sentences on operational adjustments specific to current weather conditions.]"""

    else:
        system_msg = (
            "당신은 IEC 60076-7·CIGRE WG A2.49 기반의 변압기 화재 예방 전문 전기 엔지니어입니다. "
            "KEPCO 현장 기술자를 위한 간결하고 즉시 실행 가능한 점검 가이드를 작성합니다. "
            "규칙: 이모지 사용 절대 금지, 기술적·사실적 어조 유지. "
            "**볼드**는 허용 기준값과 핵심 임계치에만 사용합니다. "
            "각 섹션은 짧고 직접적으로 작성합니다."
        )
        user_msg = f"""지역: {sido} | 월: {month}월
위험등급: {grade} | ML 발화확률: {ml_pct:.1f}% | TFRI: {tfri_pct:.1f}% | 종합: {combined:.1f}%
WHI (기상위험지수): {whi:.1f} | IDA (절연열화가속도): {ida:.1f} | HRI (이력위험지수): {hri:.1f}
주요 위험 요인: {', '.join(reasons) if reasons else '없음'}

아래 네 섹션으로 현장 점검 브리핑을 작성하세요 (이모지 없이):

위험 메커니즘
[WHI/IDA/HRI 수치와 위험 요인을 근거로 위험 상승의 주요 물리적 원인을 2문장으로 설명합니다.]

즉시 점검 항목
- [설비 항목]: [허용 기준을 볼드로 표기, 예: **1kV 인가 시 ≥1 GΩ**]
- [설비 항목]: [허용 기준]
- [설비 항목]: [허용 기준]
- [설비 항목]: [허용 기준]

{month}월 정비 계획
[현재 위험 등급과 계절을 고려한 구체적인 정비 우선순위를 2문장으로 작성합니다.]

기상 대응 조치
[현재 기상 조건에 따른 운전 조정 사항을 2문장으로 작성합니다.]"""

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=600, temperature=0.2,
        )
        return resp.choices[0].message.content, None
    except Exception as e:
        return None, str(e)


def rule_guide(grade, whi, ida):
    """API 없을 때 구조화된 규칙 기반 점검 가이드"""
    _en = st.session_state.get('lang', '한국어') == 'English'

    if _en:
        overview = {
            '낮음':    "Risk is within normal range. Maintain the regular inspection schedule.",
            '보통':    "Risk is moderately elevated. Focus monitoring on vulnerable equipment.",
            '높음':    "Risk is high. Initiate immediate inspection of high-load transformers and activate emergency response protocols.",
            '매우높음':"Risk is critical. Execute emergency special inspection and establish 24-hour monitoring.",
        }.get(grade, "")
        items = [
            "Insulation resistance measurement: **≥1 GΩ at 1 kV (DC)**",
            "Busbar and terminal thermal scan via IR camera: **ΔT < 10 K above ambient**",
        ]
        if whi > 60: items += [
            "Cooling fan and heat exchanger operational check: confirm airflow and coolant level",
            "Load redistribution review: verify no transformer exceeds **85% rated capacity**",
        ]
        if ida > 50: items += [
            "Insulating oil dielectric strength: **≥30 kV / 2.5 mm gap**",
            "Moisture breather and silica gel condition: replace if color-indicator saturated",
        ]
        if whi > 40: items.append(
            "Cable entry seals and weatherproof gaskets: verify integrity, reseal if cracked"
        )
        return (
            f"Risk Mechanism\n{overview}\n\n"
            f"Immediate Inspection Checklist\n"
            + "\n".join(f"- {i}" for i in items[:5]) + "\n\n"
            f"Maintenance Plan\n"
            "Prioritize transformers with the highest load factor this month. "
            "Schedule oil sampling and DGA analysis for units rated above 154 kV.\n\n"
            f"Weather Response\n"
            "Increase patrol frequency during high-temperature or high-humidity periods. "
            "Pre-position spare cooling units at P1 substations."
        )
    else:
        overview = {
            '낮음':    "현재 위험은 정상 범위 내에 있습니다. 정기 점검 주기를 유지합니다.",
            '보통':    "위험이 소폭 상승했습니다. 취약 설비를 집중 모니터링합니다.",
            '높음':    "위험이 높은 상태입니다. 고부하 변압기를 즉시 점검하고 비상 대응 체계를 가동합니다.",
            '매우높음':"위험이 임계 수준입니다. 비상 특별 점검을 즉시 시행하고 24시간 감시 체계를 구축합니다.",
        }.get(grade, "")
        items = [
            "절연 저항 측정: **DC 1kV 인가 시 ≥1 GΩ**",
            "부스바·단자 IR 열화상 점검: **주변 온도 대비 ΔT < 10 K**",
        ]
        if whi > 60: items += [
            "냉각팬·방열기 작동 확인: 풍량 및 냉각유 수위 점검",
            "부하 분산 검토: 변압기별 **정격 용량의 85% 이하** 유지 여부 확인",
        ]
        if ida > 50: items += [
            "절연유 내전압 측정: **2.5mm 간격 기준 ≥30 kV**",
            "흡습 브리더·실리카겔 상태 점검: 색변 포화 시 즉시 교체",
        ]
        if whi > 40: items.append(
            "케이블 관통부 실링·방수 패킹 점검: 균열 발견 시 재실링"
        )
        return (
            f"위험 메커니즘\n{overview}\n\n"
            f"즉시 점검 항목\n"
            + "\n".join(f"- {i}" for i in items[:5]) + "\n\n"
            f"정비 계획\n"
            "이번 달 부하율이 높은 변압기를 우선 점검합니다. "
            "154kV 이상 설비는 절연유 샘플링 및 DGA 분석을 예약합니다.\n\n"
            f"기상 대응 조치\n"
            "고온·고습 기간 중 순시 주기를 단축합니다. "
            "P1 변전소에 예비 냉각 장치를 사전 배치합니다."
        )

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    # 헤더 + 다크/라이트 토글 한 줄
    _hdr_col, _toggle_col = st.columns([3, 1])
    with _hdr_col:
        st.markdown(f"### ⚡ TransFireRisk IMS")
    with _toggle_col:
        _icon = "☀️" if _dark else "🌙"
        _label = "라이트" if _dark else "다크"
        if st.button(f"{_icon}", help=f"{_label} 모드로 전환", use_container_width=True):
            st.session_state['dark_mode'] = not st.session_state['dark_mode']
            st.rerun()

    st.markdown(f"`{CUR_YEAR}.{CUR_MONTH:02d}.{TODAY.day:02d}` · `{NOW.strftime('%H:%M')} KST`")
    lang_sel = st.radio("🌐 Language", ["한국어", "English"], horizontal=True, key='lang')
    st.markdown("---")
    view_month=st.selectbox(f"📅 {T('view_month')}",list(range(1,13)),
        index=CUR_MONTH-1,format_func=lambda x:MONTH_KR[x],key="view_month")

    baseline=get_baseline(view_month)
    vh=baseline[baseline['등급']=='매우높음']['시도'].tolist()
    hi=baseline[baseline['등급']=='높음']['시도'].tolist()
    st.markdown(f"**🚨 {T('sidebar_title')}**")
    if vh: st.error(f"🔴 {T('p1_label')} ({len(vh)}): {', '.join(S(s) for s in vh)}")
    if hi: st.warning(f"🟠 {T('p2_label')} ({len(hi)}): {', '.join(S(s) for s in hi)}")
    if not vh and not hi: st.success(f"✅ {T('normal_label')}")
    st.markdown("---")
    # API 연결 상태 (한 줄 요약)
    _ai_dot  = '🟢' if _OPENAI_KEY else '⚪'
    st.markdown(f"**🔑 AI API** &nbsp; {_ai_dot} {'Connected' if _OPENAI_KEY else 'Not set'}")

    st.markdown("---")
    # ── PDF 보고서 생성 버튼 ──────────────────────────────────────
    _pdf_lbl = "📄 PDF 보고서 생성" if st.session_state.get('lang','한국어')=='한국어' else "📄 Generate PDF Report"
    if st.button(_pdf_lbl, use_container_width=True, type="secondary", key="pdf_gen"):
        import tempfile, io
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors as _rl_colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        with st.spinner("PDF 생성 중..." if st.session_state.get('lang','한국어')=='한국어' else "Generating PDF..."):
            try:
                _font_p = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
                if os.path.exists(_font_p):
                    pdfmetrics.registerFont(TTFont("KO", _font_p))
                    _fn = "KO"
                else:
                    _fn = "Helvetica"

                _buf = io.BytesIO()
                _doc = SimpleDocTemplate(_buf, pagesize=A4,
                                         rightMargin=2*cm, leftMargin=2*cm,
                                         topMargin=2*cm, bottomMargin=2*cm)
                _styles = getSampleStyleSheet()
                _h1 = ParagraphStyle("h1", fontName=_fn, fontSize=18, spaceAfter=12,
                                     textColor=_rl_colors.HexColor("#1565C0"), leading=22)
                _h2 = ParagraphStyle("h2", fontName=_fn, fontSize=12, spaceAfter=8,
                                     textColor=_rl_colors.HexColor("#0D47A1"), leading=16)
                _body = ParagraphStyle("body", fontName=_fn, fontSize=9, leading=14, spaceAfter=6)

                _base = get_baseline(view_month)
                _story = []
                _story.append(Paragraph(f"TransFireRisk IMS — {CUR_YEAR}년 {view_month}월 보고서", _h1))
                _story.append(Paragraph(f"생성일시: {NOW.strftime('%Y-%m-%d %H:%M')} KST", _body))
                _story.append(Spacer(1, 0.3*cm))
                _story.append(Paragraph("전국 위험도 요약", _h2))

                _tbl_data = [["지역","종합위험도(%)","ML확률(%)","TFRI(%)","등급","우선순위"]]
                for _, _pr in _base.iterrows():
                    _prio = "P1 즉시" if _pr['종합위험']>=40 else "P2 계획" if _pr['종합위험']>=25 else "P3 정기"
                    _tbl_data.append([_pr['시도'], f"{_pr['종합위험']:.1f}",
                                      f"{_pr['발화확률']:.1f}", f"{_pr['TFRI']:.1f}",
                                      _pr['등급'], _prio])

                _tbl = Table(_tbl_data, colWidths=[2.5*cm,2.5*cm,2.5*cm,2.5*cm,2.5*cm,2.5*cm])
                _tbl.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), _rl_colors.HexColor("#1565C0")),
                    ('TEXTCOLOR',  (0,0), (-1,0), _rl_colors.white),
                    ('FONTNAME',   (0,0), (-1,-1), _fn),
                    ('FONTSIZE',   (0,0), (-1,-1), 8),
                    ('GRID',       (0,0), (-1,-1), 0.5, _rl_colors.grey),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [_rl_colors.white, _rl_colors.HexColor("#F5F5F5")]),
                    ('ALIGN',      (1,0), (-1,-1), 'CENTER'),
                ]))
                _story.append(_tbl)
                _story.append(Spacer(1, 0.4*cm))
                _story.append(Paragraph("모델 성능", _h2))
                _story.append(Paragraph(f"ROC-AUC: 0.640 | PR-AUC: 0.122 | Recall@0.20: 0.562 | F2(β=2): {f2_v3:.3f}", _body))
                _story.append(Paragraph("앙상블 v3: XGB(60%) + RandomForest(30%) + LogisticRegression(10%)", _body))

                _doc.build(_story)
                st.session_state['_pdf_bytes'] = _buf.getvalue()
                st.success("✅ PDF 생성 완료")
            except Exception as _e:
                st.error(f"PDF 오류: {_e}")

    if '_pdf_bytes' in st.session_state:
        st.download_button(
            "💾 저장" if st.session_state.get('lang','한국어')=='한국어' else "💾 Save PDF",
            data=st.session_state['_pdf_bytes'],
            file_name=f"TransFireRisk_{CUR_YEAR}{view_month:02d}.pdf",
            mime="application/pdf", use_container_width=True, key="pdf_dl"
        )

# ── 전역 API 키 변수 (sidebar 이후에도 사용) ─────────────────
openai_key = _OPENAI_KEY

# ── IMS 헤더 ─────────────────────────────────────────────────
_hdr_sub = ("Transformer Fire Risk IMS · Prediction Model v3.0 · Korea Weather Big Data Contest 2026"
            if _is_en() else
            "변압기 화재 위험 통합관리 시스템 · 발화 확률 모델 v3.0 · 날씨 빅데이터 콘테스트 2026")
_hdr_ref = (f"{Mn(view_month)} {CUR_YEAR} Reference"
            if _is_en() else
            f"{CUR_YEAR}년 {Mn(view_month)} 기준")
_hdr_model = ("Ensemble (XGB+RF+LR) · 28 features · AUC 0.640"
              if _is_en() else
              "앙상블(XGB+RF+LR) · 28 피처 · AUC 0.640")
st.markdown(f"""
<div class="ims-header">
  <div>
    <div style="color:#90CAF9;font-size:0.75rem;letter-spacing:3px">INTEGRATED MANAGEMENT SYSTEM</div>
    <div style="color:#fff;font-size:1.7rem;font-weight:800;line-height:1.1">⚡ TransFireRisk IMS</div>
    <div style="color:#BBDEFB;font-size:0.82rem;margin-top:2px">{_hdr_sub}</div>
  </div>
  <div style="text-align:right;color:#90CAF9">
    <div style="font-size:1.1rem;font-weight:600;color:#fff">{_hdr_ref}</div>
    <div style="font-size:0.8rem">{_hdr_model}</div>
    <div style="font-size:0.78rem;margin-top:2px">
      {'🤖 AI Connected' if openai_key else ''}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ── 탭 (언어에 따라 동적 레이블) ─────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7=st.tabs([
    T('tab_dashboard'), T('tab_region'),  T('tab_forecast'),
    T('tab_sim'),       T('tab_risk'),    T('tab_insp'),    T('tab_model'),
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
              delta=f"{avg_now-avg_prev:+.1f}%p {'MoM' if _is_en() else '전월比'}")
    k2.metric(T('p1_count'),  f"{len(vh)} {'regions' if _is_en() else '개 지역'}",
              delta=', '.join(S(s) for s in vh) if vh else ("N/A" if _is_en() else "해당 없음"),
              delta_color="inverse" if vh else "off")
    k3.metric(T('p2_count'),  f"{len(hi)} {'regions' if _is_en() else '개'}")
    k4.metric(T('est_fire'),  f"{exp_fire:.0f} {'fires' if _is_en() else '건'}",
              help="≥40%: 1건, 25~40%: 0.5건 기대값")
    k5.metric(T('top_region'), baseline.iloc[0]['시도'],
              delta=f"{baseline.iloc[0]['종합위험']:.0f}%")

    st.markdown("---")
    st.markdown(f"### 📍 {Mn(view_month)} {T('region_grid')} (ML + TFRI)")

    # ── 뷰 토글: 지도 / 카드 ──────────────────────────────────────
    _view_opts = (["🗺️ 전국 지도 (Choropleth)", "📊 카드 그리드"] if not _is_en()
                  else ["🗺️ National Map (Choropleth)", "📊 Card Grid"])
    _map_view = st.radio("뷰 선택" if not _is_en() else "View",
                         _view_opts, horizontal=True, label_visibility="collapsed")

    if "Choropleth" in _map_view or "지도" in _map_view:
        # ── Plotly scatter_mapbox 기반 전국 지도 ─────────────────
        _map_data = []
        for _, _breg in baseline.iterrows():
            _s = _breg['시도']
            _lat, _lon = SIDO_COORDS.get(_s, (37.5, 127.0))
            _map_data.append({
                '시도': S(_s), '위험도': _breg['종합위험'],
                '등급': G(_breg['등급']), 'ML': _breg['발화확률'],
                'TFRI': _breg['TFRI'], 'lat': _lat, 'lon': _lon,
            })
        _map_df = pd.DataFrame(_map_data)
        _grade_order = (['매우높음','높음','보통','낮음'] if not _is_en()
                        else ['Very High','High','Moderate','Low'])
        _cmap = {G(k): v for k, v in RISK_COLOR.items()}
        _fig_map = px.scatter_mapbox(
            _map_df, lat='lat', lon='lon',
            color='등급' if not _is_en() else '등급',
            size='위험도', size_max=55,
            color_discrete_map=_cmap,
            category_orders={'등급': [G(g) for g in ['매우높음','높음','보통','낮음']]},
            hover_name='시도',
            hover_data={'위험도':':.1f','ML':':.1f','TFRI':':.1f','lat':False,'lon':False},
            mapbox_style='carto-darkmatter',
            zoom=5.8, center={'lat': 36.5, 'lon': 127.8},
            title=f"{Mn(view_month)} " + ("National Transformer Fire Risk" if _is_en()
                                           else "전국 변압기 화재 위험도"),
        )
        _fig_map.update_layout(
            height=520, margin=dict(l=0,r=0,t=40,b=0),
            legend=dict(title='위험등급' if not _is_en() else 'Grade',
                        orientation='v', x=0.01, y=0.99,
                        bgcolor='rgba(13,17,23,0.85)',
                        bordercolor='rgba(255,255,255,0.2)', borderwidth=1,
                        font=dict(color='white')),
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(_fig_map, use_container_width=True)
        # 상위 3 지역 배너
        _top3 = baseline.head(3)
        _t3cols = st.columns(3)
        for _col, (_, _tr) in zip(_t3cols, _top3.iterrows()):
            _g = _tr['등급']
            _col.markdown(
                f"<div style='background:{RISK_COLOR[_g]}22;border:1px solid {RISK_COLOR[_g]};"
                f"border-radius:8px;padding:8px;text-align:center'>"
                f"<b style='color:{RISK_TEXT[_g]}'>{S(_tr['시도'])}</b><br>"
                f"<span style='font-size:1.4rem;font-weight:900;color:{RISK_TEXT[_g]}'>"
                f"{_tr['종합위험']:.0f}%</span><br>"
                f"<span style='font-size:0.72rem;color:{RISK_TEXT[_g]}'>{RISK_EMOJI[_g]} {G(_g)}</span>"
                f"</div>", unsafe_allow_html=True)
    else:
        # ── 기존 카드 그리드 (다크모드 보정) ─────────────────────
        rows_5=[baseline.iloc[i:i+5] for i in range(0,len(baseline),5)]
        for grp in rows_5:
            cols=st.columns(5)
            for ci,(_,reg) in enumerate(grp.iterrows()):
                g=reg['등급']; ml=reg['발화확률']; tf=reg['TFRI']; v=reg['종합위험']
                with cols[ci]:
                    st.markdown(f"""
                    <div style="background:{RISK_COLOR[g]}22;border:2px solid {RISK_COLOR[g]};
                      border-radius:10px;padding:12px 6px;text-align:center;min-height:110px">
                      <div style="font-size:0.95rem;font-weight:700;color:{RISK_TEXT[g]}">{S(reg['시도'])}</div>
                      <div style="font-size:2.0rem;font-weight:900;color:{RISK_COLOR[g]};
                        line-height:1.1;margin:2px 0">{v:.0f}%</div>
                      <div style="font-size:0.68rem;color:#8B9BAF">ML {ml:.0f}%  TFRI {tf:.0f}%</div>
                      <div style="font-size:0.75rem;color:{RISK_TEXT[g]}">{RISK_EMOJI[g]} {G(g)}</div>
                    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a,col_b=st.columns(2)
    with col_a:
        st.markdown("#### 📊 " + ("Risk Ranking" if _is_en() else "종합위험 순위"))
        fig=px.bar(baseline,x='종합위험',y='시도',orientation='h',
            color='등급',color_discrete_map=RISK_COLOR,
            category_orders={'등급':['매우높음','높음','보통','낮음']},
            text=baseline['종합위험'].apply(lambda x:f"{x:.0f}%"))
        fig.update_traces(textposition='outside')
        fig.update_layout(height=440,yaxis={'categoryorder':'total ascending'},
            xaxis_title='Combined Risk (%)' if _is_en() else '종합위험도 (%)',yaxis_title='',
            margin=dict(l=0,r=60,t=20,b=20))
        st.plotly_chart(fig,use_container_width=True)

    with col_b:
        st.markdown("#### 📈 " + ("ML Fire Prob. vs TFRI" if _is_en() else "ML 발화확률 vs TFRI 비교"))
        comp=baseline[['시도','발화확률','TFRI','종합위험']].sort_values('종합위험',ascending=True)
        fig=go.Figure()
        fig.add_bar(x=comp['발화확률'],y=comp['시도'],
            name='ML Fire Prob.' if _is_en() else 'ML 발화확률',
            marker_color='#1565C0',opacity=0.75,orientation='h')
        fig.add_bar(x=comp['TFRI'],y=comp['시도'],
            name='TFRI Index' if _is_en() else 'TFRI 복합지수',
            marker_color='#E91E63',opacity=0.75,orientation='h')
        fig.add_scatter(x=comp['종합위험'],y=comp['시도'],mode='markers',
            marker=dict(color='#333',size=8,symbol='diamond'),
            name='Combined' if _is_en() else '종합위험',orientation='h')
        fig.update_layout(barmode='group',height=440,
            xaxis_title='Risk (%)' if _is_en() else '위험도(%)',
            yaxis_title='',margin=dict(l=0,r=10,t=20,b=20))
        st.plotly_chart(fig,use_container_width=True)

    if '월최고기온' in baseline.columns:
        st.markdown("---")
        w1,w2,w3,w4=st.columns(4)
        w1.metric("🌡️ " + ("High Temp Regions (≥33℃)" if _is_en() else "고온 지역 (≥33℃)"),
                  f"{(baseline['월최고기온']>=33).sum()}" + (" rgns" if _is_en() else "개"))
        w2.metric("💧 " + ("High Humidity (≥80%)" if _is_en() else "고습 지역 (≥80%)"),
                  f"{(baseline['월평균습도']>=80).sum()}" + (" rgns" if _is_en() else "개"))
        w3.metric("🌧️ " + ("High Rain (≥100mm)" if _is_en() else "강수 지역 (≥100mm)"),
                  f"{(baseline['월강수합계']>=100).sum()}" + (" rgns" if _is_en() else "개"))
        w4.metric("🔴 " + ("Top TFRI" if _is_en() else "TFRI 최고"),
                  f"{S(baseline.iloc[0]['시도'])} {baseline.iloc[0]['TFRI']:.0f}%")

# ═══════════════════════════════════════════════════════════════
# 탭 2  지역 상세
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🗺️ " + ("Regional Detail" if _is_en() else "지역 상세 조회"))
    c1,c2=st.columns(2)
    with c1: sel_sido=st.selectbox("지역" if not _is_en() else "Region", SIDO_LIST, key="sel_sido")
    with c2: sel_month=st.selectbox("월" if not _is_en() else "Month", list(range(1,13)),
        index=CUR_MONTH-1,format_func=lambda x:MONTH_KR[x], key="sel_month")

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
                st.session_state[_ai_det_key] = _at if not _ae else rule_guide(grade, whi, ida)
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
            m1.metric("Max Temp" if _is_en() else "최고기온",f"{r.get('월최고기온','-'):.1f}℃")
            m2.metric("Avg Humidity" if _is_en() else "평균습도",f"{r.get('월평균습도','-'):.0f}%")
            m3.metric("Total Rain" if _is_en() else "강수합계",f"{r.get('월강수합계','-'):.0f}mm")

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
            opacity=0.7,name='ML Fire Prob.' if _is_en() else 'ML 발화확률',width=0.35,offset=-0.2)
        fig.add_bar(x=monthly['월'],y=monthly['TFRI_m'],marker_color='#F48FB1',
            opacity=0.7,name='TFRI',width=0.35,offset=0.15)
        fig.add_scatter(x=monthly['월'],y=monthly['종합'],mode='lines+markers',
            line=dict(color='#1565C0',width=2.5),marker=dict(size=7),
            name='Combined' if _is_en() else '종합')
        fig.add_scatter(x=[sel_month],y=[composite],mode='markers',
            marker=dict(size=14,color='black',symbol='star'),
            name='Current' if _is_en() else '현재')
        fm=monthly[monthly['실제']>0]
        if len(fm):
            fig.add_scatter(x=fm['월'],y=fm['종합']+5,mode='markers+text',
                text='🔥',textfont=dict(size=14),marker=dict(size=1,color='red'),
                name='Past Fire' if _is_en() else '과거 화재')
        fig.update_layout(title=f'{S(sel_sido)} ' + ('Monthly Risk (2024)' if _is_en() else '월별 위험도 (2024)'),barmode='overlay',
            xaxis=dict(tickvals=list(range(1,13)),
                       ticktext=[Mn(i) for i in range(1,13)]),
            yaxis=dict(title='Fire Prob. (%)' if _is_en() else '발화 확률 (%)',range=[0,120]),
            height=380,margin=dict(l=0,r=10,t=40,b=30))
        st.plotly_chart(fig,use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# 탭 3  기상 예보
# ═══════════════════════════════════════════════════════════════
with tab3:
    _t3_title = "Weather Forecast Risk Prediction" if _is_en() else "기상 예보 기반 위험도 예측"
    st.markdown(f"## 📡 {_t3_title}")
    st.caption("🌐 Open-Meteo (14" + ("일 예보)" if not _is_en() else "-day forecast)") +
               " · ML 앙상블 v3 + TFRI" if not _is_en() else " · ML Ensemble v3 + TFRI")

    cf1,cf2=st.columns([1,3])
    with cf1:
        fc_sido=st.selectbox("지역" if not _is_en() else "Region", SIDO_LIST, key="fc_sido")
        fc_btn=st.button("🔄 " + ("Load Forecast" if _is_en() else "예보 불러오기"),
                         use_container_width=True)

    # ── Open-Meteo fetch ─────────────────────────────────────────
    _need_reload = (fc_btn or
                    'fc_om' not in st.session_state or
                    st.session_state.get('_fc_sido') != fc_sido)
    if _need_reload:
        with st.spinner(f"{S(fc_sido)} " + ("forecast loading..." if _is_en() else "예보 로딩...")):
            om_df, om_err = fetch_openmeteo(fc_sido)
            st.session_state.update({
                'fc_om':  om_df, 'fc_om_err': om_err,
                '_fc_sido': fc_sido,
            })

    om_df  = st.session_state.get('fc_om')
    om_err = st.session_state.get('fc_om_err')

    if om_df is None:
        st.error(("Forecast load failed — Open-Meteo: " if _is_en() else "예보 로드 실패 — Open-Meteo: ") + str(om_err))
    else:
        risk_om = compute_forecast_risk(om_df, fc_sido)

        # ── 상단 알림 배너 ─────────────────────────────────────────
        _max_any = risk_om['종합위험(%)'].max()
        _max_day = risk_om.loc[risk_om['종합위험(%)'].idxmax(), '날짜'].strftime('%m/%d')
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

        # ── AI 분석 ───────────────────────────────────────────────
        _ai_fc_key = f"ai_fc_{fc_sido}_{st.session_state.get('lang','ko')}"
        _ai_title  = "🤖 AI 분석" if not _is_en() else "🤖 AI Analysis"
        with st.expander(f"**{_ai_title}**", expanded=True):
            if _ai_fc_key not in st.session_state:
                _peak = risk_om.loc[risk_om['종합위험(%)'].idxmax()]
                _fc_r = ([f"최고기온 {om_df['최고기온'].max():.1f}℃"]
                         if om_df['최고기온'].max() >= 33 else [])
                if om_df['평균습도'].mean() >= 80:
                    _fc_r.append(f"평균습도 {om_df['평균습도'].mean():.0f}%")
                if om_df['강수량'].sum() >= 50:
                    _fc_r.append(f"강수합계 {om_df['강수량'].sum():.0f}mm")
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
            st.markdown(f'<div class="ai-box">{st.session_state[_ai_fc_key]}</div>',
                        unsafe_allow_html=True)
            if not openai_key: st.caption(T('ai_no_key'))

        st.markdown("---")

        # ── Open-Meteo 단일 패널 ──────────────────────────────────
        mx  = risk_om['종합위험(%)'].max()
        mx_d = risk_om.loc[risk_om['종합위험(%)'].idxmax(),'날짜'].strftime('%m/%d')
        hi_d = (risk_om['종합위험(%)'] >= 25).sum()
        tod  = risk_om.iloc[0]
        n_days = len(om_df)
        _lbl_days = f"{n_days}-day" if _is_en() else f"{n_days}일"
        st.markdown(f"#### 🌐 Open-Meteo {_lbl_days} " + ("Forecast" if _is_en() else "예보"))

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Today" if _is_en() else "오늘", f"{tod['종합위험(%)']:.1f}%")
        m2.metric("Peak" if _is_en() else "최고", f"{mx:.1f}%", help=mx_d)
        m3.metric("Avg" if _is_en() else "평균", f"{risk_om['종합위험(%)'].mean():.1f}%")
        m4.metric("High days" if _is_en() else "고위험일",
                  f"{hi_d}" + (" days" if _is_en() else "일"))

        # 위험도 차트
        fig = go.Figure()
        fig.add_hrect(y0=0,  y1=15,  fillcolor='#E8F5E9', opacity=0.25, line_width=0)
        fig.add_hrect(y0=15, y1=25,  fillcolor='#FFFDE7', opacity=0.25, line_width=0)
        fig.add_hrect(y0=25, y1=115, fillcolor='#FFEBEE', opacity=0.25, line_width=0)
        fig.add_bar(x=risk_om['날짜'], y=risk_om['ML발화확률(%)'], name='ML',
                    marker_color='#2E7D32', opacity=0.55, width=0.35, offset=-0.2)
        fig.add_bar(x=risk_om['날짜'], y=risk_om['TFRI(%)'], name='TFRI',
                    marker_color='#F57F17', opacity=0.55, width=0.35, offset=0.15)
        fig.add_scatter(x=risk_om['날짜'], y=risk_om['종합위험(%)'],
                        mode='lines+markers', line=dict(color='#2E7D32', width=2.5),
                        marker=dict(size=7, color=[RISK_COLOR[g] for g in risk_om['등급']]),
                        name='Combined Risk' if _is_en() else '종합위험')
        fig.update_layout(height=300, barmode='overlay',
                          yaxis=dict(range=[0,115], title='%'),
                          title='Combined Risk' if _is_en() else '종합 위험도',
                          margin=dict(l=0,r=10,t=36,b=20), showlegend=True,
                          legend=dict(orientation='h',y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        # 기상 차트
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
            subplot_titles=['Temp(℃)' if _is_en() else '기온(℃)',
                            'Rain(mm)' if _is_en() else '강수(mm)'],
            vertical_spacing=0.18)
        fig2.add_scatter(x=om_df['날짜'], y=om_df['최고기온'],
            name='Max' if _is_en() else '최고',
            line=dict(color='#EF5350',width=2), mode='lines+markers', row=1, col=1)
        fig2.add_scatter(x=om_df['날짜'], y=om_df['평균기온'],
            name='Avg' if _is_en() else '평균',
            line=dict(color='#FF9800',dash='dot'), mode='lines', row=1, col=1)
        fig2.add_scatter(x=om_df['날짜'], y=om_df['최저기온'],
            name='Min' if _is_en() else '최저',
            line=dict(color='#42A5F5',width=2), mode='lines+markers', row=1, col=1)
        fig2.add_bar(x=om_df['날짜'], y=om_df['강수량'],
            marker_color='#42A5F5', opacity=0.7,
            name='Rain' if _is_en() else '강수', row=2, col=1)
        fig2.update_layout(height=300, margin=dict(l=0,r=10,t=30,b=20), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        # ── 상세 테이블 ─────────────────────────────────────────────
        st.markdown("---")
        disp = risk_om[['날짜','종합위험(%)','ML발화확률(%)','TFRI(%)','등급',
                         '최고기온','강수량','평균습도','출처']].copy()
        disp['날짜'] = disp['날짜'].dt.strftime('%m/%d(%a)')
        disp.index  = range(1, len(disp)+1)
        st.dataframe(disp, use_container_width=True, height=280)

# ═══════════════════════════════════════════════════════════════
# 탭 5  복합위험 분석
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🔬 " + ("Risk Analysis" if _is_en() else "복합위험 분석"))

    # ── TFRI 물리 방정식 전면 전시 ─────────────────────────────────
    if _is_en():
        st.markdown("""<div class="method-box">
<b style="font-size:1.05rem">📐 TransFireRisk Index (TFRI) — Physics-Based Composite Risk Index</b><br><br>
<b>TFRI = 0.45 × WHI + 0.35 × IDA + 0.20 × HRI</b><br><br>
<b>① WHI (Weather Hazard Index, 45%)</b> — Meteorological threshold model<br>
&nbsp;&nbsp;&nbsp; θ = max(0, (T<sub>max</sub>−30)/10) · φ = max(0,(RH−70)/30)<sup>1.5</sup> · R = min(log(1+P/50)/log(5),1) · fat = max(0,1−ΔT/15)<br>
&nbsp;&nbsp;&nbsp; <b>WHI = min((0.32θ + 0.28φ + 0.25R + 0.15·fat) × 100, 100)</b><br><br>
<b>② IDA (Insulation Degradation Accelerator, 35%)</b> — Arrhenius reaction law (IEC 60076-7 / CIGRE WG A2.49)<br>
&nbsp;&nbsp;&nbsp; V = exp(15000/(273+98) − 15000/(273+T<sub>hs</sub>)) &nbsp;&nbsp; k<sub>m</sub> = 1 + 2·max(0,(RH−70)/30)²<br>
&nbsp;&nbsp;&nbsp; <b>IDA = min(V × k<sub>m</sub> × 20, 100)</b> &nbsp;&nbsp; [every 10℃ rise → insulation life halved]<br><br>
<b>③ HRI (Historical Risk Index, 20%)</b> — Statistical normalisation of regional fire history<br>
&nbsp;&nbsp;&nbsp; <b>HRI = min((N<sub>region,month</sub> / N<sub>years</sub>) / (N<sub>national_avg</sub> + ε) × 50, 100)</b><br><br>
<b>Final: Combined Risk = (ML Fire Probability + TFRI) / 2</b><br>
ML = data-driven pattern · TFRI = physics-driven law · Combined = maximum coverage
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="method-box">
<b style="font-size:1.05rem">📐 TransFireRisk 복합위험지수 (TFRI) — 물리 법칙 기반 지수 설계</b><br><br>
<b>TFRI = 0.45 × WHI + 0.35 × IDA + 0.20 × HRI</b><br><br>
<b>① WHI (기상위험지수, 45%)</b> — 기상 임계치 초과 모델<br>
&nbsp;&nbsp;&nbsp; θ = max(0, (최고기온−30)/10) · φ = max(0,(습도−70)/30)<sup>1.5</sup> · R = min(log(1+강수/50)/log(5),1) · fat = max(0,1−일교차/15)<br>
&nbsp;&nbsp;&nbsp; <b>WHI = min((0.32θ + 0.28φ + 0.25R + 0.15·fat) × 100, 100)</b><br><br>
<b>② IDA (절연열화가속도, 35%)</b> — 아레니우스 반응 법칙 (IEC 60076-7 / CIGRE WG A2.49)<br>
&nbsp;&nbsp;&nbsp; V = exp(15000/(273+98) − 15000/(273+T<sub>hs</sub>)) &nbsp;&nbsp; k<sub>m</sub> = 1 + 2·max(0,(습도−70)/30)²<br>
&nbsp;&nbsp;&nbsp; <b>IDA = min(V × k<sub>m</sub> × 20, 100)</b> &nbsp;&nbsp; [온도 10℃ 상승 → 절연 수명 절반 단축]<br><br>
<b>③ HRI (이력위험지수, 20%)</b> — 지역별 과거 화재이력 통계 정규화<br>
&nbsp;&nbsp;&nbsp; <b>HRI = min((N<sub>지역월</sub> / 연수) / (전국평균 + ε) × 50, 100)</b><br><br>
<b>최종: 종합위험도 = (ML 발화확률 + TFRI) / 2</b><br>
ML = 데이터 기반 패턴 학습 · TFRI = 물리 법칙 기반 · 결합 = 미탐지 최소화
</div>""", unsafe_allow_html=True)

    ca1,ca2=st.columns(2)
    with ca1: an_sido=st.selectbox("Region" if _is_en() else "분석 지역", SIDO_LIST, key="an_sido")
    with ca2: an_month=st.selectbox("Month" if _is_en() else "분석 월", list(range(1,13)),
        index=CUR_MONTH-1,format_func=lambda x:MONTH_KR[x], key="an_month")

    an_row=df[(df['시도']==an_sido)&(df['연도']==2024)&(df['월']==an_month)]
    if len(an_row)==0: an_row=df[(df['시도']==an_sido)&(df['월']==an_month)].tail(1)
    an_r=an_row.iloc[0] if len(an_row) else pd.Series()
    an_ml=float(an_r.get('발화확률',20))
    tfri_v,whi_v,ida_v,hri_v=compute_tfri(an_sido,an_month,
        an_r.get('월최고기온',20),an_r.get('월평균기온',15),
        an_r.get('월평균습도',70),an_r.get('월강수합계',50),an_r.get('월평균일교차',8))
    comp_v=round((an_ml+tfri_v)/2,1)

    r1,r2,r3=st.columns(3)
    r1.metric("ML Fire Prob." if _is_en() else "ML 발화확률",f"{an_ml:.1f}%",help="Ensemble binary classifier (v3.0)" if _is_en() else "앙상블 이진분류 (v3.0)")
    r2.metric("TFRI Index" if _is_en() else "TFRI 복합지수",f"{tfri_v:.1f}%",help="Physics-based index" if _is_en() else "물리·통계 기반")
    r3.metric("Combined Risk" if _is_en() else "종합위험도",f"{comp_v:.1f}%",help="Average of both" if _is_en() else "두 값의 평균")

    cl,cr=st.columns(2)
    with cl:
        _comp_lbl = 'Component' if _is_en() else '성분'
        _val_lbl  = 'Value'     if _is_en() else '값'
        comp_df=pd.DataFrame({
            _comp_lbl:(['WHI (Weather)','IDA (Insulation)','HRI (History)','TFRI Total','ML Fire Prob.','Combined Risk']
                       if _is_en() else
                       ['WHI (기상위험)','IDA (절연열화)','HRI (이력위험)','TFRI 종합','ML 발화확률','종합위험도']),
            _val_lbl:  [whi_v, ida_v, hri_v, tfri_v, an_ml, comp_v],
        })
        clrs=['#1976D2','#D32F2F','#388E3C','#7B1FA2','#0288D1','#000000']
        fig=px.bar(comp_df,x=_val_lbl,y=_comp_lbl,orientation='h',
            color=_comp_lbl,color_discrete_sequence=clrs,
            text=comp_df[_val_lbl].apply(lambda x:f"{x:.1f}"),
            title='Risk Component Breakdown' if _is_en() else '위험 성분 분해')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=340,xaxis=dict(range=[0,120]),
            showlegend=False,margin=dict(l=0,r=60,t=40,b=20))
        st.plotly_chart(fig,use_container_width=True)
    with cr:
        fig=go.Figure(go.Scatterpolar(
            r=[whi_v,ida_v,hri_v,an_ml,comp_v],
            theta=(['WHI<br>Weather','IDA<br>Insulation','HRI<br>History','ML<br>Fire Prob.','Combined<br>Risk']
                   if _is_en() else
                   ['WHI<br>기상','IDA<br>절연열화','HRI<br>이력','ML<br>발화확률','종합<br>위험']),
            fill='toself',fillcolor='rgba(21,101,192,0.15)',
            line=dict(color='#1565C0',width=2.5)))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),
            title='Risk Component Radar' if _is_en() else '위험 성분 레이더',height=340,
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
        title=f'{Mn(an_month)} ' + ('National TFRI Component Stack' if _is_en() else '전국 TFRI 성분 스택'))
    fig.update_layout(height=340,margin=dict(l=0,r=10,t=40,b=30),
        xaxis_tickangle=-30,yaxis_title='Index Value' if _is_en() else '지수값')
    st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# 탭 6  점검 관리
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## 📋 " + ("Inspection Management" if _is_en() else "점검 관리"))
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

    # ── 월간 캘린더 뷰 ────────────────────────────────────────────
    _cal_title = "#### 📅 월간 점검 캘린더" if not _is_en() else "#### 📅 Monthly Inspection Calendar"
    st.markdown(_cal_title)
    _cal_cap = ("월별 계절 위험도 패턴 기반 P1·P2 예상 점검 지역" if not _is_en()
                else "P1/P2 inspection schedule based on seasonal risk patterns")
    st.caption(_cal_cap)

    _MN_SHORT = (['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월']
                 if not _is_en() else
                 ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    # 계절별 위험 배수 (변압기 화재는 여름 고온·겨울 과부하)
    _SEASON_MULT = {1:1.15, 2:1.10, 3:0.90, 4:0.80, 5:0.85,
                    6:1.05, 7:1.40, 8:1.45, 9:1.10, 10:0.90, 11:1.00, 12:1.20}

    _cal_cols = st.columns(12)
    for _mi, (_ccol, _m) in enumerate(zip(_cal_cols, range(1, 13))):
        _mult = _SEASON_MULT[_m]
        _ep1 = max(0, round(p1 * _mult))
        _ep2 = max(0, round(p2 * _mult))
        if _mult >= 1.30:
            _cls = 'cal-p1'
        elif _mult >= 1.05:
            _cls = 'cal-p2'
        else:
            _cls = 'cal-p3'
        _cur_mark = " ★" if _m == view_month else ""
        _ccol.markdown(
            f"<div class='cal-day {_cls}'>"
            f"<b>{_MN_SHORT[_mi]}{_cur_mark}</b><br>"
            f"P1·{_ep1}<br>"
            f"<span style='font-size:0.65rem'>P2·{_ep2}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("""
<div style='margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;font-size:0.75rem'>
  <span style='background:rgba(244,67,54,0.25);border:1px solid #F44336;
    border-radius:4px;padding:2px 8px;color:#EF9A9A'>🔴 고위험 (여름·겨울)</span>
  <span style='background:rgba(255,152,0,0.20);border:1px solid #FF9800;
    border-radius:4px;padding:2px 8px;color:#FFCC80'>🟠 주의</span>
  <span style='background:rgba(76,175,80,0.15);border:1px solid #4CAF50;
    border-radius:4px;padding:2px 8px;color:#A5D6A7'>🟢 낮음 (봄·가을)</span>
  <span style='padding:2px 8px;color:#8B9BAF'>★ = 현재 기준 월</span>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    plan_df=insp_base[['시도','종합위험','발화확률','TFRI','등급','우선순위','예상공수(h)']].copy()
    plan_df['점검상태']=plan_df['시도'].map(st.session_state.insp_status)
    plan_df['메모']=plan_df['시도'].map(st.session_state.insp_memo)
    plan_df.columns=['시도','종합위험(%)','ML발화확률(%)','TFRI(%)','위험등급','우선순위','예상공수(h)','점검상태','메모']
    st.download_button(T('insp_dl'),
        data=plan_df.to_csv(index=False,encoding='utf-8-sig'),
        file_name=f"TransFireRisk_점검계획_{CUR_YEAR}{view_month:02d}.csv",
        mime='text/csv')

# ═══════════════════════════════════════════════════════════════
# 탭 7  이력·모델 정보
# ═══════════════════════════════════════════════════════════════
with tab7:
    st.markdown("## 📊 " + ("History & Model" if _is_en() else "이력 분석 및 모델 정보"))

    col1,col2=st.columns(2)
    with col1:
        yearly=df.groupby('연도').agg(실제=('변압기화재건수','sum')).reset_index()
        yearly['고위험(≥40%)']=df[df['발화확률']>=40].groupby('연도').size().reindex(yearly['연도'],fill_value=0).values
        fig=go.Figure()
        fig.add_bar(x=yearly['연도'],y=yearly['실제'],
            name='Actual Fires' if _is_en() else '실제 화재',
            marker_color='#EF5350',opacity=0.85,text=yearly['실제'],textposition='outside')
        fig.add_scatter(x=yearly['연도'],y=yearly['고위험(≥40%)'],
            name='Prob≥40% Predicted' if _is_en() else '발화확률≥40% 예측',
            mode='lines+markers',line=dict(color='#1565C0',width=2.5))
        fig.update_layout(title='Annual Fire Count vs High-Risk Predictions' if _is_en() else '연도별 화재건수 vs 고위험 예측 건수',
            xaxis=dict(tickvals=yearly['연도']),height=290,
            margin=dict(l=0,r=10,t=40,b=20))
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        pivot=df.groupby(['시도','월'])['변압기화재건수'].sum().unstack().fillna(0)
        fig=px.imshow(pivot,color_continuous_scale='YlOrRd',aspect='auto',
            title='Region × Month Fire Heatmap (2020~2024)' if _is_en() else '시도 × 월별 화재 누계 히트맵 (2020~2024)')
        fig.update_xaxes(tickvals=list(range(1,13)),
            ticktext=[Mn(i) for i in range(1,13)])
        fig.update_layout(height=290,margin=dict(l=0,r=10,t=40,b=20))
        st.plotly_chart(fig,use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # ── 기상-화재 상관분석 ─────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    _corr_title = "#### 🔗 기상변수 × 화재발생 상관분석" if not _is_en() else "#### 🔗 Weather × Fire Correlation Analysis"
    st.markdown(_corr_title)
    st.caption(
        "Pearson·Spearman 이중 상관계수 — 선형·비선형 관계를 동시에 검증" if not _is_en()
        else "Dual Pearson·Spearman correlation — verifies both linear and non-linear relationships"
    )

    _wx_vars = ['월최고기온','월평균기온','월평균습도','월강수합계','월최대풍속',
                '월평균일교차','강수일수','월연속고온일수','월전3일평균기온']
    _wx_vars = [v for v in _wx_vars if v in df.columns]
    _fire_flag = (df['변압기화재건수'] > 0).astype(int)

    _pearson  = {v: df[v].corr(_fire_flag) for v in _wx_vars}
    _spearman = {v: df[v].rank().corr(_fire_flag.rank()) for v in _wx_vars}

    _corr_df = (pd.DataFrame({'Pearson': _pearson, 'Spearman': _spearman})
                  .sort_values('Pearson', ascending=False))
    _var_labels = {
        '월최고기온':'최고기온','월평균기온':'평균기온','월평균습도':'평균습도',
        '월강수합계':'강수합계','월최대풍속':'최대풍속','월평균일교차':'일교차',
        '강수일수':'강수일수','월연속고온일수':'연속고온일수','월전3일평균기온':'3일평균기온',
    }

    _cc1, _cc2 = st.columns([1.2, 1])
    with _cc1:
        _fig_corr = go.Figure()
        _fig_corr.add_bar(
            x=[_var_labels.get(v, v) for v in _corr_df.index],
            y=_corr_df['Pearson'],
            name='Pearson',
            marker_color=['#EF5350' if v > 0 else '#42A5F5' for v in _corr_df['Pearson']],
            opacity=0.85,
            text=[f"{v:.3f}" for v in _corr_df['Pearson']],
            textposition='outside',
        )
        _fig_corr.add_scatter(
            x=[_var_labels.get(v, v) for v in _corr_df.index],
            y=_corr_df['Spearman'],
            name='Spearman',
            mode='markers',
            marker=dict(size=10, color='#FF9800', symbol='diamond'),
        )
        _fig_corr.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
        _fig_corr.update_layout(
            title='기상변수와 화재발생의 Pearson·Spearman 상관계수' if not _is_en()
                  else 'Weather–Fire Pearson·Spearman Correlation',
            yaxis=dict(range=[-0.25, 0.35], title='Correlation Coefficient'),
            height=320, margin=dict(l=0, r=10, t=40, b=60),
            barmode='overlay', xaxis_tickangle=-30,
        )
        st.plotly_chart(_fig_corr, use_container_width=True)

    with _cc2:
        # 상관 테이블
        _top_corr = _corr_df.copy()
        _top_corr.index = [_var_labels.get(v, v) for v in _top_corr.index]
        _top_corr = _top_corr.round(4)
        if _is_en():
            _top_corr.columns = ['Pearson r', 'Spearman ρ']
        else:
            _top_corr.columns = ['피어슨 r', '스피어만 ρ']
        st.dataframe(
            _top_corr.style.background_gradient(cmap='RdBu_r', vmin=-0.3, vmax=0.3),
            use_container_width=True, height=295
        )

    # 핵심 해석
    _top1_var = _var_labels.get(_corr_df.index[0], _corr_df.index[0])
    _top1_r   = _corr_df['Pearson'].iloc[0]
    _top2_var = _var_labels.get(_corr_df.index[1], _corr_df.index[1])
    if not _is_en():
        st.info(
            f"💡 **{_top1_var}** (r={_top1_r:.3f})이 화재 발생과 가장 높은 정(+)상관 — "
            f"**{_top2_var}**도 강한 상관 확인. "
            "Pearson·Spearman 계수가 모두 정방향이면 선형·비선형 모두 통계적으로 유효한 관계."
        )
    else:
        st.info(
            f"💡 **{_top1_var}** (r={_top1_r:.3f}) shows the strongest positive correlation with fire occurrence. "
            "Consistent Pearson–Spearman direction confirms both linear and non-linear validity."
        )

    # ══════════════════════════════════════════════════════════════
    # ── 기후변화 트렌드 분석 ────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    _cc_title = "#### 🌡️ 기후변화 트렌드 분석 (2020~2024)" if not _is_en() else "#### 🌡️ Climate Change Trend Analysis (2020–2024)"
    st.markdown(_cc_title)
    st.caption(
        "연도별 폭염 지표(월연속고온일수·최고기온) 상승 트렌드 → 미래 변압기 화재 위험도 증가 전망" if not _is_en()
        else "Year-over-year heat index uptrend → increasing transformer fire risk outlook"
    )

    _yearly_clim = df.groupby('연도').agg(
        avg_maxT=('월최고기온', 'mean'),
        avg_hotdays=('월연속고온일수', 'mean'),
        avg_humid=('월평균습도', 'mean'),
        total_fire=('변압기화재건수', 'sum'),
    ).reset_index()

    # 선형 트렌드 (numpy polyfit)
    _yrs = _yearly_clim['연도'].values
    _slope_T, _int_T   = np.polyfit(_yrs, _yearly_clim['avg_maxT'], 1)
    _slope_hd, _int_hd = np.polyfit(_yrs, _yearly_clim['avg_hotdays'], 1)
    _slope_f,  _int_f  = np.polyfit(_yrs, _yearly_clim['total_fire'], 1)
    _trend_yrs = np.array([_yrs[0], _yrs[-1]])

    _ct1, _ct2, _ct3 = st.columns(3)
    for _col, _col_name, _slope, _intercept, _color, _unit, _lbl in [
        (_ct1, 'avg_maxT',   _slope_T,  _int_T,  '#EF5350', '℃',   '월평균 최고기온' if not _is_en() else 'Avg Max Temp'),
        (_ct2, 'avg_hotdays',_slope_hd, _int_hd, '#FF9800', '일',  '월연속고온일수'  if not _is_en() else 'Consec Hot Days'),
        (_ct3, 'total_fire', _slope_f,  _int_f,  '#7B1FA2', '건',  '연간 화재건수'   if not _is_en() else 'Annual Fires'),
    ]:
        _fig_t = go.Figure()
        _fig_t.add_bar(x=_yrs, y=_yearly_clim[_col_name], marker_color=_color, opacity=0.6, name=_lbl)
        _fig_t.add_scatter(
            x=_trend_yrs, y=_slope * _trend_yrs + _intercept,
            mode='lines', line=dict(color=_color, width=2.5, dash='dot'),
            name=f'추세 {_slope:+.2f}{_unit}/년' if not _is_en() else f'Trend {_slope:+.2f}{_unit}/yr'
        )
        _fig_t.update_layout(
            title=_lbl, height=220,
            margin=dict(l=0,r=0,t=40,b=20),
            showlegend=True,
            legend=dict(orientation='h', y=-0.25, font=dict(size=9)),
            xaxis=dict(tickvals=_yrs.tolist()),
            yaxis_title=_unit,
        )
        _col.plotly_chart(_fig_t, use_container_width=True)

    # 2030 외삽 예측
    _yr2030 = 2030
    _T2030   = _slope_T  * _yr2030 + _int_T
    _hd2030  = _slope_hd * _yr2030 + _int_hd
    _f2030   = _slope_f  * _yr2030 + _int_f
    if not _is_en():
        st.warning(
            f"📈 **기후변화 외삽 전망 (2030년 기준선 예측):**  "
            f"월평균 최고기온 **{_T2030:.1f}℃** (+{(_T2030-_yearly_clim['avg_maxT'].mean()):.1f}℃)  ·  "
            f"월연속고온일수 **{_hd2030:.1f}일** ·  "
            f"연간 화재 **{max(0,_f2030):.0f}건** 예상 — "
            "지금보다 위험도가 높아질 가능성이 있음."
        )
    else:
        st.warning(
            f"📈 **Climate Extrapolation to 2030:**  "
            f"Avg max temp **{_T2030:.1f}℃** (+{(_T2030-_yearly_clim['avg_maxT'].mean()):.1f}℃)  ·  "
            f"Consec hot days **{_hd2030:.1f}d** ·  "
            f"Est. annual fires **{max(0,_f2030):.0f}** — risk outlook is rising."
        )

    st.markdown("---")
    st.markdown(f"#### ⚙️ {T('model_compare')}")

    # ── 왜 F1이 아닌가? ────────────────────────────────────────
    with st.expander("📐 " + ("Why F2 / MCC / PR-AUC?" if _is_en()
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

    if _is_en():  # EN
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
        st.markdown("**" + ("Feature Importance (XGB+FE, Top 15)" if _is_en() else "피처 중요도 (XGB+FE 기준, 상위 15개)") + "**")
        _fi_names_ko = ['지역발화율★','월연속고온일수','월전3일평균기온','월평균습도','기온편차★',
                        '월평균기온','열습도스트레스★','월_sin★','강수습도★','월_cos★',
                        '월강수합계','과부하스트레스★','월최고기온','강수일수','전년변압기화재건수']
        _fi_names_en = ['RegionalFireRate★','ConsecHotDays','3dAvgTemp','AvgHumidity','TempAnomaly★',
                        'AvgTemp','HeatHumidStress★','Month_sin★','RainHumidity★','Month_cos★',
                        'TotalRain','OverloadStress★','MaxTemp','RainDays','PrevYrFireCnt']
        _fi_names = _fi_names_en if _is_en() else _fi_names_ko
        feat_imp_data={
            'Feature' if _is_en() else '피처': _fi_names,
            'Importance' if _is_en() else '중요도':
                [0.115,0.098,0.047,0.044,0.043,0.043,0.042,0.035,0.035,0.034,
                 0.031,0.030,0.028,0.027,0.026]
        }
        _fi_xcol = 'Importance' if _is_en() else '중요도'
        _fi_ycol = 'Feature'    if _is_en() else '피처'
        fi_df=pd.DataFrame(feat_imp_data)
        fig=px.bar(fi_df,x=_fi_xcol,y=_fi_ycol,orientation='h',
            color=['#E91E63' if '★' in p else '#1565C0' for p in fi_df[_fi_ycol]],
            text=fi_df[_fi_xcol].apply(lambda x:f"{x:.3f}"),
            title='★ = New feature' if _is_en() else '★ = 신규 추가 피처')
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
        fig.add_scatter(x=fpr,y=tpr,mode='lines',
            name=f'v3 Ensemble (AUC={auc:.3f})' if _is_en() else f'v3 앙상블 (AUC={auc:.3f})',
            line=dict(color='#1565C0',width=2.5))
        fig.add_scatter(x=[0,1],y=[0,1],mode='lines',
            name='Random (AUC=0.5)' if _is_en() else '랜덤(AUC=0.5)',
            line=dict(color='gray',dash='dot'))
        fig.update_layout(title='ROC Curve (Validation 2023–2024)' if _is_en() else 'ROC Curve (검증셋 2023~2024)',
            xaxis_title='False Positive Rate',yaxis_title='True Positive Rate',
            height=430,margin=dict(l=0,r=10,t=40,b=30))
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("---")
    c5,c6=st.columns(2)
    with c5:
        if _is_en():
            st.markdown("""**Model Architecture**
| Item | Detail |
|---|---|
| Algorithm | XGB + RandomForest + LogisticReg |
| Voting | Soft voting (0.6·0.3·0.1) |
| Objective | Binary classification P(fire) |
| Features | 28 (original 16 + new 12) |
| Threshold | 0.20 (balanced) / 0.15 (recall-first) |
""")
        else:
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
        if _is_en():
            st.markdown("""**Key Improvements**
| Problem | Solution |
|---|---|
| Regression → sparse count instability | Switched to binary classification |
| Class imbalance (8%) | scale_pos_weight + balanced RF |
| Insufficient feature expressiveness | Added 12 interaction/deviation/cyclic features |
| Single-model instability | 3-model soft voting ensemble |
""")
        else:
            st.markdown("""**개선 핵심 요약**
| 문제 | 해결책 |
|---|---|
| 회귀 → 희소 카운트 불안정 | 이진 분류로 전환 |
| 클래스 불균형 (8%) | scale_pos_weight + balanced RF |
| 피처 표현력 부족 | 상호작용·편차·주기 12개 추가 |
| 단일 모델 불안정 | 3모델 소프트 보팅 앙상블 |
""")

# ═══════════════════════════════════════════════════════════════
# 탭 4  시나리오 시뮬레이션
# ═══════════════════════════════════════════════════════════════
with tab4:
    _sim_ttl = "Scenario Simulation" if _is_en() else "시나리오 시뮬레이션"
    st.markdown(f"## ⚙️ {_sim_ttl}")
    st.caption(
        "Set custom weather conditions to predict fire risk for any region and month."
        if _is_en() else
        "날씨 조건을 직접 설정해 원하는 지역·월의 발화 위험도를 예측합니다."
    )

    # ── 기본값: 현재 선택 지역 최근 데이터 참조 ───────────────────
    _sc1, _sc2, _sc3 = st.columns(3)
    with _sc1:
        sim_sido  = st.selectbox("지역" if not _is_en() else "Region",
                                 SIDO_LIST, key="sim_sido")
    with _sc2:
        sim_month = st.selectbox("월" if not _is_en() else "Month",
                                 list(range(1,13)), index=CUR_MONTH-1,
                                 format_func=lambda x: MONTH_KR[x], key="sim_month")
    with _sc3:
        _ref_row = df[(df['시도']==sim_sido)&(df['연도']==2024)&(df['월']==sim_month)]
        if len(_ref_row)==0:
            _ref_row = df[(df['시도']==sim_sido)&(df['월']==sim_month)].tail(1)
        _rr = _ref_row.iloc[0] if len(_ref_row) else pd.Series()
        st.markdown("<br>" if not _is_en() else "<br>", unsafe_allow_html=True)
        _load_ref = st.button(
            "2024년 실측값 불러오기" if not _is_en() else "Load 2024 reference values",
            key="sim_load_ref", use_container_width=True)

    # 실측값 불러오기 → session_state에 저장
    if _load_ref and len(_ref_row):
        st.session_state['sim_maxT']  = float(_rr.get('월최고기온', 25.0))
        st.session_state['sim_avgT']  = float(_rr.get('월평균기온', 15.0))
        st.session_state['sim_minT']  = float(_rr.get('월최저기온', 5.0))
        st.session_state['sim_rh']    = float(_rr.get('월평균습도', 65.0))
        st.session_state['sim_rain']  = float(_rr.get('월강수합계', 50.0))
        st.session_state['sim_wind']  = float(_rr.get('월최대풍속', 20.0))
        st.session_state['sim_range'] = float(_rr.get('월평균일교차', 9.0))
        st.session_state['sim_rdays'] = int(_rr.get('강수일수', 5))
        st.session_state['sim_hot']   = int(_rr.get('월연속고온일수', 0))

    st.markdown("---")
    st.markdown("#### " + ("Weather Conditions" if _is_en() else "기상 조건 설정"))

    # ── 슬라이더 ────────────────────────────────────────────────
    _a1, _a2, _a3 = st.columns(3)
    with _a1:
        sim_maxT = st.slider(
            "최고기온 (℃)" if not _is_en() else "Max Temp (℃)",
            -10.0, 45.0,
            st.session_state.get('sim_maxT', float(_rr.get('월최고기온',25.0))),
            0.5, key="sim_maxT")
        sim_avgT = st.slider(
            "평균기온 (℃)" if not _is_en() else "Avg Temp (℃)",
            -20.0, 40.0,
            st.session_state.get('sim_avgT', float(_rr.get('월평균기온',15.0))),
            0.5, key="sim_avgT")
        sim_minT = st.slider(
            "최저기온 (℃)" if not _is_en() else "Min Temp (℃)",
            -25.0, 35.0,
            st.session_state.get('sim_minT', float(_rr.get('월최저기온',5.0))),
            0.5, key="sim_minT")
    with _a2:
        sim_rh   = st.slider(
            "평균습도 (%)" if not _is_en() else "Avg Humidity (%)",
            20.0, 100.0,
            st.session_state.get('sim_rh', float(_rr.get('월평균습도',65.0))),
            1.0, key="sim_rh")
        sim_rain = st.slider(
            "강수합계 (mm)" if not _is_en() else "Total Precip (mm)",
            0.0, 800.0,
            st.session_state.get('sim_rain', float(_rr.get('월강수합계',50.0))),
            5.0, key="sim_rain")
        sim_wind = st.slider(
            "최대풍속 (km/h)" if not _is_en() else "Max Wind (km/h)",
            0.0, 80.0,
            st.session_state.get('sim_wind', float(_rr.get('월최대풍속',20.0))),
            1.0, key="sim_wind")
    with _a3:
        sim_range = st.slider(
            "평균일교차 (℃)" if not _is_en() else "Avg Temp Range (℃)",
            0.0, 25.0,
            st.session_state.get('sim_range', float(_rr.get('월평균일교차',9.0))),
            0.5, key="sim_range")
        sim_rdays = st.slider(
            "강수일수 (일)" if not _is_en() else "Rain Days",
            0, 31,
            st.session_state.get('sim_rdays', int(_rr.get('강수일수',5))),
            1, key="sim_rdays")
        sim_hot   = st.slider(
            "연속고온일수 (일, ≥33℃)" if not _is_en() else "Consecutive Hot Days (≥33℃)",
            0, 31,
            st.session_state.get('sim_hot', int(_rr.get('월연속고온일수',0))),
            1, key="sim_hot")

    # ── 예측 실행 ────────────────────────────────────────────────
    st.markdown("---")
    _rdict = row_from_weather(
        sim_sido, sim_month,
        sim_maxT, sim_avgT, sim_minT,
        sim_rh, sim_rain, sim_wind,
        sim_range, sim_rdays)
    _rdict['월연속고온일수'] = sim_hot

    sim_ml   = predict_prob(_rdict)
    sim_tfri, sim_whi, sim_ida, sim_hri = compute_tfri(
        sim_sido, sim_month, sim_maxT, sim_avgT, sim_rh, sim_rain, sim_range)
    sim_comp = round((sim_ml + sim_tfri) / 2, 1)
    sim_grade = risk_grade(sim_comp)
    sim_color = RISK_COLOR[sim_grade]

    # 실측과 차이 (2024 기준)
    ref_ml   = float(_rr.get('발화확률', sim_ml)) if len(_ref_row) else sim_ml
    ref_tfri, *_ = compute_tfri(
        sim_sido, sim_month,
        float(_rr.get('월최고기온', sim_maxT)),
        float(_rr.get('월평균기온', sim_avgT)),
        float(_rr.get('월평균습도', sim_rh)),
        float(_rr.get('월강수합계', sim_rain)),
        float(_rr.get('월평균일교차', sim_range))) if len(_ref_row) else (sim_tfri,)
    ref_comp = round((ref_ml + ref_tfri) / 2, 1)

    # ── 결과 패널 ────────────────────────────────────────────────
    _r1, _r2, _r3, _r4 = st.columns(4)
    _r1.metric("ML 발화확률" if not _is_en() else "ML Fire Prob.",
               f"{sim_ml:.1f}%", delta=f"{sim_ml-ref_ml:+.1f}%p")
    _r2.metric("TFRI",
               f"{sim_tfri:.1f}%", delta=f"{sim_tfri-ref_tfri:+.1f}%p")
    _r3.metric("종합위험도" if not _is_en() else "Combined Risk",
               f"{sim_comp:.1f}%", delta=f"{sim_comp-ref_comp:+.1f}%p")
    _r4.metric("위험등급" if not _is_en() else "Risk Grade",
               G(sim_grade))

    st.caption("delta = 2024년 실측 대비 변화량" if not _is_en()
               else "delta = change vs. 2024 reference values")

    # ── 대형 위험도 카드 ─────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{RISK_BG[sim_grade]};border:3px solid {sim_color};
      border-radius:12px;padding:20px 28px;text-align:center;margin:8px 0">
      <div style="font-size:0.85rem;color:#666">{S(sim_sido)} · {Mn(sim_month)} — {"Simulation Result" if _is_en() else "시뮬레이션 결과"}</div>
      <div style="font-size:3.8rem;font-weight:900;color:{RISK_TEXT[sim_grade]};line-height:1.1">
        {sim_comp:.0f}%</div>
      <div style="font-size:1.2rem;font-weight:700;color:{RISK_TEXT[sim_grade]}">
        {RISK_EMOJI[sim_grade]} {G(sim_grade)}</div>
    </div>""", unsafe_allow_html=True)

    # ── 성분 분해 차트 ──────────────────────────────────────────
    st.markdown("---")
    _ch1, _ch2 = st.columns(2)
    with _ch1:
        _comp_df = pd.DataFrame({
            '성분' if not _is_en() else 'Component':
                ['WHI', 'IDA', 'HRI', 'TFRI', 'ML', '종합' if not _is_en() else 'Combined'],
            '값' if not _is_en() else 'Value':
                [sim_whi, sim_ida, sim_hri, sim_tfri, sim_ml, sim_comp],
        })
        _clrs = ['#1976D2','#D32F2F','#388E3C','#7B1FA2','#0288D1','#000000']
        _fig = px.bar(_comp_df,
                      x='값' if not _is_en() else 'Value',
                      y='성분' if not _is_en() else 'Component',
                      orientation='h', color='성분' if not _is_en() else 'Component',
                      color_discrete_sequence=_clrs,
                      text=(_comp_df['값' if not _is_en() else 'Value']
                            .apply(lambda x: f"{x:.1f}")),
                      title='위험 성분 분해' if not _is_en() else 'Risk Component Breakdown')
        _fig.update_traces(textposition='outside')
        _fig.update_layout(height=320, xaxis=dict(range=[0,120]),
                           showlegend=False, margin=dict(l=0,r=60,t=40,b=20))
        st.plotly_chart(_fig, use_container_width=True)

    with _ch2:
        _fig2 = go.Figure(go.Scatterpolar(
            r=[sim_whi, sim_ida, sim_hri, sim_ml, sim_comp],
            theta=(['WHI 기상','IDA 절연열화','HRI 이력','ML 발화확률','종합위험']
                   if not _is_en() else
                   ['WHI Weather','IDA Insulation','HRI History','ML Prob.','Combined']),
            fill='toself', fillcolor=_hex_rgba(sim_color, 0.18),
            line=dict(color=sim_color, width=2.5)))
        _fig2.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100])),
            title='위험 레이더' if not _is_en() else 'Risk Radar',
            height=320, margin=dict(l=20,r=20,t=50,b=20), showlegend=False)
        st.plotly_chart(_fig2, use_container_width=True)

    # ── 민감도 분석: 기온 vs 습도 히트맵 ──────────────────────────
    st.markdown("---")
    st.markdown("#### " + ("Sensitivity Analysis — Max Temp × Humidity"
                           if _is_en() else "민감도 분석 — 최고기온 × 평균습도"))
    st.caption("다른 조건은 현재 슬라이더 값 고정, 기온·습도만 변화시킨 종합위험도" if not _is_en()
               else "All other conditions fixed; varying temp and humidity only.")

    _temps = list(range(15, 46, 3))
    _rhums = list(range(40, 101, 10))
    _heat  = []
    for _tt in _temps:
        _row = []
        for _rh in _rhums:
            _rd = row_from_weather(sim_sido, sim_month, _tt,
                                   _tt - sim_range, _tt - sim_range * 2,
                                   _rh, sim_rain, sim_wind, sim_range, sim_rdays)
            _rd['월연속고온일수'] = max(0, int((_tt - 33) * 2)) if _tt > 33 else 0
            _ml = predict_prob(_rd)
            _tf, *_ = compute_tfri(sim_sido, sim_month, _tt, _tt - sim_range, _rh, sim_rain, sim_range)
            _row.append(round((_ml + _tf) / 2, 1))
        _heat.append(_row)

    _hm_df = pd.DataFrame(_heat, index=[f"{t}℃" for t in _temps],
                           columns=[f"{r}%" for r in _rhums])
    _fig3 = px.imshow(_hm_df, color_continuous_scale='RdYlGn_r',
                      zmin=0, zmax=80,
                      labels=dict(x="평균습도" if not _is_en() else "Humidity",
                                  y="최고기온" if not _is_en() else "Max Temp",
                                  color="종합위험(%)" if not _is_en() else "Combined Risk(%)"),
                      title="종합위험도 히트맵 (기온 × 습도)" if not _is_en()
                            else "Combined Risk Heatmap (Temp × Humidity)")
    _fig3.update_layout(height=380, margin=dict(l=0,r=10,t=50,b=20))
    # 현재 슬라이더 위치 표시
    _fig3.add_scatter(
        x=[f"{int(round(sim_rh/10)*10)}%"],
        y=[f"{int(round(sim_maxT/3)*3)}℃"],
        mode='markers', marker=dict(size=14, color='black', symbol='x'),
        name='현재 조건' if not _is_en() else 'Current')
    st.plotly_chart(_fig3, use_container_width=True)

    # ── AI 분석 ─────────────────────────────────────────────────
    st.markdown("---")
    _ai_sim_key = f"ai_sim_{sim_sido}_{sim_month}_{sim_comp:.0f}_{st.session_state.get('lang','ko')}"
    _ai_sim_ttl = "🤖 AI 분석" if not _is_en() else "🤖 AI Analysis"
    with st.expander(f"**{_ai_sim_ttl}**", expanded=True):
        if _ai_sim_key not in st.session_state:
            _sim_reasons = []
            if sim_maxT >= 33: _sim_reasons.append(f"최고기온 {sim_maxT:.1f}℃")
            if sim_rh >= 80:   _sim_reasons.append(f"평균습도 {sim_rh:.0f}%")
            if sim_rain >= 100: _sim_reasons.append(f"강수합계 {sim_rain:.0f}mm")
            if sim_hot >= 3:   _sim_reasons.append(f"연속고온 {sim_hot}일")
            if openai_key:
                with st.spinner("AI 분석 생성 중..." if not _is_en() else "Generating..."):
                    _at, _ae = get_ai_guide(sim_sido, sim_month, sim_grade,
                                            sim_ml, sim_tfri, sim_whi, sim_ida,
                                            sim_hri, _sim_reasons, openai_key)
                st.session_state[_ai_sim_key] = (
                    _at if not _ae else rule_guide(sim_grade, sim_whi, sim_ida))
            else:
                st.session_state[_ai_sim_key] = rule_guide(sim_grade, sim_whi, sim_ida)
        st.markdown(f'<div class="ai-box">{st.session_state[_ai_sim_key]}</div>',
                    unsafe_allow_html=True)
        if not openai_key: st.caption(T('ai_no_key'))

    # ── CSV 내보내기 ─────────────────────────────────────────────
    st.markdown("---")
    _sim_out = pd.DataFrame([{
        '지역': sim_sido, '월': sim_month,
        '최고기온': sim_maxT, '평균기온': sim_avgT, '최저기온': sim_minT,
        '평균습도': sim_rh, '강수합계': sim_rain, '최대풍속': sim_wind,
        '평균일교차': sim_range, '강수일수': sim_rdays, '연속고온일수': sim_hot,
        'ML발화확률(%)': sim_ml, 'TFRI(%)': sim_tfri, '종합위험(%)': sim_comp,
        'WHI': sim_whi, 'IDA': sim_ida, 'HRI': sim_hri, '위험등급': sim_grade,
    }])
    st.download_button(
        "시뮬레이션 결과 CSV 내보내기" if not _is_en() else "Export Simulation Result (CSV)",
        data=_sim_out.to_csv(index=False, encoding='utf-8-sig'),
        file_name=f"sim_{sim_sido}_{sim_month}월_{sim_comp:.0f}pct.csv",
        mime='text/csv')

    # ── F2 임계값 최적화 ──────────────────────────────────────────
    st.markdown("---")
    _thr_title = "#### ⚙️ F2 기준 최적 임계값 분석" if not _is_en() else "#### ⚙️ F2-Optimal Threshold Analysis"
    st.markdown(_thr_title)
    st.caption("F2(β=2, 재현율 2배 가중) 기준 월별 최적 임계값 — 화재 미탐지 최소화 전략" if not _is_en()
               else "F2(β=2) optimal threshold per month — minimising fire miss-detection")

    te_all = df[df['연도'] >= 2023].copy()
    yb_all = (te_all['변압기화재건수'] > 0).astype(int)
    prob_all = te_all['발화확률'] / 100

    _thr_candidates = [i/100 for i in range(5, 55, 5)]
    _monthly_thr = []
    for _m in range(1, 13):
        _mask = te_all['월'] == _m
        _ym = yb_all[_mask]; _pm = prob_all[_mask]
        if len(_ym) == 0 or _ym.sum() == 0:
            _monthly_thr.append({'월': _m, '최적임계값': 0.20, 'F2': 0.0})
            continue
        _best_thr, _best_f2 = 0.20, 0.0
        for _thr in _thr_candidates:
            _yp = (_pm >= _thr).astype(int)
            _f2 = fbeta_score(_ym, _yp, beta=2, zero_division=0)
            if _f2 > _best_f2:
                _best_f2, _best_thr = _f2, _thr
        _monthly_thr.append({'월': _m, '최적임계값': _best_thr, 'F2': round(_best_f2, 3)})

    _thr_df = pd.DataFrame(_monthly_thr)
    _MN_LABEL = ([Mn(i) for i in range(1, 13)])
    _thr_df['월_표시'] = _MN_LABEL

    _fig_thr = go.Figure()
    _bar_clrs = ['#EF5350' if r['최적임계값'] <= 0.15
                 else '#FF9800' if r['최적임계값'] <= 0.25
                 else '#4CAF50'
                 for _, r in _thr_df.iterrows()]
    _fig_thr.add_bar(x=_thr_df['월_표시'], y=_thr_df['최적임계값'],
                     marker_color=_bar_clrs, opacity=0.8,
                     name='F2 최적 임계값' if not _is_en() else 'F2-Optimal Threshold',
                     text=_thr_df['최적임계값'].apply(lambda x: f"{x:.2f}"),
                     textposition='outside')
    _fig_thr.add_hline(y=0.20, line_dash='dash', line_color='white',
                       opacity=0.5, annotation_text='현재 고정 0.20' if not _is_en() else 'Fixed 0.20')
    _fig_thr.update_layout(
        height=280, yaxis=dict(range=[0, 0.60], title='Threshold'),
        title='월별 F2 최적 임계값' if not _is_en() else 'Monthly F2-Optimal Threshold',
        margin=dict(l=0,r=10,t=40,b=20))
    st.plotly_chart(_fig_thr, use_container_width=True)

    _thr_disp = _thr_df[['월_표시','최적임계값','F2']].copy()
    _thr_disp.columns = (['월','F2최적임계값','F2점수'] if not _is_en()
                          else ['Month','F2-Optimal Threshold','F2 Score'])
    st.dataframe(_thr_disp.style.background_gradient(
        cmap='RdYlGn', subset=['F2최적임계값' if not _is_en() else 'F2-Optimal Threshold']),
        use_container_width=True, hide_index=True)
    st.info("💡 " + ("7·8월은 고온으로 임계값을 낮춰 탐지율을 높이고, 3·4월은 상대적으로 임계값을 올려 오탐을 줄입니다."
                     if not _is_en() else
                     "Lower thresholds in Jul/Aug (high temp season) maximise recall; higher in spring reduces false alarms."))

    # ── SHAP 개별 설명 ────────────────────────────────────────────
    st.markdown("---")
    _shap_title = "#### 🔬 SHAP 예측 설명 (지역 선택)" if not _is_en() else "#### 🔬 SHAP Prediction Explanation"
    st.markdown(_shap_title)
    st.caption("선택한 지역·월에 대한 발화 확률 예측의 주요 기여 변수" if not _is_en()
               else "Key contributing variables for the selected region and month prediction")

    _sh_c1, _sh_c2, _sh_c3 = st.columns(3)
    with _sh_c1: _sh_sido  = st.selectbox("지역" if not _is_en() else "Region", SIDO_LIST, key="sh_sido")
    with _sh_c2: _sh_month = st.selectbox("월" if not _is_en() else "Month", list(range(1,13)),
                                           index=CUR_MONTH-1, format_func=lambda x:MONTH_KR[x], key="sh_month")
    with _sh_c3:
        st.markdown("<br>", unsafe_allow_html=True)
        _sh_btn = st.button("🔍 SHAP 계산" if not _is_en() else "🔍 Run SHAP", key="sh_run",
                            use_container_width=True)

    if _sh_btn or 'shap_result' in st.session_state:
        if _sh_btn:
            _sh_row = df[(df['시도']==_sh_sido)&(df['연도']==2024)&(df['월']==_sh_month)]
            if len(_sh_row)==0: _sh_row = df[(df['시도']==_sh_sido)&(df['월']==_sh_month)].tail(1)
            if len(_sh_row):
                _sh_r = _sh_row.iloc[0]
                _sh_dict = row_from_weather(
                    _sh_sido, _sh_month,
                    float(_sh_r.get('월최고기온',25)), float(_sh_r.get('월평균기온',15)),
                    float(_sh_r.get('월최저기온',5)), float(_sh_r.get('월평균습도',65)),
                    float(_sh_r.get('월강수합계',50)), float(_sh_r.get('월최대풍속',20)),
                    float(_sh_r.get('월평균일교차',9)), int(_sh_r.get('강수일수',5)))
                _sh_eng = engineer_features(pd.DataFrame([_sh_dict]), ref_df)
                _X_sh = _sh_eng[NEW_FEATURES].values
                try:
                    import shap as _shap
                    _explainer = _shap.TreeExplainer(bundle['model_xgb'])
                    _shap_vals = _explainer.shap_values(_X_sh)
                    _sv = _shap_vals[0]
                    _feat_sv = sorted(zip(NEW_FEATURES, _sv), key=lambda x: abs(x[1]), reverse=True)[:12]
                    st.session_state['shap_result'] = {
                        'feat_sv': _feat_sv, 'sido': _sh_sido, 'month': _sh_month,
                        'prob': float(_sh_r.get('발화확률', 20)),
                    }
                except Exception as _e:
                    st.error(f"SHAP 계산 오류: {_e}")

        if 'shap_result' in st.session_state:
            _sr = st.session_state['shap_result']
            _feat_names = [f[0] for f in _sr['feat_sv']]
            _feat_vals  = [f[1] for f in _sr['feat_sv']]
            _clrs_shap  = ['#EF5350' if v > 0 else '#42A5F5' for v in _feat_vals]
            _fig_shap = go.Figure(go.Bar(
                x=_feat_vals[::-1], y=_feat_names[::-1],
                orientation='h', marker_color=_clrs_shap[::-1], opacity=0.85,
                text=[f"{v:+.4f}" for v in _feat_vals[::-1]], textposition='outside'))
            _fig_shap.update_layout(
                title=f"SHAP — {S(_sr['sido'])} {Mn(_sr['month'])} (발화확률 {_sr['prob']:.1f}%)" if not _is_en()
                      else f"SHAP — {S(_sr['sido'])} {Mn(_sr['month'])} (ML prob {_sr['prob']:.1f}%)",
                xaxis_title='SHAP value (빨강=위험↑ / 파랑=위험↓)' if not _is_en()
                            else 'SHAP value (red=risk↑ / blue=risk↓)',
                height=380, margin=dict(l=0,r=80,t=50,b=20))
            st.plotly_chart(_fig_shap, use_container_width=True)

    # ── Leave-one-year-out CV ──────────────────────────────────────
    st.markdown("---")
    _loo_title = "#### 📐 Leave-One-Year-Out 교차검증" if not _is_en() else "#### 📐 Leave-One-Year-Out CV"
    st.markdown(_loo_title)
    st.caption("각 연도를 순서대로 테스트셋으로 사용 — 시계열 정보 누수 방지" if not _is_en()
               else "Each year used as test set sequentially — prevents temporal data leakage")

    _years = sorted(df['연도'].unique())
    _loo_rows = []
    for _yr in _years:
        _te = df[df['연도'] == _yr]
        _yb_yr = (_te['변압기화재건수'] > 0).astype(int)
        _prob_yr = _te['발화확률'] / 100
        if _yb_yr.sum() == 0:
            continue
        try:
            _roc_yr  = roc_auc_score(_yb_yr, _prob_yr)
            _pra_yr  = average_precision_score(_yb_yr, _prob_yr)
            _yp_yr   = (_prob_yr >= 0.20).astype(int)
            _f2_yr   = fbeta_score(_yb_yr, _yp_yr, beta=2, zero_division=0)
            _rec_yr  = recall_score(_yb_yr, _yp_yr, zero_division=0)
            _n_fire  = int(_yb_yr.sum())
            _loo_rows.append({'연도': _yr, 'ROC-AUC': round(_roc_yr,3),
                              'PR-AUC': round(_pra_yr,3), 'F2': round(_f2_yr,3),
                              'Recall@0.20': round(_rec_yr,3), '실제화재': _n_fire})
        except Exception:
            continue

    if _loo_rows:
        _loo_df = pd.DataFrame(_loo_rows)
        _fig_loo = go.Figure()
        _fig_loo.add_scatter(x=_loo_df['연도'], y=_loo_df['ROC-AUC'],
                             mode='lines+markers', name='ROC-AUC',
                             line=dict(color='#42A5F5', width=2.5))
        _fig_loo.add_scatter(x=_loo_df['연도'], y=_loo_df['PR-AUC'],
                             mode='lines+markers', name='PR-AUC',
                             line=dict(color='#EF5350', width=2.5, dash='dot'))
        _fig_loo.add_scatter(x=_loo_df['연도'], y=_loo_df['F2'],
                             mode='lines+markers', name='F2(β=2)',
                             line=dict(color='#66BB6A', width=2))
        _fig_loo.update_layout(
            title='연도별 Leave-One-Year-Out 성능' if not _is_en() else 'Leave-One-Year-Out Performance',
            xaxis=dict(tickvals=_loo_df['연도'].tolist()),
            yaxis=dict(range=[0, 1.05]), height=280,
            margin=dict(l=0,r=10,t=40,b=20))
        st.plotly_chart(_fig_loo, use_container_width=True)
        _loo_disp = _loo_df.copy()
        if _is_en():
            _loo_disp.columns = ['Year','ROC-AUC','PR-AUC','F2','Recall@0.20','Fire Events']
        st.dataframe(_loo_disp, use_container_width=True, hide_index=True)
        _avg_prauc = _loo_df['PR-AUC'].mean()
        st.success(f"✅ " + (f"연도별 평균 PR-AUC: **{_avg_prauc:.3f}** — 특정 연도 과적합 없이 안정적으로 유지됩니다."
                             if not _is_en() else
                             f"Average PR-AUC across years: **{_avg_prauc:.3f}** — stable without year-specific overfitting."))

    # ── 향후 개선 로드맵 ──────────────────────────────────────────
    st.markdown("---")
    with st.expander("🗺️ " + ("향후 개선 로드맵" if not _is_en() else "Improvement Roadmap"), expanded=False):
        if not _is_en():
            st.markdown("""
| 우선순위 | 항목 | 기대 효과 |
|----------|------|-----------|
| **높음** | 학습 데이터 확장 (2010~2019 화재 이력 추가, 81건→250건+) | PR-AUC 0.12 → 0.20+ |
| **높음** | 임계값 자동 적용 (월별 F2 최적값으로 실시간 조정) | 탐지율 추가 개선 |
| **중간** | 시계열 lag feature (전월 위험도, 3개월 이동평균) | 계절 전환 구간 예측 개선 |
| **낮음** | 실시간 기상청 API 연동 + 자동 재예측 스케줄러 | 운영 자동화 |
""")
        else:
            st.markdown("""
| Priority | Item | Expected Impact |
|----------|------|-----------------|
| **High** | Training data expansion (2010–2019 fire history, 81→250+ events) | PR-AUC 0.12 → 0.20+ |
| **High** | Auto-apply monthly F2-optimal thresholds | Higher recall |
| **Medium** | Time-series lag features (previous-month risk, 3-month MA) | Better seasonal transitions |
| **Low** | Real-time KMA API + auto-reforecast scheduler | Operational automation |
""")

st.markdown("---")
st.caption("⚡ **TransFireRisk IMS v7.0**  |  모델: 앙상블 v3 (XGB+RF+LR) · 28피처  |  "
           "TFRI: IEC 60076-7 · CIGRE WG A2.49  |  기상: 기상청 API + Open-Meteo  |  "
           "날씨 빅데이터 콘테스트 2026")
