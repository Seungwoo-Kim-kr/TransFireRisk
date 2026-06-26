"""
보고서용 Figure 생성
  - fig_pipeline.png   : 연구 파이프라인 (research paper 스타일)
  - fig_gui_tabs.png   : 대시보드 7탭 구성 개요
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os

matplotlib.rcParams['font.family'] = ['AppleGothic', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUT = os.path.expanduser("~/Desktop/TransFireRisk/output/")
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# Figure 1: 연구 파이프라인 (Research Paper Style)
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(15, 6.5))
ax.set_xlim(0, 15); ax.set_ylim(0, 6.5)
ax.axis('off')
fig.patch.set_facecolor('white')

# 색상
C_DATA  = '#1565C0'   # 진파랑 — 데이터
C_PREP  = '#0288D1'   # 하늘 — 전처리
C_MOD   = '#2E7D32'   # 진녹 — 모델
C_RISK  = '#E65100'   # 오렌지 — 위험도
C_DASH  = '#6A1B9A'   # 보라 — 대시보드
C_ARROW = '#37474F'
C_SUBBOX= '#ECEFF1'
C_SUBTXT= '#455A64'

def rounded_box(ax, x, y, w, h, color, label, sublabel=None, fontsize=12):
    box = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.08", linewidth=1.5,
        edgecolor=color, facecolor=color + '18')
    ax.add_patch(box)
    # 상단 색 띠
    bar = FancyBboxPatch((x, y + h - 0.38), w, 0.38,
        boxstyle="round,pad=0.01", linewidth=0,
        edgecolor='none', facecolor=color)
    ax.add_patch(bar)
    ax.text(x + w/2, y + h - 0.19, label,
            ha='center', va='center', fontsize=fontsize,
            color='white', fontweight='bold')
    if sublabel:
        for i, sub in enumerate(sublabel):
            ax.text(x + w/2, y + h - 0.75 - i*0.38, sub,
                    ha='center', va='center', fontsize=8.5,
                    color=C_SUBTXT)

def arrow(ax, x1, y, x2, color=C_ARROW):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle='->', color=color,
                        lw=2.0, mutation_scale=18))

# ── 박스 정의 ─────────────────────────────────────────
boxes = [
    (0.3, 1.5, 2.4, 3.2, C_DATA, "① 데이터 수집",
     ["기상청 월별 관측값", "소방청 화재 이력", "전력화재 통계 (HRI)"]),
    (3.3, 1.5, 2.6, 3.2, C_PREP, "② 전처리 & FE",
     ["결측 처리·월별 집계", "원본 16 + 파생 12", "= 28개 피처 구성"]),
    (6.5, 1.5, 2.6, 3.2, C_MOD,  "③ 앙상블 학습",
     ["XGB × 0.6", "RF × 0.3", "LR × 0.1"]),
    (9.7, 1.5, 2.6, 3.2, C_RISK, "④ 위험도 산출",
     ["ML 발화확률 P", "TFRI 물리 지수", "4등급 분류"]),
    (12.7, 1.5, 2.0, 3.2, C_DASH, "⑤ 대시보드",
     ["7탭 GUI", "한/영 이중언어", "AI 브리핑"]),
]

for (x, y, w, h, c, lbl, subs) in boxes:
    rounded_box(ax, x, y, w, h, c, lbl, subs, fontsize=10.5)

# ── 화살표 ────────────────────────────────────────────
arrow_pairs = [(2.7, 4.6, 3.3, C_DATA),
               (5.9, 4.6, 6.5, C_PREP),
               (9.1, 4.6, 9.7, C_MOD),
               (12.3, 4.6, 12.7, C_RISK)]
for x1, y, x2, c in arrow_pairs:
    arrow(ax, x1, y, x2, c)

# ── 하단 학습/검증 분리 표시 ─────────────────────────
ax.add_patch(FancyBboxPatch((6.5, 0.2), 2.6, 0.9,
    boxstyle="round,pad=0.05", linewidth=1.2,
    edgecolor=C_MOD+'AA', facecolor='#E8F5E9'))
ax.text(7.8, 0.65, '학습: 2020~2022 | 검증: 2023~2024',
        ha='center', va='center', fontsize=8.5, color=C_MOD)

# 점선 연결
ax.annotate('', xy=(7.8, 1.5), xytext=(7.8, 1.1),
    arrowprops=dict(arrowstyle='->', color=C_MOD+'88',
                    lw=1.2, linestyle='dashed', mutation_scale=12))

# ── 하단 TFRI 박스 ─────────────────────────────────
ax.add_patch(FancyBboxPatch((9.7, 0.2), 2.6, 0.9,
    boxstyle="round,pad=0.05", linewidth=1.2,
    edgecolor=C_RISK+'AA', facecolor='#FFF3E0'))
ax.text(11.0, 0.65, 'TFRI = WHI(45%) + IDA(35%) + HRI(20%)',
        ha='center', va='center', fontsize=8, color=C_RISK)
ax.annotate('', xy=(11.0, 1.5), xytext=(11.0, 1.1),
    arrowprops=dict(arrowstyle='->', color=C_RISK+'88',
                    lw=1.2, linestyle='dashed', mutation_scale=12))

# ── 제목 ─────────────────────────────────────────────
ax.text(7.7, 6.1, 'TransFireRisk IMS — 분석 파이프라인',
        ha='center', va='center', fontsize=14, fontweight='bold',
        color='#1A237E')

plt.tight_layout(pad=0.3)
plt.savefig(OUT + 'fig_pipeline.png', dpi=180, bbox_inches='tight',
            facecolor='white')
plt.close()
print("✅ fig_pipeline.png")

# ══════════════════════════════════════════════════════════════════
# Figure 2: 대시보드 7탭 구성 개요 (GUI Overview)
# ══════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(15, 8))
ax2.set_xlim(0, 15); ax2.set_ylim(0, 8)
ax2.axis('off')
fig2.patch.set_facecolor('#F8F9FA')

# 탭별 정보
tabs = [
    ("Tab 1", "[Map] 종합 현황판",
     ["전국 Choropleth 지도", "KPI 메트릭 카드", "AI 운영 브리핑", "P1·P2 경보 배너"],
     '#0D47A1', '#E3F2FD'),
    ("Tab 2", "[Region] 지역 상세",
     ["17개 시도 선택", "12개월 위험도 추이", "WHI·IDA·HRI 레이더", "AI 점검 가이드"],
     '#1565C0', '#E3F2FD'),
    ("Tab 3", "[Forecast] 기상 예보",
     ["Open-Meteo 14일 예보", "ML+TFRI 일별 예측", "기상 차트", "예보 AI 분석"],
     '#1976D2', '#E3F2FD'),
    ("Tab 4", "[Sim] 시나리오",
     ["기상 슬라이더 8개", "2024 실측 불러오기", "기온×습도 히트맵", "AI 시뮬레이션"],
     '#0288D1', '#E1F5FE'),
    ("Tab 5", "[TFRI] 복합위험",
     ["TFRI 성분 분해", "WHI·IDA·HRI 막대", "전국 스택 차트", "물리 수식 전시"],
     '#00838F', '#E0F7FA'),
    ("Tab 6", "[Mgmt] 점검 관리",
     ["P1/P2/P3 리스트", "월간 캘린더 뷰", "완료율 프로그레스", "CSV 내보내기"],
     '#2E7D32', '#E8F5E9'),
    ("Tab 7", "[Model] 이력·모델",
     ["연도별 화재 현황", "F2 임계값 최적화", "SHAP 개별 설명", "LOO-CV 검증"],
     '#6A1B9A', '#F3E5F5'),
]

cols = 4
rows_g = 2
box_w, box_h = 3.4, 3.5
gap_x, gap_y = 0.18, 0.25
start_x, start_y = 0.2, 0.4

for idx, (tab_id, name, features, hdr_c, bg_c) in enumerate(tabs):
    col = idx % cols
    row = idx // cols
    x = start_x + col * (box_w + gap_x)
    y = start_y + (rows_g - 1 - row) * (box_h + gap_y)

    # 박스 배경
    ax2.add_patch(FancyBboxPatch((x, y), box_w, box_h,
        boxstyle="round,pad=0.1", linewidth=1.5,
        edgecolor=hdr_c + '88', facecolor=bg_c + 'BB'))
    # 헤더
    ax2.add_patch(FancyBboxPatch((x, y + box_h - 0.7), box_w, 0.7,
        boxstyle="round,pad=0.05", linewidth=0,
        edgecolor='none', facecolor=hdr_c))
    ax2.text(x + box_w/2, y + box_h - 0.35, f"{tab_id}  {name}",
             ha='center', va='center', fontsize=9.5,
             color='white', fontweight='bold')
    # 기능 목록
    for fi, feat in enumerate(features):
        ax2.text(x + 0.22, y + box_h - 1.05 - fi * 0.58, f"• {feat}",
                 ha='left', va='center', fontsize=8.5, color='#37474F')

# 마지막 빈 칸에 범례 요약
lx = start_x + 3 * (box_w + gap_x)
ly = start_y
ax2.add_patch(FancyBboxPatch((lx, ly), box_w, box_h,
    boxstyle="round,pad=0.1", linewidth=1.5,
    edgecolor='#455A64', facecolor='#ECEFF1'))
ax2.text(lx + box_w/2, ly + box_h - 0.35, "⚡ 시스템 특징",
         ha='center', va='center', fontsize=10, fontweight='bold', color='#1A237E')
features_sys = [
    "한국어/English 전환",
    "다크/라이트 모드",
    "모바일 반응형 CSS",
    "PDF 보고서 즉시 생성",
    "GPT-4o-mini AI 브리핑",
    "Open-Meteo 14일 예보",
]
for fi, feat in enumerate(features_sys):
    ax2.text(lx + 0.22, ly + box_h - 1.05 - fi * 0.48, f"• {feat}",
             ha='left', va='center', fontsize=8.5, color='#455A64')

# 제목
ax2.text(7.5, 7.7, 'TransFireRisk IMS v8.0 — 대시보드 구성 개요 (7탭)',
         ha='center', va='center', fontsize=14, fontweight='bold', color='#1A237E')

plt.tight_layout(pad=0.2)
plt.savefig(OUT + 'fig_gui_tabs.png', dpi=180, bbox_inches='tight',
            facecolor='#F8F9FA')
plt.close()
print("✅ fig_gui_tabs.png")

# ══════════════════════════════════════════════════════════════════
# Figure 3: ROC Curve (간단)
# ══════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, joblib
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

df     = pd.read_csv(os.path.expanduser('~/Desktop/TransFireRisk/model/full_predictions_final.csv'))
te     = df[df['연도'] >= 2023]
yb     = (te['변압기화재건수'] > 0).astype(int)
prob   = te['발화확률'] / 100

fpr, tpr, _ = roc_curve(yb, prob)
roc_auc     = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(yb, prob)
ap          = average_precision_score(yb, prob)

fig3, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(11, 4.5))
fig3.patch.set_facecolor('white')

# ROC
ax_r.plot(fpr, tpr, color='#1565C0', lw=2.2, label=f'앙상블 v3 (AUC = {roc_auc:.3f})')
ax_r.plot([0,1],[0,1],'--', color='#9E9E9E', lw=1.2, label='랜덤 기준선 (AUC = 0.500)')
ax_r.set_xlim([0,1]); ax_r.set_ylim([0,1.02])
ax_r.set_xlabel('False Positive Rate', fontsize=11); ax_r.set_ylabel('True Positive Rate', fontsize=11)
ax_r.set_title('ROC Curve (검증셋 2023~2024)', fontsize=12, fontweight='bold')
ax_r.legend(fontsize=9.5, loc='lower right')
ax_r.fill_between(fpr, tpr, alpha=0.08, color='#1565C0')
ax_r.spines['top'].set_visible(False); ax_r.spines['right'].set_visible(False)
ax_r.grid(True, alpha=0.3)

# PR Curve
ax_p.plot(rec, prec, color='#E65100', lw=2.2, label=f'앙상블 v3 (AP = {ap:.3f})')
ax_p.axhline(y=yb.mean(), color='#9E9E9E', lw=1.2, linestyle='--',
             label=f'랜덤 기준선 (AP = {yb.mean():.3f})')
ax_p.set_xlim([0,1]); ax_p.set_ylim([0,1.02])
ax_p.set_xlabel('Recall', fontsize=11); ax_p.set_ylabel('Precision', fontsize=11)
ax_p.set_title('Precision-Recall Curve (검증셋 2023~2024)', fontsize=12, fontweight='bold')
ax_p.legend(fontsize=9.5, loc='upper right')
ax_p.fill_between(rec, prec, alpha=0.08, color='#E65100')
ax_p.spines['top'].set_visible(False); ax_p.spines['right'].set_visible(False)
ax_p.grid(True, alpha=0.3)

plt.tight_layout(pad=1.2)
plt.savefig(OUT + 'fig_roc_pr.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ fig_roc_pr.png")

print("\n모든 figure 생성 완료!")
