"""
변압기 화재 위험도 예측 모델 - Step 3: 시각화
프로젝트: TransFireRisk
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import platform, os, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'AppleGothic' if platform.system()=='Darwin' else 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

BASE  = os.path.expanduser("~/Desktop/TransFireRisk/")
MODEL = BASE + "model/"
OUT   = BASE + "output/"
os.makedirs(OUT, exist_ok=True)

df   = pd.read_csv(MODEL + "full_predictions.csv")
fi   = pd.read_csv(MODEL + "feature_importance.csv", index_col=0, header=0)
fi.columns = ['중요도']
fi   = fi.sort_values('중요도', ascending=False)

RISK_COLORS = {'낮음':'#4CAF50','보통':'#FFC107','높음':'#FF9800','매우높음':'#F44336'}
SIDO_ORDER  = ['경기','서울','인천','부산','경남','전남','대구','경북','충남',
               '충북','강원','전북','광주','대전','울산','제주','세종']

# ── Fig 1: 피처 중요도 ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9,6))
colors  = ['#1565C0' if i<3 else '#42A5F5' if i<6 else '#90CAF9' for i in range(len(fi))]
fi['중요도'].plot(kind='barh', ax=ax, color=colors[::-1], edgecolor='white')
ax.set_title('변수 중요도 — 변압기 화재 위험 예측 모델', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('중요도 (XGBoost Feature Importance)')
ax.invert_yaxis()
ax.axvline(fi['중요도'].mean(), color='red', ls='--', alpha=0.6, label='평균')
ax.legend(); ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + 'fig1_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close(); print("✅ fig1_feature_importance.png")

# ── Fig 2: 연도별 실제 vs 예측 ────────────────────────────────
yearly = df.groupby('연도').agg(실제=('변압기화재건수','sum'), 예측=('예측건수','sum')).reset_index()
fig, ax = plt.subplots(figsize=(9,5))
x = yearly['연도']
ax.bar(x-0.2, yearly['실제'], 0.4, label='실제', color='#EF5350', alpha=0.85)
ax.bar(x+0.2, yearly['예측'], 0.4, label='예측', color='#42A5F5', alpha=0.85)
for i,(r,p) in enumerate(zip(yearly['실제'], yearly['예측'])):
    ax.text(x.iloc[i]-0.2, r+0.3, str(int(r)), ha='center', fontsize=10, fontweight='bold', color='#B71C1C')
    ax.text(x.iloc[i]+0.2, p+0.3, f'{p:.1f}', ha='center', fontsize=10, color='#1565C0')
ax.set_title('연도별 변압기 화재: 실제 vs 예측', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('연도'); ax.set_ylabel('화재 건수')
ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3); ax.set_xticks(yearly['연도'])
plt.tight_layout()
plt.savefig(OUT + 'fig2_yearly_comparison.png', dpi=150, bbox_inches='tight')
plt.close(); print("✅ fig2_yearly_comparison.png")

# ── Fig 3: 시도×월 위험도 히트맵 ─────────────────────────────
pivot = df.groupby(['시도','월'])['예측건수'].mean().unstack().reindex(SIDO_ORDER)
fig, ax = plt.subplots(figsize=(13,7))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd',
            linewidths=0.5, ax=ax, cbar_kws={'label':'예측 화재건수 (월평균)'})
ax.set_title('시도×월별 변압기 화재 위험도 히트맵 (2020~2024 평균)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('월'); ax.set_ylabel('')
ax.set_xticklabels([f'{i}월' for i in range(1,13)], rotation=0)
plt.tight_layout()
plt.savefig(OUT + 'fig3_heatmap_sido_month.png', dpi=150, bbox_inches='tight')
plt.close(); print("✅ fig3_heatmap_sido_month.png")

# ── Fig 4: 날씨 vs 화재 산점도 (2×2) ─────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12,9))
for ax, (var, xlabel, cause) in zip(axes.flat, [
    ('월최고기온',  '월 최고기온 (℃)',    '→ 과부하/과전류'),
    ('월평균습도',  '월 평균습도 (%)',     '→ 트래킹·절연열화'),
    ('월강수합계',  '월 강수합계 (mm)',    '→ 누전·지락'),
    ('월평균일교차','월 평균 기온일교차 (℃)','→ 절연열화'),
]):
    no_fire = df[df['변압기화재건수']==0]
    fire    = df[df['변압기화재건수']>0]
    ax.scatter(no_fire[var], no_fire['변압기화재건수'] + np.random.uniform(-0.05,0.05,len(no_fire)),
               alpha=0.15, s=15, color='#90CAF9', label='화재 없음')
    ax.scatter(fire[var], fire['변압기화재건수'] + np.random.uniform(-0.05,0.05,len(fire)),
               alpha=0.7, s=40, color='#EF5350', zorder=5, label='화재 발생')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('변압기 화재 건수', fontsize=10)
    ax.set_title(f'{xlabel}\n{cause}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.suptitle('날씨 변수별 변압기 화재 발생 분포', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT + 'fig4_weather_scatter.png', dpi=150, bbox_inches='tight')
plt.close(); print("✅ fig4_weather_scatter.png")

# ── Fig 5: 2024년 시도별 위험도 ──────────────────────────────
df24 = df[df['연도']==2024].groupby('시도').agg(
    실제=('변압기화재건수','sum'), 예측=('예측건수','sum')
).reset_index().sort_values('예측', ascending=True)
fig, ax = plt.subplots(figsize=(10,7))
from matplotlib.patches import Patch
bar_colors = ['#F44336' if v>=1.2 else '#FF9800' if v>=0.7 else '#FFC107' if v>=0.3 else '#4CAF50'
              for v in df24['예측']]
ax.barh(df24['시도'], df24['예측'], color=bar_colors, edgecolor='white', alpha=0.9)
ax.scatter(df24['실제'], df24['시도'], color='black', zorder=5, s=60, marker='D', label='실제 발생건수')
ax.set_title('2024년 시도별 변압기 화재 위험도 예측', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('예측 화재건수 합계')
ax.legend(handles=[
    Patch(fc='#F44336', label='매우높음 (≥1.2)'), Patch(fc='#FF9800', label='높음 (0.7~1.2)'),
    Patch(fc='#FFC107', label='보통 (0.3~0.7)'),  Patch(fc='#4CAF50', label='낮음 (<0.3)'),
    plt.scatter([],[],color='black',marker='D',label='실제 발생건수')
], loc='lower right', fontsize=9)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + 'fig5_sido_risk_2024.png', dpi=150, bbox_inches='tight')
plt.close(); print("✅ fig5_sido_risk_2024.png")

print(f"\n✅ 모든 시각화 저장 완료: TransFireRisk/output/")
