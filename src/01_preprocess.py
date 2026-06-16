"""
변압기 화재 위험도 예측 모델 - Step 1: 데이터 전처리 및 피처 엔지니어링
프로젝트: TransFireRisk (날씨 빅데이터 콘테스트 2026)
분석 단위: 시도 × 월 (1,020행 = 17시도 × 5년 × 12월)
"""
import pandas as pd
import numpy as np
import os

BASE = os.path.expanduser("~/Desktop/TransFireRisk/")
DATA = BASE + "data/"
MODEL = BASE + "model/"
os.makedirs(MODEL, exist_ok=True)

SIDO_MAP = {
    '서울특별시':'서울','부산광역시':'부산','대구광역시':'대구','인천광역시':'인천',
    '광주광역시':'광주','대전광역시':'대전','울산광역시':'울산','세종특별자치시':'세종',
    '경기도':'경기','강원특별자치도':'강원','충청북도':'충북','충청남도':'충남',
    '전북특별자치도':'전북','전라북도':'전북','전라남도':'전남',
    '경상북도':'경북','경상남도':'경남','제주특별자치도':'제주'
}
SIDO_LIST = ['서울','부산','대구','인천','광주','대전','울산','세종',
             '경기','강원','충북','충남','전북','전남','경북','경남','제주']

# ── 1. 기상 데이터 월별 집계 ───────────────────────────────────
print("1. 기상 데이터 월별 집계...")
df_w = pd.read_csv(DATA + "weather_sido_2020_2024.csv")
df_w['날짜'] = pd.to_datetime(df_w['날짜'])
df_w['연도'] = df_w['날짜'].dt.year
df_w['월']   = df_w['날짜'].dt.month

df_w = df_w.sort_values(['시도','날짜'])
df_w['전3일평균기온'] = df_w.groupby('시도')['평균기온'].transform(lambda x: x.rolling(3, min_periods=1).mean())
df_w['전7일강수합계'] = df_w.groupby('시도')['강수량(mm)'].transform(lambda x: x.rolling(7, min_periods=1).sum())
df_w['연속고온일수']  = df_w.groupby('시도')['최고기온'].transform(
    lambda x: x.ge(33).groupby((~x.ge(33)).cumsum()).cumsum()
)

weather_monthly = df_w.groupby(['시도','연도','월']).agg(
    월최고기온     = ('최고기온',    'max'),
    월평균기온     = ('평균기온',    'mean'),
    월최저기온     = ('최저기온',    'min'),
    월강수합계     = ('강수량(mm)',  'sum'),
    월평균습도     = ('평균습도(%)', 'mean'),
    월최대풍속     = ('최대풍속(km/h)','max'),
    월평균일교차   = ('기온일교차',  'mean'),
    월전3일평균기온 = ('전3일평균기온','mean'),
    월전7일강수합계 = ('전7일강수합계','mean'),
    월연속고온일수  = ('연속고온일수','max'),
    강수일수       = ('강수량(mm)', lambda x: (x > 1).sum()),
).reset_index()
print(f"   → {len(weather_monthly):,}행")

# ── 2. 화재 데이터 월별 집계 ──────────────────────────────────
print("2. 변압기 화재 월별 집계...")
df_fire = pd.read_csv(DATA + "fire_data_20241231.csv", encoding="euc-kr", on_bad_lines='skip')
df_fire['화재발생년원일'] = pd.to_datetime(df_fire['화재발생년원일'], errors='coerce')
df_fire['연도'] = df_fire['화재발생년원일'].dt.year
df_fire['월']   = df_fire['화재발생년원일'].dt.month
df_fire['시도_매핑'] = df_fire['시도'].map(SIDO_MAP)

tr_mask = df_fire['장소소분류'].isin(['변압기','변전소','송, 배전설비','전력구, 통신구'])
df_tr   = df_fire[tr_mask].copy()

fire_monthly = df_tr.groupby(['시도_매핑','연도','월']).agg(
    변압기화재건수 = ('장소소분류','count'),
    인명피해합계   = ('인명피해(명)소계','sum'),
    재산피해합계   = ('재산피해소계','sum'),
).reset_index().rename(columns={'시도_매핑':'시도'})
print(f"   → 화재 발생 월 {len(fire_monthly)}개")

# ── 3. 지역 이력 변수 ─────────────────────────────────────────
print("3. 지역 이력 변수 생성...")
df_elec  = df_fire[df_fire['발화요인대분류'] == '전기적 요인'].copy()
elec_hist = df_elec.groupby(['시도_매핑','연도']).size().reset_index(name='전기화재건수')
elec_hist.columns = ['시도','연도','전기화재건수']
elec_lag  = elec_hist.copy()
elec_lag['연도'] = elec_lag['연도'] + 1
elec_lag  = elec_lag.rename(columns={'전기화재건수':'전년전기화재건수'})

tr_hist  = df_tr.groupby(['시도_매핑','연도']).size().reset_index(name='변압기화재건수_연')
tr_hist.columns = ['시도','연도','변압기화재건수_연']
tr_lag   = tr_hist.copy()
tr_lag['연도'] = tr_lag['연도'] + 1
tr_lag   = tr_lag.rename(columns={'변압기화재건수_연':'전년변압기화재건수'})

# ── 4. 최종 격자 구성 ─────────────────────────────────────────
print("4. 최종 데이터 결합...")
grid = pd.MultiIndex.from_product(
    [SIDO_LIST, [2020,2021,2022,2023,2024], list(range(1,13))],
    names=['시도','연도','월']
)
df_model = pd.DataFrame(index=grid).reset_index()

df_model = (df_model
    .merge(weather_monthly, on=['시도','연도','월'], how='left')
    .merge(fire_monthly,    on=['시도','연도','월'], how='left')
    .merge(elec_lag,        on=['시도','연도'],      how='left')
    .merge(tr_lag,          on=['시도','연도'],      how='left')
)

df_model['변압기화재건수']   = df_model['변압기화재건수'].fillna(0).astype(int)
df_model['전년전기화재건수'] = df_model['전년전기화재건수'].fillna(0)
df_model['전년변압기화재건수']= df_model['전년변압기화재건수'].fillna(0)
df_model['인명피해합계']     = df_model['인명피해합계'].fillna(0)
df_model['재산피해합계']     = df_model['재산피해합계'].fillna(0)

df_model['계절'] = df_model['월'].map({
    12:'겨울',1:'겨울',2:'겨울',
    3:'봄',4:'봄',5:'봄',
    6:'여름',7:'여름',8:'여름',
    9:'가을',10:'가을',11:'가을'
})
df_model['계절코드'] = df_model['계절'].map({'봄':1,'여름':2,'가을':3,'겨울':4})
df_model['시도코드'] = pd.Categorical(df_model['시도'], categories=SIDO_LIST).codes

print(f"   → 최종 {len(df_model):,}행 × {df_model.shape[1]}열")
print(f"   → Y=0: {(df_model['변압기화재건수']==0).sum()} | Y≥1: {(df_model['변압기화재건수']>0).sum()}")
print(f"   → 결측 없음: {df_model.isnull().sum().sum() == 0}")

df_model.to_csv(MODEL + "model_input.csv", index=False, encoding='utf-8-sig')
print(f"\n✅ 저장 완료: TransFireRisk/model/model_input.csv")
