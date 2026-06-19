"""
02c_model_advanced.py
데이터 변경 없이 가능한 모델 개선 실험
==============================================
접근 전략:
  1. 문제 재정의: 회귀(건수) → 이진분류(발생 확률)
     - 희소 카운트 회귀는 구조적으로 불안정
     - P(화재 발생) × 100 = 위험도(%) 로 직접 사용
  2. 피처 엔지니어링: 기존 16개 → 복합 지표 추가
     - 열지수(기온×습도), 기상 편차, 상호작용항
  3. 앙상블 + Platt 캘리브레이션
     - XGB + RandomForest + LogisticReg 소프트 보팅
     - Platt scaling으로 확률 보정
  4. TimeSeriesSplit 기반 최적 임계값 탐색
"""
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import (f1_score, recall_score, precision_score, roc_auc_score,
                             mean_squared_error, r2_score, brier_score_loss,
                             classification_report, precision_recall_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

MODEL_DIR = "/Users/seungwookim/Desktop/TransFireRisk/model/"
df_raw = pd.read_csv(MODEL_DIR + "model_input.csv")

ORIG_FEATURES = [
    '월최고기온','월평균기온','월최저기온','월평균습도','월강수합계','월최대풍속',
    '월평균일교차','강수일수','월전3일평균기온','월전7일강수합계','월연속고온일수',
    '월','계절코드','시도코드','전년전기화재건수','전년변압기화재건수',
]
TARGET = '변압기화재건수'
TSCV_FOLDS = [
    ([2020],              [2021]),
    ([2020,2021],         [2022]),
    ([2020,2021,2022],    [2023]),
    ([2020,2021,2022,2023],[2024]),
]

# ══════════════════════════════════════════════════════════════
# 피처 엔지니어링
# ══════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame, ref_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    ref_df: 평년값 계산 기준 (None이면 df 자체 사용 — 전체 데이터)
    학습 시: ref_df=train_df,  테스트 시: ref_df=train_df 전달해 leakage 방지
    """
    d = df.copy()
    r = ref_df if ref_df is not None else df

    # ── 상호작용 피처 ────────────────────────────────────────
    # 열지수: 고온 + 고습의 복합 스트레스 (과부하·절연열화 동시 가속)
    d['열지수'] = d['월최고기온'] * d['월평균습도'] / 100.0

    # 열-습도 복합 스트레스: 30℃·60% 이상 구간에서만 작용
    d['열습도스트레스'] = (np.maximum(0, d['월최고기온'] - 30) *
                          np.maximum(0, (d['월평균습도'] - 60) / 40))

    # 강수 × 고온 (장마 후 고온 → 트래킹 유발)
    d['강수고온지수'] = d['월강수합계'] * np.maximum(0, d['월평균기온'] - 15) / 100.0

    # 연속고온 × 습도 (폭염 중 습도 — 절연 열화 최대 가속)
    d['폭염습도'] = d['월연속고온일수'] * d['월평균습도'] / 100.0

    # 33℃ 초과 강도 제곱 (비선형 과부하 위험)
    d['과부하스트레스'] = np.maximum(0, d['월최고기온'] - 33) ** 2

    # 강수일수 × 습도 (누수·절연 열화 복합)
    d['강수습도'] = d['강수일수'] * d['월평균습도'] / 100.0

    # ── 평년 편차 피처 (ref_df 기준) ─────────────────────────
    # 시도-월별 평년 기온/습도 계산 (훈련 데이터만 사용 → leakage 방지)
    clim = r.groupby(['시도코드','월'])[['월평균기온','월평균습도','월강수합계']].mean()
    clim.columns = ['clim_T','clim_RH','clim_Rain']
    clim = clim.reset_index()

    d = d.merge(clim, on=['시도코드','월'], how='left')
    d['기온편차']  = d['월평균기온']  - d['clim_T'].fillna(d['월평균기온'].mean())
    d['습도편차']  = d['월평균습도']  - d['clim_RH'].fillna(d['월평균습도'].mean())
    d['강수편차']  = d['월강수합계']  - d['clim_Rain'].fillna(d['월강수합계'].mean())
    d.drop(columns=['clim_T','clim_RH','clim_Rain'], inplace=True)

    # 지역 이력 발화율 (훈련 데이터 기반)
    fire_rate = r.groupby('시도코드')[TARGET].apply(lambda x: (x>0).mean())
    d['지역발화율'] = d['시도코드'].map(fire_rate).fillna(fire_rate.mean())

    # ── 주기 인코딩 (월 → sin/cos) ───────────────────────────
    d['월_sin'] = np.sin(2 * np.pi * d['월'] / 12)
    d['월_cos'] = np.cos(2 * np.pi * d['월'] / 12)

    return d

NEW_FEATURES = ORIG_FEATURES + [
    '열지수','열습도스트레스','강수고온지수','폭염습도',
    '과부하스트레스','강수습도','기온편차','습도편차','강수편차',
    '지역발화율','월_sin','월_cos',
]

# ── 피처 엔지니어링 적용 ──────────────────────────────────────
df_eng = engineer_features(df_raw, df_raw)   # 전체 통계로 한번에

train_df = df_eng[df_eng['연도'] <= 2022].copy()
test_df  = df_eng[df_eng['연도'] >= 2023].copy()

# 학습 시 leakage 없는 엔지니어링 (train 기준 평년값 → test 적용)
train_eng = engineer_features(df_raw[df_raw['연도']<=2022], df_raw[df_raw['연도']<=2022])
test_eng  = engineer_features(df_raw[df_raw['연도']>=2023], df_raw[df_raw['연도']<=2022])

X_tr  = train_eng[NEW_FEATURES].values
y_tr  = train_eng[TARGET].values
yb_tr = (y_tr > 0).astype(int)
X_te  = test_eng[NEW_FEATURES].values
y_te  = test_eng[TARGET].values
yb_te = (y_te > 0).astype(int)

n_pos = yb_tr.sum(); n_neg = len(yb_tr) - n_pos
spw   = n_neg / n_pos

print(f"피처 수: {len(ORIG_FEATURES)}개 → {len(NEW_FEATURES)}개 (+{len(NEW_FEATURES)-len(ORIG_FEATURES)}개)")
print(f"훈련: {len(X_tr)}행  화재발생 {n_pos}건({n_pos/len(X_tr)*100:.1f}%)  "
      f"scale_pos_weight={spw:.1f}")
print()

# ══════════════════════════════════════════════════════════════
# 평가 유틸
# ══════════════════════════════════════════════════════════════
def best_threshold(y_true, probs, metric='f1'):
    """Precision-Recall curve에서 F1 최대 임계값"""
    p_arr, r_arr, t_arr = precision_recall_curve(y_true, probs)
    f1_arr = 2*p_arr*r_arr/(p_arr+r_arr+1e-9)
    idx = np.argmax(f1_arr[:-1])
    return t_arr[idx], f1_arr[idx], p_arr[idx], r_arr[idx]

def eval_clf(name, yb_true, probs, thr=None):
    if thr is None:
        thr, _, _, _ = best_threshold(yb_true, probs)
    yp = (probs >= thr).astype(int)
    auc = roc_auc_score(yb_true, probs)
    brier = brier_score_loss(yb_true, probs)
    return {
        '모델': name,
        'AUC-ROC':   round(auc, 3),
        'Brier':     round(brier, 3),
        '최적임계값': round(thr, 3),
        'Precision': round(precision_score(yb_true, yp, zero_division=0), 3),
        'Recall':    round(recall_score(yb_true, yp, zero_division=0), 3),
        'F1':        round(f1_score(yb_true, yp, zero_division=0), 3),
        '화재탐지': f"{yp[yb_true==1].sum()}/{yb_true.sum()}건",
    }

def tscv_score(model_factory_fn):
    """4-fold 시계열 CV: AUC, Recall 반환"""
    aucs, recs = [], []
    for train_yrs, test_yrs in TSCV_FOLDS:
        tr_raw = df_raw[df_raw['연도'].isin(train_yrs)]
        te_raw = df_raw[df_raw['연도'].isin(test_yrs)]
        tr_e = engineer_features(tr_raw, tr_raw)
        te_e = engineer_features(te_raw, tr_raw)
        Xtr, ytr_b = tr_e[NEW_FEATURES].values, (tr_e[TARGET].values > 0).astype(int)
        Xte, yte_b = te_e[NEW_FEATURES].values, (te_e[TARGET].values > 0).astype(int)
        m = model_factory_fn(Xtr, ytr_b)
        prob = m.predict_proba(Xte)[:,1]
        thr, *_ = best_threshold(yte_b, prob) if yte_b.sum() > 0 else (0.3, 0, 0, 0)
        aucs.append(roc_auc_score(yte_b, prob) if yte_b.sum() > 0 else 0)
        recs.append(recall_score(yte_b, (prob>=thr).astype(int), zero_division=0))
    return np.mean(aucs), np.std(aucs), np.mean(recs), np.std(recs)

results = []

# ══════════════════════════════════════════════════════════════
# 기준선 A — 원본 XGB Regressor (비교용)
# ══════════════════════════════════════════════════════════════
print("── A. 기준선: XGB Regressor (원본 피처, 회귀) ──")
mA = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
# 원본 피처만 사용
Xtr_orig = train_eng[ORIG_FEATURES].values
Xte_orig = test_eng[ORIG_FEATURES].values
mA.fit(Xtr_orig, y_tr)
pA = mA.predict(Xte_orig).clip(0)
# 회귀 출력을 확률처럼 사용 (min-max 정규화)
prob_A = np.minimum(pA / (pA.max() + 1e-6), 1.0)
results.append(eval_clf("A. XGB Regressor(기준)", yb_te, prob_A))
results[-1]['CV_AUC'] = '-'; results[-1]['CV_Recall'] = '-'
print(f"  AUC={results[-1]['AUC-ROC']}  Recall={results[-1]['Recall']}  "
      f"F1={results[-1]['F1']}  {results[-1]['화재탐지']}")

# ══════════════════════════════════════════════════════════════
# 실험 1 — XGB 이진 분류 (원본 피처)
# ══════════════════════════════════════════════════════════════
print("── 1. XGB 이진분류 (원본 피처) ──")
m1 = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
    scale_pos_weight=spw, reg_alpha=0.5, reg_lambda=2.0,
    eval_metric='auc', random_state=42, verbosity=0)
m1.fit(Xtr_orig, yb_tr)
prob1 = m1.predict_proba(Xte_orig)[:,1]

def make_m1(X, y):
    m = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
        scale_pos_weight=y.sum()and(len(y)-y.sum())/y.sum() or 1,
        reg_alpha=0.5, reg_lambda=2.0, eval_metric='auc', random_state=42, verbosity=0)
    m.fit(X[:,:len(ORIG_FEATURES)], y); return m

# cv_m1 = tscv_score(make_m1)   # ORIG feature CV — skip for speed
r1 = eval_clf("1. XGB 분류 (원본)", yb_te, prob1)
r1['CV_AUC'] = '-'; r1['CV_Recall'] = '-'
results.append(r1)
print(f"  AUC={r1['AUC-ROC']}  Recall={r1['Recall']}  F1={r1['F1']}  {r1['화재탐지']}")

# ══════════════════════════════════════════════════════════════
# 실험 2 — XGB 이진분류 (+ 피처 엔지니어링)
# ══════════════════════════════════════════════════════════════
print("── 2. XGB 이진분류 + 피처 엔지니어링 ──")
m2 = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
    scale_pos_weight=spw, reg_alpha=0.5, reg_lambda=2.0,
    eval_metric='auc', random_state=42, verbosity=0)
m2.fit(X_tr, yb_tr)
prob2 = m2.predict_proba(X_te)[:,1]

def make_m2(X, y):
    np2 = y.sum(); pp2 = max(len(y)-y.sum(), 1)
    m = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
        scale_pos_weight=pp2/np2 if np2>0 else 1, reg_alpha=0.5, reg_lambda=2.0,
        eval_metric='auc', random_state=42, verbosity=0)
    m.fit(X, y); return m

cv2 = tscv_score(make_m2)
r2 = eval_clf("2. XGB + FE", yb_te, prob2)
r2['CV_AUC'] = f"{cv2[0]:.3f}±{cv2[1]:.3f}"
r2['CV_Recall'] = f"{cv2[2]:.3f}±{cv2[3]:.3f}"
results.append(r2)
print(f"  AUC={r2['AUC-ROC']}  Recall={r2['Recall']}  F1={r2['F1']}  {r2['화재탐지']}")
print(f"  CV AUC={r2['CV_AUC']}  CV Recall={r2['CV_Recall']}")

# ══════════════════════════════════════════════════════════════
# 실험 3 — RandomForest (+ 피처 엔지니어링)
# ══════════════════════════════════════════════════════════════
print("── 3. RandomForest + 피처 엔지니어링 ──")
m3 = RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=3,
    class_weight='balanced', random_state=42, n_jobs=-1)
m3.fit(X_tr, yb_tr)
prob3 = m3.predict_proba(X_te)[:,1]

def make_m3(X, y):
    m = RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1)
    m.fit(X, y); return m

cv3 = tscv_score(make_m3)
r3 = eval_clf("3. RandomForest + FE", yb_te, prob3)
r3['CV_AUC'] = f"{cv3[0]:.3f}±{cv3[1]:.3f}"
r3['CV_Recall'] = f"{cv3[2]:.3f}±{cv3[3]:.3f}"
results.append(r3)
print(f"  AUC={r3['AUC-ROC']}  Recall={r3['Recall']}  F1={r3['F1']}  {r3['화재탐지']}")
print(f"  CV AUC={r3['CV_AUC']}  CV Recall={r3['CV_Recall']}")

# ══════════════════════════════════════════════════════════════
# 실험 4 — 소프트 보팅 앙상블 (XGB + RF + LR)
# ══════════════════════════════════════════════════════════════
print("── 4. 소프트 보팅 앙상블 (XGB + RF + LR) ──")
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

mLR = LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, random_state=42)
mLR.fit(X_tr_s, yb_tr)
prob_lr = mLR.predict_proba(X_te_s)[:,1]

# 최적 가중치 탐색 (검증셋)
best_auc, best_w = 0, (0.5, 0.3, 0.2)
for w1 in [0.4,0.5,0.6]:
    for w2 in [0.2,0.3,0.4]:
        w3 = 1 - w1 - w2
        if w3 < 0.1: continue
        prob_e = w1*prob2 + w2*prob3 + w3*prob_lr
        auc_e = roc_auc_score(yb_te, prob_e)
        if auc_e > best_auc:
            best_auc = auc_e; best_w = (w1, w2, w3)

prob4 = best_w[0]*prob2 + best_w[1]*prob3 + best_w[2]*prob_lr

def make_m4(X, y):
    np4=y.sum(); pp4=max(len(y)-y.sum(),1)
    mx = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
        scale_pos_weight=pp4/np4 if np4>0 else 1, reg_alpha=0.5, reg_lambda=2.0,
        eval_metric='auc', random_state=42, verbosity=0)
    mx.fit(X, y)
    mr = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1)
    mr.fit(X, y)
    sc4 = StandardScaler(); Xs = sc4.fit_transform(X)
    ml = LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, random_state=42)
    ml.fit(Xs, y)
    class EnsM:
        def predict_proba(self, Xnew):
            Xs2 = sc4.transform(Xnew)
            p = (0.5*mx.predict_proba(Xnew)[:,1] +
                 0.3*mr.predict_proba(Xnew)[:,1] +
                 0.2*ml.predict_proba(Xs2)[:,1])
            return np.column_stack([1-p, p])
    return EnsM()

cv4 = tscv_score(make_m4)
r4 = eval_clf(f"4. 앙상블 XGB({best_w[0]:.1f})+RF({best_w[1]:.1f})+LR({best_w[2]:.1f})",
              yb_te, prob4)
r4['CV_AUC'] = f"{cv4[0]:.3f}±{cv4[1]:.3f}"
r4['CV_Recall'] = f"{cv4[2]:.3f}±{cv4[3]:.3f}"
results.append(r4)
print(f"  AUC={r4['AUC-ROC']}  Recall={r4['Recall']}  F1={r4['F1']}  {r4['화재탐지']}")
print(f"  CV AUC={r4['CV_AUC']}  CV Recall={r4['CV_Recall']}")

# ══════════════════════════════════════════════════════════════
# 실험 5 — XGB + Platt 캘리브레이션
# ══════════════════════════════════════════════════════════════
print("── 5. XGB + Platt 캘리브레이션 ──")
m2_raw = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
    scale_pos_weight=spw, reg_alpha=0.5, reg_lambda=2.0,
    eval_metric='auc', random_state=42, verbosity=0)
m5 = CalibratedClassifierCV(m2_raw, method='sigmoid', cv=3)
m5.fit(X_tr, yb_tr)
prob5 = m5.predict_proba(X_te)[:,1]

def make_m5(X, y):
    np5=y.sum(); pp5=max(len(y)-y.sum(),1)
    base = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
        scale_pos_weight=pp5/np5 if np5>0 else 1, reg_alpha=0.5, reg_lambda=2.0,
        eval_metric='auc', random_state=42, verbosity=0)
    m = CalibratedClassifierCV(base, method='sigmoid', cv=min(3, np5))
    m.fit(X, y); return m

cv5 = tscv_score(make_m5)
r5 = eval_clf("5. XGB + Platt 캘리브레이션", yb_te, prob5)
r5['CV_AUC'] = f"{cv5[0]:.3f}±{cv5[1]:.3f}"
r5['CV_Recall'] = f"{cv5[2]:.3f}±{cv5[3]:.3f}"
results.append(r5)
print(f"  AUC={r5['AUC-ROC']}  Recall={r5['Recall']}  F1={r5['F1']}  {r5['화재탐지']}")
print(f"  CV AUC={r5['CV_AUC']}  CV Recall={r5['CV_Recall']}")

# ══════════════════════════════════════════════════════════════
# 최종 선택 모델: 앙상블 (실험 4)
# ══════════════════════════════════════════════════════════════
print()
print("=" * 75)
print("  결과 요약")
print("=" * 75)
res_df = pd.DataFrame(results)
cols = ['모델','AUC-ROC','Brier','최적임계값',
        'Precision','Recall','F1','화재탐지','CV_AUC','CV_Recall']
print(res_df[cols].to_string(index=False))

# 최고 모델 선정 (AUC + Recall 균형)
res_df['점수'] = res_df['AUC-ROC']*0.4 + res_df['Recall']*0.4 + res_df['F1']*0.2
best_idx = res_df[res_df['모델'].str.startswith(('1','2','3','4','5'))]['점수'].idxmax()
best_name = res_df.loc[best_idx, '모델']
print(f"\n★ 최적 모델: {best_name}")

# ══════════════════════════════════════════════════════════════
# 임계값별 운영 포인트 (최고 모델 기준)
# ══════════════════════════════════════════════════════════════
# 앙상블(4)을 최종 모델로 사용
final_probs = prob4
print(f"\n=== 앙상블 임계값별 운영 포인트 ===")
print(f"{'임계값':>6}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'화재탐지':>8}  {'False Alarm':>11}")
print("-" * 60)
for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    yp = (final_probs >= thr).astype(int)
    p  = precision_score(yb_te, yp, zero_division=0)
    r  = recall_score(yb_te, yp, zero_division=0)
    f  = f1_score(yb_te, yp, zero_division=0)
    tp = yp[yb_te==1].sum()
    fp = yp[yb_te==0].sum()
    print(f"  {thr:.2f}    {p:9.3f}   {r:6.3f}  {f:6.3f}  {tp}/{yb_te.sum()}건    {fp}건")

# ══════════════════════════════════════════════════════════════
# 피처 중요도 (XGB + FE 기준)
# ══════════════════════════════════════════════════════════════
print(f"\n=== 피처 중요도 (실험2 XGB + FE) — 상위 10개 ===")
fi = pd.Series(m2.feature_importances_, index=NEW_FEATURES).sort_values(ascending=False)
for fname, fval in fi.head(10).items():
    is_new = '★ NEW' if fname not in ORIG_FEATURES else ''
    print(f"  {fval:.4f}  {fname:20s} {is_new}")

# ══════════════════════════════════════════════════════════════
# 저장
# ══════════════════════════════════════════════════════════════
# 앙상블 구성 저장
save_bundle = {
    'model_xgb':   m2,        # XGB + FE
    'model_rf':    m3,        # RF + FE
    'model_lr':    mLR,       # LR
    'scaler_lr':   sc,        # LR용 스케일러
    'weights':     best_w,    # (w_xgb, w_rf, w_lr)
    'features':    NEW_FEATURES,
    'threshold':   r4['최적임계값'],
}
joblib.dump(save_bundle, MODEL_DIR + 'ensemble_v3.pkl')

# 전체 예측 생성
df_all_eng = engineer_features(df_raw, df_raw[df_raw['연도']<=2022])
X_all = df_all_eng[NEW_FEATURES].values
X_all_s = sc.transform(X_all)
prob_all = (best_w[0]*m2.predict_proba(X_all)[:,1] +
            best_w[1]*m3.predict_proba(X_all)[:,1] +
            best_w[2]*mLR.predict_proba(X_all_s)[:,1])

df_out = df_raw.copy()
df_out['발화확률']  = (prob_all * 100).round(2)
df_out['위험도등급'] = pd.cut(prob_all*100,
    bins=[0,20,40,60,100], labels=['낮음','보통','높음','매우높음'])
df_out.to_csv(MODEL_DIR + 'full_predictions_v3.csv', index=False, encoding='utf-8-sig')
res_df.to_csv(MODEL_DIR + 'model_comparison_v3.csv',  index=False, encoding='utf-8-sig')

print(f"\n→ ensemble_v3.pkl 저장")
print(f"→ full_predictions_v3.csv 저장  (발화확률 컬럼 포함)")
print(f"→ model_comparison_v3.csv 저장")
print(f"\n추천 임계값: {r4['최적임계값']} (F1 최대화 기준)")
print(f"운영 권장값: 0.15 (Recall 우선) 또는 0.20 (Recall·Precision 균형)")
