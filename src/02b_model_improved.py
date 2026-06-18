"""
02b_model_improved.py
변압기 화재 예측 모델 개선 실험 v2.0
==============================================
문제 진단:
  - 과적합: 훈련 R²=0.87 vs 검증 R²=-0.17
  - 클래스 불균형: 화재 발생 7.8% (81/1020)
  - 극값 미반영: 발생월 실제 1.06건 vs 예측 0.17건
  - TimeSeriesSplit 미적용

개선 실험:
  A. 기준선: XGBoost Regressor (원본)
  B. XGBoost count:poisson 목적함수
  C. XGBoost count:poisson + 튜닝 + 규제 강화
  D. Two-Stage Hurdle Model (이진분류 × 조건부회귀)
  E. Poisson GLM (sklearn)
  F. XGBoost count:poisson + SMOTE
"""
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             f1_score, recall_score, precision_score, roc_auc_score,
                             classification_report)
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False
    print("⚠️  SMOTE 미사용 (imbalanced-learn 로드 실패)")

MODEL_DIR = "/Users/seungwookim/Desktop/TransFireRisk/model/"
df = pd.read_csv(MODEL_DIR + "model_input.csv")

FEATURES = [
    '월최고기온','월평균기온','월최저기온','월평균습도','월강수합계','월최대풍속',
    '월평균일교차','강수일수','월전3일평균기온','월전7일강수합계','월연속고온일수',
    '월','계절코드','시도코드','전년전기화재건수','전년변압기화재건수',
]
TARGET = '변압기화재건수'
THRESHOLD = 0.3   # 화재 발생 예측 임계값 (건/월)

# ── 데이터 분할 ────────────────────────────────────────────────
train_df = df[df['연도'] <= 2022].copy()
test_df  = df[df['연도'] >= 2023].copy()
X_tr, y_tr = train_df[FEATURES].values, train_df[TARGET].values
X_te, y_te = test_df[FEATURES].values,  test_df[TARGET].values

n_pos = (y_tr > 0).sum()
n_neg = (y_tr == 0).sum()
spw   = n_neg / n_pos   # scale_pos_weight ≈ 11.5

print(f"훈련셋: {len(X_tr)}행  (화재 발생: {n_pos}건 / {n_pos/len(X_tr)*100:.1f}%)")
print(f"검증셋: {len(X_te)}행  (화재 발생: {(y_te>0).sum()}건 / {(y_te>0).mean()*100:.1f}%)")
print(f"scale_pos_weight: {spw:.1f}")
print()

# ── TimeSeriesSplit (4-fold 확장창 CV) ──────────────────────────
TS_FOLDS = [
    ([2020],              [2021]),
    ([2020,2021],         [2022]),
    ([2020,2021,2022],    [2023]),
    ([2020,2021,2022,2023],[2024]),
]

def ts_cv(predict_fn, label=""):
    """4-fold 시계열 CV — RMSE, Recall 반환"""
    rmses, recalls = [], []
    for train_yrs, test_yrs in TS_FOLDS:
        tr = df[df['연도'].isin(train_yrs)]
        te = df[df['연도'].isin(test_yrs)]
        Xtr, ytr = tr[FEATURES].values, tr[TARGET].values
        Xte, yte = te[FEATURES].values, te[TARGET].values
        pred = predict_fn(Xtr, ytr, Xte)
        rmses.append(np.sqrt(mean_squared_error(yte, pred)))
        recalls.append(recall_score((yte>0).astype(int), (pred>THRESHOLD).astype(int),
                                    zero_division=0))
    return np.mean(rmses), np.std(rmses), np.mean(recalls)

# ── 평가 함수 ─────────────────────────────────────────────────
def evaluate(name, y_true_tr, y_pred_tr, y_true_te, y_pred_te,
             cv_rmse=None, cv_rmse_std=None, cv_recall=None):
    def reg_metrics(yt, yp):
        return (np.sqrt(mean_squared_error(yt, yp)),
                mean_absolute_error(yt, yp),
                r2_score(yt, yp))
    def cls_metrics(yt, yp):
        yb_t = (yt > 0).astype(int)
        yb_p = (yp > THRESHOLD).astype(int)
        prec = precision_score(yb_t, yb_p, zero_division=0)
        rec  = recall_score(yb_t, yb_p, zero_division=0)
        f1   = f1_score(yb_t, yb_p, zero_division=0)
        try:   auc = roc_auc_score(yb_t, yp)
        except: auc = 0.0
        return prec, rec, f1, auc

    tr_rmse, tr_mae, tr_r2 = reg_metrics(y_true_tr, y_pred_tr)
    te_rmse, te_mae, te_r2 = reg_metrics(y_true_te, y_pred_te)
    te_prec, te_rec, te_f1, te_auc = cls_metrics(y_true_te, y_pred_te)

    return {
        '모델': name,
        '훈련_RMSE': round(tr_rmse, 4),
        '검증_RMSE': round(te_rmse, 4),
        '훈련_R²':   round(tr_r2,   4),
        '검증_R²':   round(te_r2,   4),
        '검증_MAE':  round(te_mae,  4),
        '발생_Prec': round(te_prec, 3),
        '발생_Recall':round(te_rec, 3),
        '발생_F1':   round(te_f1,  3),
        'AUC-ROC':   round(te_auc, 3),
        'CV_RMSE':   f"{cv_rmse:.4f}±{cv_rmse_std:.4f}" if cv_rmse else '-',
        'CV_Recall': round(cv_recall, 3) if cv_recall else '-',
    }

results = []

# ══════════════════════════════════════════════════════════════
# 실험 A: 기준선 — XGBoost Regressor (reg:squarederror)
# ══════════════════════════════════════════════════════════════
print("── 실험 A: 기준선 XGBoost Regressor ──")
A_params = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
mA = xgb.XGBRegressor(**A_params)
mA.fit(X_tr, y_tr)
pA_tr = mA.predict(X_tr).clip(0)
pA_te = mA.predict(X_te).clip(0)

def cv_A(Xtr, ytr, Xte):
    m = xgb.XGBRegressor(**A_params); m.fit(Xtr, ytr)
    return m.predict(Xte).clip(0)
cvA = ts_cv(cv_A)
results.append(evaluate("A. XGB Regressor (기준)", y_tr, pA_tr, y_te, pA_te, *cvA))
print(f"  검증 RMSE={results[-1]['검증_RMSE']}  R²={results[-1]['검증_R²']}  "
      f"Recall={results[-1]['발생_Recall']}  F1={results[-1]['발생_F1']}")

# ══════════════════════════════════════════════════════════════
# 실험 B: XGBoost count:poisson
# ══════════════════════════════════════════════════════════════
print("── 실험 B: XGBoost count:poisson ──")
B_params = dict(objective='count:poisson', n_estimators=300, max_depth=4,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbosity=0)
mB = xgb.XGBRegressor(**B_params)
mB.fit(X_tr, y_tr)
pB_tr = mB.predict(X_tr).clip(0)
pB_te = mB.predict(X_te).clip(0)

def cv_B(Xtr, ytr, Xte):
    m = xgb.XGBRegressor(**B_params); m.fit(Xtr, ytr)
    return m.predict(Xte).clip(0)
cvB = ts_cv(cv_B)
results.append(evaluate("B. XGB count:poisson", y_tr, pB_tr, y_te, pB_te, *cvB))
print(f"  검증 RMSE={results[-1]['검증_RMSE']}  R²={results[-1]['검증_R²']}  "
      f"Recall={results[-1]['발생_Recall']}  F1={results[-1]['발생_F1']}")

# ══════════════════════════════════════════════════════════════
# 실험 C: XGBoost count:poisson + 규제 강화 + 튜닝
# ══════════════════════════════════════════════════════════════
print("── 실험 C: XGBoost count:poisson + 규제 강화 ──")
C_params = dict(objective='count:poisson', n_estimators=500, max_depth=3,
                learning_rate=0.02, subsample=0.7, colsample_bytree=0.7,
                min_child_weight=5, reg_alpha=1.0, reg_lambda=3.0,
                gamma=0.5, random_state=42, verbosity=0)
mC = xgb.XGBRegressor(**C_params)
mC.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
pC_tr = mC.predict(X_tr).clip(0)
pC_te = mC.predict(X_te).clip(0)

def cv_C(Xtr, ytr, Xte):
    m = xgb.XGBRegressor(**C_params); m.fit(Xtr, ytr)
    return m.predict(Xte).clip(0)
cvC = ts_cv(cv_C)
results.append(evaluate("C. XGB Poisson + 규제강화", y_tr, pC_tr, y_te, pC_te, *cvC))
print(f"  검증 RMSE={results[-1]['검증_RMSE']}  R²={results[-1]['검증_R²']}  "
      f"Recall={results[-1]['발생_Recall']}  F1={results[-1]['발생_F1']}")

# ══════════════════════════════════════════════════════════════
# 실험 D: Two-Stage Hurdle Model
#   Stage-1: XGBClassifier (발생/미발생)
#   Stage-2: XGBRegressor (발생 행에 대한 건수)
#   예측: P(발생) × E[건수|발생]
# ══════════════════════════════════════════════════════════════
print("── 실험 D: Two-Stage Hurdle Model ──")

class HurdleModel:
    def __init__(self, clf_p, reg_p):
        self.clf = xgb.XGBClassifier(**clf_p)
        self.reg = xgb.XGBRegressor(**reg_p)
        self._reg_mean = 1.0

    def fit(self, X, y):
        yb = (y > 0).astype(int)
        self.clf.fit(X, yb)
        fire_mask = y > 0
        if fire_mask.sum() >= 5:
            self.reg.fit(X[fire_mask], y[fire_mask])
        else:
            self._reg_mean = y[fire_mask].mean() if fire_mask.sum() > 0 else 1.0
        return self

    def predict(self, X):
        p_fire = self.clf.predict_proba(X)[:, 1]
        try:
            cnt = self.reg.predict(X).clip(0.1)
        except Exception:
            cnt = np.full(len(X), self._reg_mean)
        return (p_fire * cnt).clip(0)

    def predict_proba_fire(self, X):
        return self.clf.predict_proba(X)[:, 1]

clf_p = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
             scale_pos_weight=spw, reg_alpha=0.5, reg_lambda=2.0,
             eval_metric='logloss', random_state=42, verbosity=0)
reg_p = dict(objective='count:poisson', n_estimators=200, max_depth=3,
             learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=2, reg_alpha=0.5, reg_lambda=2.0,
             random_state=42, verbosity=0)

mD = HurdleModel(clf_p, reg_p)
mD.fit(X_tr, y_tr)
pD_tr = mD.predict(X_tr)
pD_te = mD.predict(X_te)

def cv_D(Xtr, ytr, Xte):
    m = HurdleModel(clf_p, reg_p); m.fit(Xtr, ytr)
    return m.predict(Xte)
cvD = ts_cv(cv_D)
results.append(evaluate("D. Hurdle Model (2-Stage)", y_tr, pD_tr, y_te, pD_te, *cvD))
print(f"  검증 RMSE={results[-1]['검증_RMSE']}  R²={results[-1]['검증_R²']}  "
      f"Recall={results[-1]['발생_Recall']}  F1={results[-1]['발생_F1']}")

# ══════════════════════════════════════════════════════════════
# 실험 E: Poisson GLM (sklearn)
# ══════════════════════════════════════════════════════════════
print("── 실험 E: Poisson GLM ──")
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
mE = PoissonRegressor(alpha=0.1, max_iter=2000)
mE.fit(X_tr_s, y_tr)
pE_tr = mE.predict(X_tr_s).clip(0)
pE_te = mE.predict(X_te_s).clip(0)

def cv_E(Xtr, ytr, Xte):
    sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    m = PoissonRegressor(alpha=0.1, max_iter=2000); m.fit(Xtr_s, ytr)
    return m.predict(Xte_s).clip(0)
cvE = ts_cv(cv_E)
results.append(evaluate("E. Poisson GLM", y_tr, pE_tr, y_te, pE_te, *cvE))
print(f"  검증 RMSE={results[-1]['검증_RMSE']}  R²={results[-1]['검증_R²']}  "
      f"Recall={results[-1]['발생_Recall']}  F1={results[-1]['발생_F1']}")

# ══════════════════════════════════════════════════════════════
# 실험 F: XGBoost count:poisson + SMOTE
# ══════════════════════════════════════════════════════════════
if HAS_SMOTE:
    print("── 실험 F: XGBoost count:poisson + SMOTE ──")
    try:
        smote = SMOTE(k_neighbors=3, random_state=42)
        # SMOTE는 이진 타깃 기반 오버샘플링
        yb_tr = (y_tr > 0).astype(int)
        X_sm, yb_sm = smote.fit_resample(X_tr, yb_tr)
        # 오버샘플된 행의 연속 타깃 복원: 양성 샘플 평균 사용
        y_fire_mean = y_tr[y_tr > 0].mean()
        y_sm = np.where(yb_sm == 1, y_fire_mean, 0.0)

        F_params = dict(objective='count:poisson', n_estimators=400, max_depth=4,
                        learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                        min_child_weight=3, reg_alpha=0.5, reg_lambda=2.0,
                        random_state=42, verbosity=0)
        mF = xgb.XGBRegressor(**F_params)
        mF.fit(X_sm, y_sm)
        pF_tr = mF.predict(X_tr).clip(0)
        pF_te = mF.predict(X_te).clip(0)

        def cv_F(Xtr, ytr, Xte):
            try:
                yb = (ytr > 0).astype(int)
                sm2 = SMOTE(k_neighbors=min(3, (yb==1).sum()-1), random_state=42)
                Xs, ybs = sm2.fit_resample(Xtr, yb)
                ys = np.where(ybs==1, ytr[ytr>0].mean() if (ytr>0).sum()>0 else 1.0, 0.0)
                m = xgb.XGBRegressor(**F_params); m.fit(Xs, ys)
            except Exception:
                m = xgb.XGBRegressor(**F_params); m.fit(Xtr, ytr)
            return m.predict(Xte).clip(0)
        cvF = ts_cv(cv_F)
        results.append(evaluate("F. XGB Poisson + SMOTE", y_tr, pF_tr, y_te, pF_te, *cvF))
        print(f"  검증 RMSE={results[-1]['검증_RMSE']}  R²={results[-1]['검증_R²']}  "
              f"Recall={results[-1]['발생_Recall']}  F1={results[-1]['발생_F1']}")
    except Exception as e:
        print(f"  SMOTE 실패: {e}")
else:
    print("── 실험 F: SMOTE 미설치 — 건너뜀 ──")

# ══════════════════════════════════════════════════════════════
# 결과 요약
# ══════════════════════════════════════════════════════════════
res_df = pd.DataFrame(results)

print()
print("=" * 90)
print("  실험 결과 요약")
print("=" * 90)
cols_show = ['모델','훈련_RMSE','검증_RMSE','훈련_R²','검증_R²',
             '발생_Recall','발생_F1','AUC-ROC','CV_RMSE','CV_Recall']
print(res_df[cols_show].to_string(index=False))

# 최적 모델 선정 (검증 Recall + F1 기준)
res_df['점수'] = res_df['발생_Recall'] * 0.5 + res_df['발생_F1'] * 0.3 + \
                 (1 / (res_df['검증_RMSE'] + 0.001)) * 0.2
best_idx  = res_df['점수'].idxmax()
best_name = res_df.loc[best_idx, '모델']
print()
print(f"  ★ 최적 모델 (Recall·F1·RMSE 종합): {best_name}")

# ── 저장 ────────────────────────────────────────────────────
res_df.to_csv(MODEL_DIR + "model_comparison.csv", index=False, encoding='utf-8-sig')
print(f"  → model_comparison.csv 저장 완료")

# 최적 모델 저장
best_model_map = {
    'A': mA, 'B': mB, 'C': mC, 'D': mD, 'E': mE,
}
if HAS_SMOTE and 'F' in best_name: best_model_map['F'] = mF
best_key = best_name[0]
best_model = best_model_map.get(best_key, mD)
joblib.dump(best_model, MODEL_DIR + "best_model_v2.pkl")
print(f"  → best_model_v2.pkl 저장 완료 ({best_name})")

# 최적 모델 전체 예측 저장 (기존 full_predictions와 비교용)
try:
    X_all = df[FEATURES].values
    if best_key in ['A','B','C','E']:
        if best_key == 'E':
            X_all_s = scaler.transform(X_all)
            pred_all = best_model.predict(X_all_s).clip(0)
        else:
            pred_all = best_model.predict(X_all).clip(0)
    else:  # Hurdle
        pred_all = best_model.predict(X_all)

    df_out = df.copy()
    df_out['예측건수_v2'] = pred_all.round(3)
    df_out.to_csv(MODEL_DIR + "full_predictions_v2.csv", index=False, encoding='utf-8-sig')
    print(f"  → full_predictions_v2.csv 저장 완료")
except Exception as e:
    print(f"  예측 저장 오류: {e}")

# ── 화재 발생 행 상세 분류 리포트 (최적 모델) ────────────────
print()
print(f"=== 최적 모델({best_name}) 검증셋 분류 리포트 ===")
if best_key == 'D':
    pred_te_best = mD.predict(X_te)
elif best_key == 'E':
    pred_te_best = mE.predict(scaler.transform(X_te)).clip(0)
else:
    pred_te_best = best_model.predict(X_te).clip(0)

print(classification_report(
    (y_te > 0).astype(int),
    (pred_te_best > THRESHOLD).astype(int),
    target_names=['미발생','발생'], digits=3
))

# 발생월 예측값 분포 비교
print("=== 발생월 예측값 분포 ===")
fire_mask_te = y_te > 0
print(f"  기준선(A) 발생월 예측 평균: {pA_te[fire_mask_te].mean():.3f}")
print(f"  최적모델  발생월 예측 평균: {pred_te_best[fire_mask_te].mean():.3f}")
print(f"  실제      발생월 평균:      {y_te[fire_mask_te].mean():.3f}")
