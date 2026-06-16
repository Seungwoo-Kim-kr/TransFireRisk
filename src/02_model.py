"""
변압기 화재 위험도 예측 모델 - Step 2: XGBoost 학습 및 평가
프로젝트: TransFireRisk
"""
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
import xgboost as xgb

MODEL_DIR = "/Users/seungwookim/Desktop/TransFireRisk/model/"
df = pd.read_csv(MODEL_DIR + "model_input.csv")

FEATURES = [
    # 축 1: 당월 기상
    '월최고기온','월평균기온','월최저기온',
    '월평균습도','월강수합계','월최대풍속',
    '월평균일교차','강수일수',
    # 축 1-누적: 누적 기상
    '월전3일평균기온','월전7일강수합계','월연속고온일수',
    # 축 2: 시간
    '월','계절코드',
    # 축 3: 지역 이력
    '시도코드','전년전기화재건수','전년변압기화재건수',
]
TARGET = '변압기화재건수'

train = df[df['연도'] <= 2022]
test  = df[df['연도'] >= 2023]
X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"훈련: {len(X_train)}행 (2020~2022) | 화재 발생: {(y_train>0).sum()}건")
print(f"검증: {len(X_test)}행 (2023~2024)  | 화재 발생: {(y_test>0).sum()}건")

print("\n모델 학습 중...")
model = xgb.XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred_tr = model.predict(X_train).clip(0)
y_pred_te = model.predict(X_test).clip(0)

print("\n=== 성능 평가 ===")
for split, yt, yp in [("훈련", y_train, y_pred_tr), ("검증", y_test, y_pred_te)]:
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    r2   = r2_score(yt, yp)
    print(f"[{split}] RMSE:{rmse:.4f} | MAE:{mae:.4f} | R²:{r2:.4f}")

print("\n=== 화재 발생 분류 성능 (임계값 0.3) ===")
print(classification_report(
    (y_test>0).astype(int),
    (y_pred_te>0.3).astype(int),
    target_names=['미발생','발생']
))

cv = cross_val_score(model, df[FEATURES], df[TARGET], cv=5, scoring='neg_root_mean_squared_error')
print(f"5-Fold CV RMSE: {-cv.mean():.4f} ± {cv.std():.4f}")

def risk_grade(v):
    if   v < 0.3: return "낮음"
    elif v < 0.7: return "보통"
    elif v < 1.2: return "높음"
    else:         return "매우높음"

# 전체 예측
df['예측건수']   = model.predict(df[FEATURES]).clip(0).round(3)
df['위험도등급'] = df['예측건수'].apply(risk_grade)

# 피처 중요도
fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n=== 피처 중요도 ===")
for name, val in fi.items():
    print(f"  {name:<22} {'█'*int(val*200)} {val:.4f}")

# 저장
joblib.dump(model, MODEL_DIR + "xgb_transformer_fire.pkl")
fi.to_csv(MODEL_DIR + "feature_importance.csv", header=True)
df.to_csv(MODEL_DIR + "full_predictions.csv", index=False, encoding='utf-8-sig')

test_result = test[['시도','연도','월','계절','변압기화재건수']].copy()
test_result['예측건수']  = y_pred_te.round(2)
test_result['위험도등급'] = test_result['예측건수'].apply(risk_grade)
test_result.to_csv(MODEL_DIR + "test_predictions.csv", index=False, encoding='utf-8-sig')

print("\n✅ 모델·예측 저장 완료: TransFireRisk/model/")
