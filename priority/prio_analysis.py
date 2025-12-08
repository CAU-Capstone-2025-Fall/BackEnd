import pandas as pd
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Matplotlib 등에서 발생하는 경고를 무시하여 깔끔한 출력을 보장합니다.
warnings.filterwarnings("ignore", category=UserWarning)

def analyze_adoption_features(file_path):
    """
    반려동물 입양 데이터를 분석하여 입양에 영향을 미치는 주요 특성을 찾고 시각화합니다.

    Args:
        file_path (str): 분석할 CSV 파일의 경로.
    """
    # 1. 데이터 로드
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return

    print("--- 데이터 정보 ---")
    df.info()
    print("\n--- 데이터 샘플 (상위 5개) ---")
    print(df.head())

    # PetID는 분석에 불필요하므로 제외
    df = df.drop('PetID', axis=1)

    # 2. 데이터 전처리 준비
    # 특성을 범주형과 수치형으로 분리
    categorical_features = ['PetType', 'Breed', 'Color', 'Size']
    numerical_features = ['AgeMonths', 'WeightKg', 'TimeInShelterDays', 'AdoptionFee']
    # 이진(binary) 특성은 그대로 사용 가능
    binary_features = ['Vaccinated', 'HealthCondition', 'PreviousOwner']

    # 타겟 변수 설정
    X = df.drop('AdoptionLikelihood', axis=1)
    y = df['AdoptionLikelihood']

    # 전처리 파이프라인 생성
    # - 수치형 특성: StandardScaler를 통해 스케일링
    # - 범주형 특성: OneHotEncoder를 통해 숫자 데이터로 변환
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('bin', 'passthrough', binary_features) # 이진 특성은 변환 없이 통과
        ],
        remainder='passthrough' # 나머지 열은 그대로 둠
    )

    # 3. 모델 학습
    # RandomForestClassifier 모델과 전처리 파이프라인을 결합
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True))
    ])

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 모델 학습
    model.fit(X_train, y_train)

    # 모델 성능 평가
    y_pred = model.predict(X_test)
    print("\n--- 모델 성능 평가 ---")
    print(f"정확도: {accuracy_score(y_test, y_pred):.4f}")
    print("\n[분류 리포트]")
    print(classification_report(y_test, y_pred, target_names=['입양 안됨 (0)', '입양됨 (1)']))
    print(f"OOB Score: {model.named_steps['classifier'].oob_score_:.4f} (학습에 사용되지 않은 데이터로 검증한 점수)")


    # 4. 특성 중요도 분석
    # 전처리 단계에서 생성된 전체 특성 이름 가져오기
    ohe_feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + ohe_feature_names.tolist() + binary_features

    # 특성 중요도 추출
    importances = model.named_steps['classifier'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print("\n--- 입양 결정에 중요한 상위 15개 특성 ---")
    print(feature_importance_df.head(15))

    # 5. 시각화
    plt.style.use('seaborn-v0_8-talk')
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), palette='viridis')
    plt.title('top 15 features influencing adoption likelihood', fontsize=20, pad=20)
    plt.xlabel('Feature Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()

    plt.show()

def analyze_adoption_features2(file_path):
    """
    SHAP을 사용하여 반려동물 입양에 영향을 미치는 주요 특성을 분석하고,
    양/음의 상관관계를 시각화합니다.

    Args:
        file_path (str): 분석할 CSV 파일의 경로.
    """
    # 1. 데이터 로드
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return

    df = df.drop('PetID', axis=1)

    # 2. 데이터 전처리 준비
    categorical_features = ['PetType', 'Breed', 'Color', 'Size']
    numerical_features = ['AgeMonths', 'WeightKg', 'TimeInShelterDays', 'AdoptionFee']
    binary_features = ['Vaccinated', 'HealthCondition', 'PreviousOwner']

    X = df.drop('AdoptionLikelihood', axis=1)
    y = df['AdoptionLikelihood']
    
    # 컬럼 순서를 명시적으로 고정하여 안정성 확보
    column_order = numerical_features + categorical_features + binary_features
    X = X[column_order]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features + binary_features), # binary도 스케일링
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )

    # 3. 모델 학습
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)

    print("\n--- 모델 성능 평가 ---")
    print(f"정확도: {accuracy_score(model.predict(X_test), y_test):.4f}")
    print(f"OOB Score: {model.named_steps['classifier'].oob_score_:.4f}")

    # 4. SHAP 분석을 위한 데이터 준비
    print("\n--- SHAP 분석 진행 중 ---")

    X_test_transformed = model.named_steps['preprocessor'].transform(X_test)
    ohe_feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + binary_features + list(ohe_feature_names)

    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=all_feature_names)

    # 5. SHAP Explainer 생성 및 값 계산
    explainer = shap.TreeExplainer(model.named_steps['classifier'])
    shap_values_obj = explainer(X_test_transformed_df) # SHAP 객체 생성

    print("--- SHAP 분석 완료 ---")

    # 6. SHAP 시각화
    print("\n--- SHAP 요약 플롯 생성 ---")
    
    plt.title('Features Affecting Adoption Likelihood (Class 1: Adopted)', fontsize=16, pad=20)

    # !! 최종 수정 !!
    # shap_values 객체에서 클래스 1에 대한 값만 명시적으로 선택하여 플롯합니다.
    # 객체의 .values 속성을 사용하여 순수 numpy 배열에 접근하고, 인덱싱합니다.
    shap.summary_plot(shap_values_obj.values[:, :, 1], X_test_transformed_df, show=False)

    plt.tight_layout(pad=1.5)
    plt.show()
    

if __name__ == '__main__':
    # 여기에 CSV 파일 경로를 지정하세요.
    csv_file = 'pet_adoption_data.csv'
    analyze_adoption_features2(csv_file)

