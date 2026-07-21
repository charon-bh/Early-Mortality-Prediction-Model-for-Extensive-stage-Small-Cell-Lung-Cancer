import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings
import re
from io import BytesIO
from PIL import Image
warnings.filterwarnings('ignore')

# ===================== 基础配置 =====================
BLUE_COLOR = '#1f77b4'
RED_COLOR = '#d62728'
HIGH_LOW_CMAP = plt.cm.RdBu_r

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.facecolor'] = 'white'

st.set_page_config(
    page_title="Early Mortality Prediction Model for Extensive-stage Small Cell Lung Cancer",
    page_icon="📊",
    layout="wide"
)

# ===================== 特征列表 =====================
CATEGORICAL_FEATURES = [
    'Sex',
    'Income_Group',
    'T_Stage',
    'Radiation',
    'Chemotherapy',
    'Brain_metastasis',
    'Liver_metastasis',
    'Lung_metastasis'
]
CONTINUOUS_FEATURES = ['Age_Group']
selected_features = CATEGORICAL_FEATURES + CONTINUOUS_FEATURES
target_col = 'Outcome'

# ===================== 特征映射 =====================
FEATURE_VALUE_MAPPING = {
    'Sex': {0: 'Female', 1: 'Male'},
    'Income_Group': {0: 'Low', 1: 'Medium', 2: 'High'},

    'T_Stage': {0: 'T1', 1: 'T2', 2: 'T3', 3: 'T4'},
    'Radiation': {0: 'No', 1: 'Yes'},
    'Chemotherapy': {0: 'No', 1: 'Yes'},

    'Brain_metastasis': {0: 'No', 1: 'Yes'},
    'Liver_metastasis': {0: 'No', 1: 'Yes'},
    'Lung_metastasis': {0: 'No', 1: 'Yes'},

    'Outcome': {0: 'Survival', 1: 'Early Mortality'}
}

# ===================== 工具函数 =====================
def clean_text(text):
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = text.replace('：', ':').replace('，', ',').replace('。', '.')
    return text

def get_feature_text(feat_name, feat_value):
    if feat_name in CONTINUOUS_FEATURES:
        return str(feat_value)
    mapping = FEATURE_VALUE_MAPPING.get(feat_name, {})
    return mapping.get(feat_value, str(feat_value))

# 分类特征选项
FEATURE_OPTIONS = {}
for feat in CATEGORICAL_FEATURES:
    sorted_items = sorted(FEATURE_VALUE_MAPPING[feat].items(), key=lambda x: x[0])
    FEATURE_OPTIONS[feat] = [item[1] for item in sorted_items]
    FEATURE_OPTIONS[f"{feat}_values"] = [item[0] for item in sorted_items]

# ===================== 文件路径 =====================
MODEL_PATH = "LightGBM.pkl"
TRAIN_DATA_PATH = "traindata.csv"

# ===================== 模型与SHAP初始化 =====================
@st.cache_resource
def init_model():
    try:
        model_dict = joblib.load(MODEL_PATH)
        model = model_dict['model']


        optimal_threshold = model_dict['optimal_threshold']

        train_df = pd.read_csv(TRAIN_DATA_PATH, encoding="GBK")
        X_train = train_df[selected_features].copy()
        background_data = shap.sample(X_train, 100, random_state=42) if len(X_train) > 100 else X_train

        explainer = shap.TreeExplainer(
            model=model,
            data=background_data,
            model_output="raw"
        )

        base_value = explainer.expected_value
        if isinstance(base_value, list) and len(base_value) == 2:
            base_value = base_value[1]
        base_value = float(base_value)

        st.success("✅ Model & SHAP explainer ready")
        return model, optimal_threshold, explainer, base_value

    except Exception as e:
        st.error(f"❌ Load failed: {str(e)}")
        return None, None, None, None

# ===================== 原生 SHAP 力图 =====================
def generate_force_plot(base_value, shap_values, input_df, pred_label):
    plt.figure(figsize=(16, 5))
    shap.force_plot(
        base_value=base_value,
        shap_values=shap_values,
        features=input_df.iloc[0],
        feature_names=selected_features,
        show=False,
        matplotlib=True,
        text_rotation=0
    )
    plt.title(f'SHAP Force Plot - Pred={pred_label}', fontsize=12, pad=10)
    ax = plt.gca()

    # f(x) 加粗 + 位置修复
    for text in ax.texts:
        if "f(x)" in text.get_text():
            text.set_fontweight("bold")
            text.set_fontsize(12)
            text.set_y(text.get_position()[1] + 0.02)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

# ===================== 特征重要性排序图 =====================
def generate_importance_plot(shap_values):
    sorted_idx = np.argsort(np.abs(shap_values))[::-1]
    sorted_names = [selected_features[i] for i in sorted_idx]
    sorted_vals = shap_values[sorted_idx]
    colors = [RED_COLOR if v > 0 else BLUE_COLOR for v in sorted_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(sorted_names)), sorted_vals, color=colors)
    for i, v in enumerate(sorted_vals):
        offset = 0.003 if v >= 0 else -0.003
        ha = 'left' if v >= 0 else 'right'
        ax.text(v + offset, i, f'{v:+.4f}', va='center', ha=ha, fontsize=9)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('SHAP Value')
    ax.set_title('Feature Contribution', fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

# ===================== 主界面 =====================
def main():
    st.title("📊 Early Mortality Prediction Model for Extensive-stage Small Cell Lung Cancer")
    st.markdown("### LightGBM • SHAP Explanation")
    st.markdown("---")

    model, threshold, explainer, base_value = init_model()
    if model is None:
        st.stop()

    col_input, col_result = st.columns([1, 2])

    # ========== 左侧输入 ==========
    with col_input:
        st.subheader("🔍 Input Features")
        feature_values = []

        for feat in CATEGORICAL_FEATURES:
            opts = FEATURE_OPTIONS[feat]
            vals = FEATURE_OPTIONS[f"{feat}_values"]
            sel_text = st.selectbox(feat, opts, key=f"i_{feat}")
            sel_val = vals[opts.index(sel_text)]
            feature_values.append(sel_val)

        # 连续特征：Age_Group slider
        age_val = st.slider("Age_Group", 20, 85, 50, key="input_Age")
        feature_values.append(age_val)

        run_btn = st.button("🚀 Predict & Generate SHAP Plot", type="primary")

    # ========== 右侧结果 ==========
    with col_result:
        st.subheader("📈 Prediction & SHAP Explanation")

        if not run_btn:
            st.info("👈 Complete features and click to predict")
        else:
            with st.spinner("Calculating SHAP values..."):
                X = pd.DataFrame([feature_values], columns=selected_features)
                prob = model.predict_proba(X)[0]
                p0, p1 = prob[0], prob[1]
                pred = 1 if p1 >= threshold else 0

                # SHAP值处理
                sv = explainer.shap_values(X)
                if isinstance(sv, list) and len(sv) == 2:
                    sv_final = sv[1][0]
                else:
                    sv_final = sv[0]

                out0 = FEATURE_VALUE_MAPPING['Outcome'][0]
                out1 = FEATURE_VALUE_MAPPING['Outcome'][1]
                pred_label = FEATURE_VALUE_MAPPING['Outcome'][pred]

                # 预测结果
                st.markdown(f"""
                <div style="background:#f8f9fa; padding:20px; border-radius:12px; margin-bottom:20px;">
                    <h4 style="margin:0 0 12px 0;">Optimal Threshold: {threshold:.3f}</h4>
                    <div style="display:flex; justify-content:space-around;">
                        <div style="text-align:center;">
                            <p style="margin:0; font-size:14px;">{out0} Probability</p>
                            <p style="font-size:28px; font-weight:bold; color:{BLUE_COLOR};">{p0*100:.1f}%</p>
                        </div>
                        <div style="text-align:center;">
                            <p style="margin:0; font-size:14px;">{out1} Probability</p>
                            <p style="font-size:28px; font-weight:bold; color:{RED_COLOR};">{p1*100:.1f}%</p>
                        </div>
                    </div>
                    <h3 style="text-align:center; margin:10px 0 0 0; color:{'#d62728' if pred==1 else '#1f77b4'};">
                        Prediction: {pred} ({pred_label})
                    </h3>
                </div>
                """, unsafe_allow_html=True)

                # ===================== 原生 SHAP 力图 =====================
                st.markdown("---")
                st.subheader("🌀 SHAP Force Plot")

                force_img = generate_force_plot(base_value, sv_final, X, pred)
                st.image(force_img, use_container_width=True)

                st.markdown(f"""
                💡 **Explanation**
                - **Red features**: Increase the risk of {out1}
                - **Blue features**: Decrease the risk of {out1}
                """)

                # ===================== 特征重要性排序图 =====================
                st.markdown("---")
                st.subheader("📊 Feature Importance (SHAP)")

                imp_img = generate_importance_plot(sv_final)
                st.image(imp_img, use_container_width=True)

if __name__ == "__main__":
    main()
