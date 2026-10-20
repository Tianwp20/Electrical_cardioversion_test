import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import imblearn

# 加载保存的随机森林模型
model = joblib.load('best_RF_model.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "AtrialFibrillationType": {"type": "categorical", "options": [0, 1], "default": 0},
    "HeartValveDisease": {"type": "categorical", "options": [0, 1], "default": 0},
    "LateRelapse": {"type": "categorical", "options": [0, 1], "default": 0},
    "HF": {"type": "categorical", "options": [0, 1], "default": 0},
    "HbA1c": {"type": "numerical", "min": 0.000, "max": 20.000, "default": 7.0},
    "Cardiomyopathy": {"type": "categorical", "options": [0, 1], "default": 0},
    "B": {"type": "categorical", "options": [0, 1], "default": 0},
    "Amiodarone": {"type": "categorical", "options": [0, 1], "default": 0},
    "Rivaroxaban": {"type": "categorical", "options": [0, 1], "default": 0},
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of AKI is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    X_input = pd.DataFrame(
        [feature_values],
        columns=list(feature_ranges.keys())
    )
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)
        # 兼容不同 SHAP 版本的返回格式
        if isinstance(shap_values, list):
            # 旧版 SHAP：每个类别对应一个数组
            values_for_class = shap_values[predicted_class][0]
            base_value = np.asarray(
                explainer.expected_value
            ).reshape(-1)[predicted_class]
        else:
            # 新版 SHAP：通常为 (样本数, 特征数, 类别数)
            values_array = np.asarray(shap_values)
            if values_array.ndim == 3:
                values_for_class = values_array[0, :, predicted_class]
                base_value = np.asarray(
                    explainer.expected_value
                ).reshape(-1)[predicted_class]
            elif values_array.ndim == 2:
                values_for_class = values_array[0]
                base_value = np.asarray(
                    explainer.expected_value
                ).reshape(-1)[0]
            else:
                raise ValueError(
                    f"无法识别 SHAP 值维度：{values_array.shape}"
                )
        # 绘制力图
        shap_fig = shap.force_plot(
            base_value,
            values_for_class,
            X_input.iloc[0],
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()
    except Exception as e:
        st.error(f"SHAP 解释失败：{type(e).__name__}: {e}")
        st.info("请检查模型类型、SHAP 版本及模型训练时的特征格式。")
