import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import imblearn

# 加载保存的随机森林模型
model = joblib.load('best_rf_model.pkl')

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
        # 提取 Pipeline 最后一步的模型
        if hasattr(model, "steps"):
            classifier = model.steps[-1][1]
            preprocessors = model.steps[:-1]
            # 对输入数据执行前面的转换步骤
            X_transformed = X_input.copy()
            for name, step in preprocessors:
                # SMOTE 等采样器只在训练时采样，
                # 预测时不应再次执行
                if hasattr(step, "fit_resample"):
                    continue
                if hasattr(step, "transform"):
                    X_transformed = step.transform(X_transformed)
            # 如果转换后仍为数组，恢复为 DataFrame
            if not isinstance(X_transformed, pd.DataFrame):
                X_transformed = pd.DataFrame(
                    X_transformed,
                    columns=getattr(
                        classifier,
                        "feature_names_in_",
                        None
                    )
                )
        else:
            classifier = model
            X_transformed = X_input

        # 创建树模型解释器
        explainer = shap.TreeExplainer(classifier)
        shap_exp = explainer(X_transformed)

        # 选取当前预测类别对应的 SHAP 值
        if shap_exp.values.ndim == 3:
            class_index = list(classifier.classes_).index(
                predicted_class
            )
            shap_exp = shap_exp[..., class_index]

        # 取第一个（唯一）样本的解释，瀑布图需要 n_features 维
        if shap_exp.values.ndim == 2:
            shap_exp = shap_exp[0]

        # 绘制 SHAP 瀑布图
        shap.plots.waterfall(
            shap_exp,
            max_display=10,
            show=False
        )
        st.pyplot(plt.gcf())
        plt.close()
    except Exception as e:
        st.error(f"SHAP 解释失败：{type(e).__name__}: {e}")
