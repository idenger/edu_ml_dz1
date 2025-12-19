import re

import numpy as np
import seaborn as sns
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

from matplotlib import pyplot as plt

st.set_page_config(page_title="Den App", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

MODEL_PATH = Path(__file__).resolve().parent / "models.pkl"
INIT_PATH = Path(__file__).resolve().parent / "init.pkl"

# ---------------------------------------------------------------------------------

# Кэшируем модель (загружается только один раз)
@st.cache_resource
def load_models():
    with open(MODEL_PATH, "rb") as f:
        models = pickle.load(f)
    return models

@st.cache_resource
def load_init_data():
    with open(INIT_PATH, "rb") as f:
        init_data = pickle.load(f)
    return init_data

def prepare_features(df):
    df_proc = df.copy()

    if not "max_torque_rpm" in df_proc.columns:
        df_proc[['torque', 'max_torque_rpm']] = df_proc['torque'].apply(lambda x: pd.Series(torque_parser(x)))
        df_proc["torque"] = df_proc["torque"].fillna(0)
        df_proc["max_torque_rpm"] = df_proc["max_torque_rpm"].fillna(0)

    df_proc = df_proc.reindex(columns=FEATURE_NAMES)

    for column in ['mileage', 'engine', 'max_power']:
        if column in df_proc.columns:
            df_proc[column] = float_parser(df_proc[column])

    df_proc = pd.DataFrame(SCALER.transform(df_proc), columns=df_proc.columns, index=df_proc.index)
    return df_proc

def prepare_input(df):
    df_proc = df.reindex(columns=FEATURE_NAMES)
    df_proc = pd.DataFrame(SCALER.transform(df_proc), columns=df_proc.columns, index=df_proc.index)
    return df_proc

def float_parser(x):
    """Чистим числовые характеристики от букв внутри"""
    try:
        return x.str.extract(r"([\d\.]+)").astype(float)
    except ValueError:
        return float(x)

def torque_parser(text):
    """Разбиваем torque на torque и max_torque_rpm"""
    if not isinstance(text, str):
        return None, None
    text = text.lower()
    numbers = [float(num) for num in re.findall(r"\d+\.?\d*", text.replace(",", ""))]
    # Если три числа, то первые два - это диапазон rpm. Берем макс.
    if len(numbers) == 3:
        rpm = numbers[2]
        torque = numbers[0]
    # Если два - уже смотрим на порядок
    elif len(numbers) == 2:
        i_torque = abs(text.find("kgm") * text.find("nm")) # либо-либо
        i_rpm = text.find("rpm")
        if i_rpm != -1 and i_rpm < i_torque:
            rpm = numbers[0]
            torque = numbers[1]
        else:
            rpm = numbers[1]
            torque = numbers[0]
    else:
         return None, None
    # Если torque в kgm, то переводим в nm
    if not "nm" in text and "kgm" in text:
        torque = torque * 9.81
    return float(torque), int(rpm)

# ---------------------------------------------------------------------------------

# Подгружаем данные
try:
    models = load_models()
    init_data = load_init_data()
    MODEL = models["lasso_scaled_gs"]["model"]
    FEATURE_NAMES = models["lasso_scaled_gs"]["features"]
    SCALER = models["lasso_scaled_gs"]["scaler"]
    INFO = init_data["info"]
except Exception as e:
    st.error(f"❌ Ошибка загрузки моделей: {e}")
    st.stop()


st.title("🎯 Den App - Предсказание стоимости автомобилей")

with st.sidebar:
    uploaded_file = st.file_uploader("Загрузите CSV файл 👇", type=['csv'])

if uploaded_file is None:
    st.info("👈 Загрузите CSV файл для начала работы")
    st.info("👇 Или сделайте предсказание по параметрам")
else:

    try:
        csv_df = pd.read_csv(uploaded_file)
        csv_df.drop(columns="selling_price", inplace=True)
        csv_df.dropna(inplace=True)
        prepared_input = prepare_features(csv_df)
    except Exception as e:
        st.error(f"❌ Ошибка при обработке данных: {e}")
        st.stop()

    st.subheader("🔮 Предсказание готово 🔮")
    y_pred = MODEL.predict(prepared_input)

    results = csv_df.copy()
    results["selling_price"] = np.round(y_pred, 2)
    prepared_input["selling_price"] = np.round(y_pred, 2)

    st.dataframe(
        results.style
            .set_properties(
                subset=["selling_price"], **{"color": "green", "font-weight": "bold"}
            )
            .format({
                "selling_price": lambda x: "Ничего не стоит" if x < 0 else x
            })
    )

    st.subheader("📊 EDA")
    sns.set_style("dark")

    st.pyplot(sns.pairplot(prepared_input, diag_kind='kde').fig)

    pearson = prepared_input.corr(method="pearson", numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(pearson, annot=True, cmap="bwr", ax=ax)
    st.pyplot(fig)


# Форма для предсказания
st.subheader("🔮 Сделать предсказание вручную 🔮")

with st.form("prediction_form"):
    input_data = {}
    st.write("Укажите параметры:")
    for col in FEATURE_NAMES:
        val = int(INFO[col]["50%"])
        input_data[col] = st.number_input(col, value=val, key=f"num_{col}")

    submitted = st.form_submit_button("Предсказать", use_container_width=True)


if submitted:
    try:
        input_df = pd.DataFrame([input_data])
        prepared_input = prepare_input(input_df)
        y_pred = round(MODEL.predict(prepared_input)[0], 2)
        if (y_pred > 0):
            st.success(f"**Расчетная стоимость:**  {y_pred}")
        else:
            st.warning("К сожалению, такое ведро никого не заинтересует 😅")
    except Exception as e:
        st.error(f"❌ Ошибка при предсказании: {e}")


st.subheader("⚖️ Применяется модель \"Lasso + StandardScaler + GridSearch\" с весами:")

weights = pd.DataFrame({"feature": FEATURE_NAMES, "weight": MODEL.coef_.ravel()}).sort_values("weight", key=abs, ascending=False)
st.dataframe(weights.style.background_gradient(subset=["weight"]))
