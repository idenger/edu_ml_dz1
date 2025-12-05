import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from phik import phik_matrix
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="🚗 Домашнее задание 1",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Кастомный CSS для красивого оформления
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #grey;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    h1 {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    h2 {
        color: #2c3e50;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚗 Домашнее задание 1")

# Кэширование загрузки данных
@st.cache_data
def load_data():
    """Загрузка данных из URL"""
    df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')
    return df_train, df_test

@st.cache_data
def preprocess_features(df_train, df_test):
    """Предобработка признаков как в ноутбуке"""
    train = df_train.copy()
    test = df_test.copy()
    
    # Удаление дубликатов
    train = train.drop_duplicates(subset=train.columns.drop('selling_price'), keep='first', ignore_index=True).copy()
    
    # Обработка mileage, engine, max_power
    columns = ['mileage', 'engine', 'max_power']
    for column in columns:
        train[column] = train[column].str.extract(r"([\d\.]+)").astype(float)
        test[column] = test[column].str.extract(r"([\d\.]+)").astype(float)
    
    # Обработка torque
    def parser(text):
        if not isinstance(text, str):
            return None, None
        text = text.lower()
        numbers = [float(num) for num in re.findall(r"\d+\.?\d*", text.replace(",", ""))]
        if len(numbers) == 3:
            rpm = numbers[2]
            torque = numbers[0]
        elif len(numbers) == 2:
            i_torque = abs(text.find("kgm") * text.find("nm"))
            i_rpm = text.find("rpm")
            if i_rpm != -1 and i_rpm < i_torque:
                rpm = numbers[0]
                torque = numbers[1]
            else:
                rpm = numbers[1]
                torque = numbers[0]
        else:
            return None, None
        if not "nm" in text and "kgm" in text:
            torque = torque * 9.81
        return float(torque), int(rpm)
    
    train[['torque', 'max_torque_rpm']] = train['torque'].apply(lambda x: pd.Series(parser(x)))
    test[['torque', 'max_torque_rpm']] = test['torque'].apply(lambda x: pd.Series(parser(x)))
    
    # Заполнение пропусков медианами из train
    columns = ['mileage', 'engine', 'max_power', 'torque', 'seats', 'max_torque_rpm']
    medians = train[columns].median()
    train[columns] = train[columns].fillna(medians)
    test[columns] = test[columns].fillna(medians)
    
    # Приведение к int
    for column in ['engine', 'seats']:
        train[column] = train[column].astype(float).round().astype(int)
        test[column] = test[column].astype(float).round().astype(int)
    
    return train, test

# Загрузка данных
with st.spinner("Загрузка данных..."):
    df_train_raw, df_test_raw = load_data()
    df_train, df_test = preprocess_features(df_train_raw, df_test_raw)

# Боковая панель
st.sidebar.title("📊 Настройки")
show_raw = st.sidebar.checkbox("Показать сырые данные", False)

# ========== ОСНОВНАЯ ИНФОРМАЦИЯ ==========
st.header("📋 Основная информация")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Train строк", df_train.shape[0])
col2.metric("Train столбцов", df_train.shape[1])
col3.metric("Test строк", df_test.shape[0])
col4.metric("Test столбцов", df_test.shape[1])

if show_raw:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Train (первые 5 строк)")
        st.dataframe(df_train.head(5))
    with col2:
        st.subheader("Test (первые 5 строк)")
        st.dataframe(df_test.head(5))

# ========== ПРОПУСКИ И ДУБЛИКАТЫ ==========
st.header("🔍 Пропуски и дубликаты")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Пропуски в Train")
    missing_train = df_train_raw.isna().sum()
    missing_train = missing_train[missing_train > 0]
    if len(missing_train) > 0:
        st.bar_chart(missing_train)
        st.write(missing_train.to_frame('Количество'))
    else:
        st.success("Пропусков нет!")

with col2:
    st.subheader("Пропуски в Test")
    missing_test = df_test_raw.isna().sum()
    missing_test = missing_test[missing_test > 0]
    if len(missing_test) > 0:
        st.bar_chart(missing_test)
        st.write(missing_test.to_frame('Количество'))
    else:
        st.success("Пропусков нет!")

col1, col2 = st.columns(2)
col1.metric("Дубликаты Train", df_train_raw.duplicated().sum())
col2.metric("Дубликаты Test", df_test_raw.duplicated().sum())

# ========== РАСПРЕДЕЛЕНИЯ ЧИСЛОВЫХ ПРИЗНАКОВ ==========
st.header("📊 Распределения числовых признаков")

numeric_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'selling_price' in numeric_cols:
    numeric_cols.remove('selling_price')

selected_features = st.multiselect(
    "Выберите признаки для анализа:",
    numeric_cols,
    default=numeric_cols[:6] if len(numeric_cols) >= 6 else numeric_cols
)

if selected_features:
    # Гистограммы Train vs Test
    n_cols = min(3, len(selected_features))
    n_rows = (len(selected_features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(selected_features):
        if idx < len(axes):
            axes[idx].hist(df_train[col].dropna(), bins=50, alpha=0.6, label='Train', color='blue', edgecolor='black')
            axes[idx].hist(df_test[col].dropna(), bins=50, alpha=0.6, label='Test', color='orange', edgecolor='black')
            axes[idx].set_title(f'Распределение: {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Частота')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    # Скрыть лишние subplots
    for idx in range(len(selected_features), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

# ========== СВЯЗЬ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ==========
st.header("💰 Связь признаков с ценой (selling_price)")

if selected_features:
    n_cols = min(3, len(selected_features))
    n_rows = (len(selected_features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(selected_features):
        if idx < len(axes):
            axes[idx].scatter(df_train[col], df_train['selling_price'], alpha=0.3, s=10)
            axes[idx].set_title(f'{col} vs Цена')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Цена (selling_price)')
            axes[idx].grid(True, alpha=0.3)
    
    for idx in range(len(selected_features), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

# ========== BOXPLOTS ПО КАТЕГОРИЯМ ==========
st.header("📦 Boxplots цены по категориальным признакам")

categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()

if categorical_cols:
    n_cols = min(2, len(categorical_cols))
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(categorical_cols):
        if idx < len(axes):
            df_train.boxplot(column="selling_price", by=col, ax=axes[idx], showfliers=False, showmeans=True)
            axes[idx].set_title(f'Цена по {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Цена')
            axes[idx].grid(True, alpha=0.3)
    
    for idx in range(len(categorical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

# ========== КОРРЕЛЯЦИИ ==========
st.header("🔗 Матрицы корреляций")

corr_tabs = st.tabs(["Pearson", "Spearman", "Phik"])

with corr_tabs[0]:
    st.subheader("Корреляция Пирсона (линейная)")
    pearson = df_train.corr(method="pearson", numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pearson, annot=True, cmap="bwr", vmin=-1, vmax=1, ax=ax, fmt='.2f', square=True)
    plt.title("Корреляция Пирсона (Train)", fontsize=14, pad=20)
    st.pyplot(fig)
    
    # Топ корреляций с целевой переменной
    if 'selling_price' in pearson.columns:
        target_corr = pearson['selling_price'].abs().sort_values(ascending=False)
        target_corr = target_corr[target_corr.index != 'selling_price']
        st.subheader("Топ корреляций с ценой")
        fig, ax = plt.subplots(figsize=(10, 6))
        target_corr.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Абсолютная корреляция')
        ax.set_title('Корреляция признаков с selling_price')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

with corr_tabs[1]:
    st.subheader("Корреляция Спирмена (ранговая)")
    spearman = df_train.corr(method="spearman", numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(spearman, annot=True, cmap="bwr", vmin=-1, vmax=1, ax=ax, fmt='.2f', square=True)
    plt.title("Корреляция Спирмена (Train)", fontsize=14, pad=20)
    st.pyplot(fig)

with corr_tabs[2]:
    st.subheader("Корреляция Phik (любые зависимости)")
    try:
        df_phik = df_train.drop(columns=['name'] if 'name' in df_train.columns else [])
        phik_corr = df_phik.phik_matrix()
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(phik_corr, annot=True, ax=ax, fmt='.2f', square=True, cmap='viridis')
        plt.title("Корреляция Phik (Train)", fontsize=14, pad=20)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Ошибка при вычислении Phik: {e}")

# ========== PAIRPLOT ==========
st.header("🔀 Pairplot (попарные распределения)")

pairplot_features = st.multiselect(
    "Выберите признаки для pairplot (рекомендуется 3-5):",
    numeric_cols,
    default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols[:len(numeric_cols)]
)

if len(pairplot_features) > 0:
    dataset_choice = st.radio("Выберите датасет:", ["Train", "Test"], horizontal=True)
    df = df_train if dataset_choice == "Train" else df_test
    
    fig = sns.pairplot(df[pairplot_features + ['selling_price']], diag_kind='kde', plot_kws={'alpha': 0.6, 's': 10})
    st.pyplot(fig)

# ========== СТАТИСТИКИ ==========
st.header("📈 Описательные статистики")

stat_tabs = st.tabs(["Train", "Test", "Сравнение"])

with stat_tabs[0]:
    st.subheader("Train данные")
    numeric_cols_stat = df_train.select_dtypes(include=['int64', 'float64']).columns
    st.dataframe(df_train[numeric_cols_stat].describe(), use_container_width=True)
    
    if len(df_train.select_dtypes(include=['object']).columns) > 0:
        st.subheader("Категориальные признаки")
        st.dataframe(df_train.select_dtypes(include=['object']).describe().T, use_container_width=True)

with stat_tabs[1]:
    st.subheader("Test данные")
    numeric_cols_stat = df_test.select_dtypes(include=['int64', 'float64']).columns
    st.dataframe(df_test[numeric_cols_stat].describe(), use_container_width=True)
    
    if len(df_test.select_dtypes(include=['object']).columns) > 0:
        st.subheader("Категориальные признаки")
        st.dataframe(df_test.select_dtypes(include=['object']).describe().T, use_container_width=True)

with stat_tabs[2]:
    st.subheader("Сравнение Train и Test")
    comparison_cols = ['year', 'selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    comparison_data = []
    for col in comparison_cols:
        if col in df_train.columns and col in df_test.columns:
            comparison_data.append({
                'Признак': col,
                'Train Mean': round(df_train[col].mean(), 2),
                'Train Median': round(df_train[col].median(), 2),
                'Test Mean': round(df_test[col].mean(), 2),
                'Test Median': round(df_test[col].median(), 2),
                'Разница Mean': round(abs(df_train[col].mean() - df_test[col].mean()), 2)
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Визуализация сравнения
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        key_cols = ['year', 'selling_price', 'km_driven', 'max_power']
        
        for idx, col in enumerate(key_cols):
            if idx < 4 and col in df_train.columns:
                ax = axes[idx // 2, idx % 2]
                ax.hist(df_train[col].dropna(), bins=50, alpha=0.6, label='Train', color='blue', edgecolor='black')
                ax.hist(df_test[col].dropna(), bins=50, alpha=0.6, label='Test', color='orange', edgecolor='black')
                ax.set_title(f'Распределение: {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Частота')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

