import pandas as pd
import numpy as np

def outliers_iqr_mod(data: pd.DataFrame, feature: str, left: float = 1.5, right: float = 1.5):
    """
    Поиск выбросов по методу Тьюки с предварительной очисткой нечисловых значений.
    
    Args:
        data (pd.DataFrame): Исходный датафрейм.
        feature (str): Наименование признака.
        left (float): Коэффициент для нижней границы (по умолчанию 1.5).
        right (float): Коэффициент для верхней границы (по умолчанию 1.5).
    
    Returns:
        outliers (pd.DataFrame): Строки с выбросами.
        cleaned (pd.DataFrame): Очищенный датафрейм (без выбросов).
        lower_bound (float): Нижняя граница.
        upper_bound (float): Верхняя граница.
    """
    # Копируем, чтобы не менять оригинал
    x = data[feature].copy()
    
    # Преобразуем в числовой тип; всё, что не число (включая "unknown"), станет NaN
    x_num = pd.to_numeric(x, errors='coerce')
    
    # Если после конвертации нет ни одного валидного числа — нельзя искать выбросы
    if x_num.isna().all():
        raise ValueError(f"В колонке '{feature}' нет валидных числовых значений для поиска выбросов.")
    
    quartile_1 = x_num.quantile(0.25)
    quartile_3 = x_num.quantile(0.75)
    iqr = quartile_3 - quartile_1
    
    lower_bound = quartile_1 - (iqr * left)
    upper_bound = quartile_3 + (iqr * right)
    
    # Маски: работаем с числовым рядом x_num, но фильтруем исходный data
    mask_outlier = (x_num <= lower_bound) | (x_num >= upper_bound)
    mask_clean = (x_num > lower_bound) & (x_num < upper_bound)  # или <=, если хочешь оставить границы
    
    outliers = data[mask_outlier]
    cleaned = data[mask_clean]
    
    return outliers, cleaned, lower_bound, upper_bound


def create_marketing_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Поведенческие флаги
    data['has_previous_contact'] = (data['previous'] > 0).astype(int)
    data['has_success_previous'] = (data['poutcome'] == 'success').astype(int)
    data['never_contacted_before'] = (data['pdays'] == 999).astype(int)
    data['recent_contact'] = ((data['pdays'] >= 0) & (data['pdays'] <= 7)).astype(int)

    median_duration = data['duration'].median()
    data['long_call'] = (data['duration'] > median_duration).astype(int)

    # Финансовые признаки
    q90 = data['balance'].quantile(0.90)
    q10 = data['balance'].quantile(0.10)
    data['is_high_balance'] = (data['balance'] >= q90).astype(int)
    data['is_low_balance'] = (data['balance'] <= q10).astype(int)

    # ИСПРАВЛЕНИЕ: безопасный логарифм для отрицательных балансов
    # Формула: sign(x) * log1p(|x|)
    abs_bal = np.abs(data['balance'])
    sign_bal = np.sign(data['balance'])
    data['balance_log'] = sign_bal * np.log1p(abs_bal)

    # Демография и группы
    data['age_group'] = pd.cut(data['age'], bins=[0, 30, 45, 60, 100],
                               labels=['young', 'mid_young', 'mid_old', 'senior'])
    data['job_marital'] = data['job'].astype(str) + '_' + data['marital'].astype(str)
    data['risk_profile_flag'] = ((data['default'] == 'yes') | (data['loan'] == 'yes')).astype(int)
    
    # Флаг наличия просрочки (default)
    data['has_default'] = (data['default'] == 'yes').astype(int)
    
    # Флаг наличия кредита на жилье (housing)
    data['has_housing'] = (data['housing'] == 'yes').astype(int)
    
    # Флаг наличия кредита на личные нужды (loan)
    data['has_loan'] = (data['loan'] == 'yes').astype(int)

    # Временные признаки
    data['is_end_of_month'] = (data['day'] >= 25).astype(int)
    hot_months = [3, 6, 12]
    data['is_hot_month'] = data['month'].isin(hot_months).astype(int)
    data['is_peak_deposit_month'] = data['month'].isin(['dec', 'mar', 'sep']).astype(int)
    data['is_low_deposit_month'] = data['month'].isin(['may', 'jan']).astype(int)

    # Признаки по кампании
    median_campaign = data['campaign'].median()
    data['many_contacts'] = (data['campaign'] > median_campaign).astype(int)
    data['first_contact_campaign'] = (data['campaign'] == 1).astype(int)

    return data