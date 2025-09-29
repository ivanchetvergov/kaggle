import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    rm_median = df_copy['rm'].median()
    df_copy.fillna({'rm' : rm_median}, inplace=True)

    df_copy['log_crim'] = np.log(df_copy['crim'] + 1)
    df_copy['log_dis'] = np.log(df_copy['dis'])

    df_copy['lstat_SQ'] = df_copy['lstat'] ** 2

    def categorize_rad(rad):
        if rad <= 5:
            return 'low'
        elif rad < 10:
            return 'medium'
        else: # rad >= 10
            return 'high'

    df_copy['rad_gr'] = df_copy['rad'].apply(categorize_rad)

    df_copy = pd.get_dummies(df_copy, columns=['rad_gr'], prefix='rad', drop_first=True)

    df_copy.drop(['crim', 'dis', 'lstat', 'rad'], axis=1, inplace=True)
    
    return df_copy


def split_scale_data(df_processed: pd.DataFrame, target_col: str = 'medv', test_size: float = 0.3, random_state: int = 42):
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train_cols = X_train.columns

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest будет использовать немасштабированные данные, поэтому возвращаем оба варианта
    return X_train_scaled, X_test_scaled, X_train.values, X_test.values, y_train, y_test, X_train_cols