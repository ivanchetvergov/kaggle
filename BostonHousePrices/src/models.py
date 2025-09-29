from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import numpy as np

def train_and_evaluate_linear_regression(X_train_scaled, X_test_scaled, y_train, y_test):

    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred = lr_model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    return {
        'model': lr_model,
        'R2': r2,
        'RMSE': rmse,
        'Coefficients': lr_model.coef_
    }


def train_and_evaluate_random_forest(X_train_unscaled, X_test_unscaled, y_train, y_test, n_estimators=100):
    
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train_unscaled, y_train)
    y_pred = rf_model.predict(X_test_unscaled)
    
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    return {
        'model': rf_model,
        'R2': r2,
        'RMSE': rmse,
        'Feature_Importances': rf_model.feature_importances_
    }