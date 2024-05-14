import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from math import sqrt

def bidirectional_stepwise_regression(x, y):
    included = []
    while True:
        changed = False
        for feature in x.columns:
            if feature not in included:
                model = sm.OLS(y, sm.add_constant(x[included + [feature]])).fit()
                new_feature_pvalue = model.pvalues[feature]
                if new_feature_pvalue < 0.02:  
                    included.append(feature)
                    changed = True
        for feature in included:
            model = sm.OLS(y, sm.add_constant(x[included])).fit()
            max_pvalue = model.pvalues.idxmax()
            if max_pvalue != 'const' and model.pvalues[max_pvalue] > 0.02:
                included.remove(max_pvalue)
                changed = True
        if not changed:
            break
    x_selected = sm.add_constant(x[included])
    model_final = sm.OLS(y, x_selected).fit()
    r2 = r2_score(y, model_final.predict(x_selected))

    predictions = model_final.predict(x_selected)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    equation = f"y = {model_final.params['const']:.4f} + "
    for feature in included:
        coef = model_final.params[feature]
        equation += f"{coef:.4f} * {feature} + "
    equation = equation[:-3]

    return included, r2, mae, rmse, equation
filename = r'C:\Users\Lenovo\LY2_Ligand.csv'
data = pd.read_csv(filename)
x = data.iloc[:, 1:]
y = data['y']
selected_features, r2, mae, rmse, equation = bidirectional_stepwise_regression(x, y)

print("Selected Features:", selected_features)
print("R-squared:", r2)
print("MAE:", mae)
print("RMSE:", rmse)
print("Equation:", equation)
