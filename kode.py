# Import package
import os
import warnings
warnings.filterwarnings('ignore')

import xgboost  as xgb
import lightgbm as lgb
import catboost as cat

from sklearn.feature_selection     import RFE, SelectKBest
from sklearn.model_selection       import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.neural_network        import MLPRegressor
from sklearn.preprocessing         import MinMaxScaler, StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder, RobustScaler
from sklearn.linear_model          import LinearRegression
from sklearn.neighbors             import KNeighborsRegressor
from sklearn.ensemble              import RandomForestRegressor
from sklearn.metrics               import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree                  import DecisionTreeRegressor
from sklearn.svm                   import SVR
from scipy.stats                   import randint
from catboost                      import CatBoostRegressor
from pyforest                      import *

from imblearn.over_sampling import RandomOverSampler

from smogn import smoter

from google.colab import drive
drive.mount('/content/drive')

# Data Understanding
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/Kode/ML/DBS-Expert/garments_worker_productivity.csv')

# missing value
df.isnull().sum()
df['wip'].fillna(df['wip'].mean(), inplace=True)

df2 = df.drop(['date'], axis=1)

# Outlier
objek_vars           = []
numerik_vars         = []

for col in df2.columns:

    if df2[col].dtype == 'object':
        objek_vars.append(col)

    else:
        numerik_vars.append(col)

df3 = df2.copy()

num                = df3[numerik_vars]
n                  = len(num.columns)
rows               = n // 2
cols               = 2
sns.set(font_scale = 1)
fig, ax            = plt.subplots(rows, cols, figsize=(30, 45))

for i in range(rows):

    for j in range(cols):

        index      = i * cols + j
        if index   < n:
            col    = num.columns[index]
            sns.boxplot(ax=ax[i, j],
                        data=num,
                        y=num[col],
                        width=0.50)

            ax[i, j].set_title(f'Kolom {col}', fontdict={'fontsize': 20})

# handle outlier
def handle_outlier(df, variabel):
    Q1          = df[variabel].quantile(0.25)
    Q3          = df[variabel].quantile(0.75)
    IQR         = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df.loc[df[variabel] > upper_limit, variabel] = upper_limit
    df.loc[df[variabel] < lower_limit, variabel] = lower_limit

    return df

variabel_outlir = ['targeted_productivity',
                   'incentive',
                   'wip',
                   'over_time']

for var in variabel_outlir:
    df3  = handle_outlier(df3, var)

# EDA: Analisis Univariate
numerik_vars2 = [
                 'targeted_productivity',
                 'smv',
                 'wip',
                 'over_time',
                 'incentive',
                 'idle_time',
                 'idle_men',
                 'no_of_style_change',
                 'no_of_workers']

def scatter_plot(df, numerik_vars, target_var):
    num               = df[numerik_vars]
    n                 = len(num.columns)
    rows              = n // 2 + n % 2
    cols              = 2
    sns.set(font_scale=1)
    fig, ax           = plt.subplots(rows, cols, figsize=(20, rows * 5))

    for i in range(rows):

        for j in range(cols):
            index     = i * cols + j

            if index  < n:

                col   = num.columns[index]
                sns.scatterplot(ax=ax[i, j],
                                data=df,
                                x=col,
                                y=target_var)

                ax[i, j].set_title(f'Hubungan antara {col} terhadap actual_productivity', fontdict={'fontsize': 15})
                ax[i, j].set_xlabel(col)
                ax[i, j].set_ylabel('actual_productivity')

    plt.tight_layout()
    plt.show()

scatter_plot(df3, numerik_vars2, 'actual_productivity')

def bar_plot(df, kategorik_vars, target_var):
    cat          = df[kategorik_vars]
    n            = len(cat.columns)
    rows         = n // 2 + n % 2
    cols         = 2
    sns.set(style="whitegrid", font_scale=1)
    fig, ax      = plt.subplots(rows, cols, figsize=(20, rows * 5))

    for i in range(rows):

        for j in range(cols):

            index    = i * cols + j

            if index < n:

                col  = cat.columns[index]
                order = df.groupby(col)[target_var].mean().sort_values().index # rata-rata

                unique_values = df[col].nunique()
                palette = sns.color_palette("Blues", unique_values)

                sns.barplot(ax=ax[i, j],
                            data=df,
                            x=col,
                            y=target_var,
                            order=order,
                            palette=palette)

                ax[i, j].set_title(f'Hubungan antara {col} terhadap actual_productivity', fontdict={'fontsize': 15})
                ax[i, j].set_xlabel(col)
                ax[i, j].set_ylabel('actual_productivity')

    plt.tight_layout()
    plt.show()

bar_plot(df3, objek_vars, 'actual_productivity')

# 'Day'
one_hot_day                 = OneHotEncoder(sparse_output=False)
day_reshaped                = df3['day'].values.reshape(-1, 1)
day_one_hot                 = one_hot_day.fit_transform(day_reshaped)
day_encoded                 = pd.DataFrame(day_one_hot, columns=one_hot_day.get_feature_names_out(['day']))
df3_day                     = pd.concat([df3.drop(columns=['day']), day_encoded], axis=1)

# 'Quarter'
ordinal                     = OrdinalEncoder()
df3_day['quarter_encod']    = ordinal.fit_transform(df3_day[['quarter']])

# 'Department'
one_hot_department          = OneHotEncoder(sparse_output=False)
department_reshaped         = df3_day['department'].values.reshape(-1, 1)
department_one_hot          = one_hot_department.fit_transform(department_reshaped)
department_encoded          = pd.DataFrame(department_one_hot, columns=one_hot_department.get_feature_names_out(['department']))
df3_day                     = pd.concat([df3_day.drop(columns=['department']), department_encoded], axis=1)

df3_day.drop(columns=['quarter'], inplace=True)

distribusi = ['smv', 'wip', 'over_time', 'incentive', 'no_of_workers', 'idle_time']

rows    = (len(distribusi) + 1) // 2
cols    = 2
fig, ax = plt.subplots(rows,
                       cols,
                       figsize=(15, 6 * rows))

for i, var in enumerate(distribusi):

    row = i // cols
    col = i % cols
    sns.histplot(df3[var], ax=ax[row, col], kde=True, bins=20)
    ax[row, col].set_title(f'Distribusi {var}', fontsize=14)
    ax[row, col].set_xlabel('')
    ax[row, col].set_ylabel('Jumlah', fontsize=12)

plt.tight_layout()
plt.show()

X = df3_day.drop(['actual_productivity'], axis=1)
y = df3_day.actual_productivity

smv_scaler               = MinMaxScaler()
wip_scaler               = RobustScaler()
over_time_scaler         = MinMaxScaler()
incentive_scaler         = np.log1p
no_of_workers_scaler     = MinMaxScaler()

X_skala                  = X.copy()
X_skala['smv']           = smv_scaler.fit_transform(X[['smv']])
X_skala['wip']           = wip_scaler.fit_transform(X[['wip']])
X_skala['over_time']     = over_time_scaler.fit_transform(X[['over_time']])
X_skala['incentive']     = incentive_scaler(X['incentive'])
X_skala['idle_time']     = incentive_scaler(X['idle_time'])
X_skala['no_of_workers'] = no_of_workers_scaler.fit_transform(X[['no_of_workers']])

df_3 = pd.concat([X_skala, y], axis=1)

X = X_skala.copy()
df = df_3.copy()
y = y.copy()

# Feature Selection
# viz
sns.set(font_scale=1)
plt.figure(figsize=(20, 20))
sns.set_style("white")
cmap = sns.light_palette("green", as_cmap=True)

mask = np.triu(np.ones_like(df.corr(), dtype=bool))

sns.heatmap(df.corr(),
            annot=True,
            cmap=cmap,
            fmt='.2f',
            mask=mask)

plt.show()

model_fitur_linear = LinearRegression()
n_fitur_pilih      = 13

rfe                = RFE(estimator=model_fitur_linear,
                         n_features_to_select=n_fitur_pilih)

rfe.fit(X, y)

fitur_pilih = X.columns[rfe.support_]
fitur_rank = pd.Series(rfe.ranking_, index=X.columns)

X_baru = X[fitur_pilih]

# Visualisasi bar plot untuk menampilkan jumlah kasus imbalanced
threshold              = 0.5
cases_kurang_produktif = df[df['actual_productivity'] < threshold].shape[0]
cases_produktif        = df[df['actual_productivity'] >= threshold].shape[0]

df_2 = pd.concat([X_baru, y], axis=1)

over_df = smoter(df_2, y='actual_productivity')
X_resampled = over_df.drop('actual_productivity', axis=1)

X_resampled.rename(columns={X_resampled.columns[11]: 'department_finishing2'}, inplace=True)

y_resampled = over_df['actual_productivity']


threshold = 0.5
cases_kurang_produktif = (y_resampled < threshold).sum()
cases_produktif = (y_resampled >= threshold).sum()

# Modeling
X_train, X_test, y_train, y_test = train_test_split(X_resampled,
                                                    y_resampled,
                                                    test_size=0.25,
                                                    random_state=42)

def regression_model_report(y_true, y_pred):
    y_true = np.ravel(y_true)  # array 1D
    y_pred = np.ravel(y_pred)  # array 1D

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Mencari smape dengan rumus
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

    print("MAE: {:.4f}".format(mae))  # Mean Absolute Error
    print("MSE: {:.4f}".format(mse))  # Mean Squared Error
    print("RMSE: {:.4f}".format(rmse))  # Root Mean Squared Error
    print("sMAPE: {:.4f}%".format(smape))  # Symmetric Mean Absolute Percentage Error

linear_regression       = LinearRegression()
knn_regressor           = KNeighborsRegressor(n_neighbors=30, algorithm="auto")  # 'auto' mencari algoritma terbaiknya
decision_tree_regressor = DecisionTreeRegressor(max_depth=10, criterion="squared_error")
random_forest_regressor = RandomForestRegressor(max_depth=15, criterion="squared_error", n_estimators=38)
svm_rbf_regressor       = SVR(kernel='rbf')
mlp_regressor           = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)
xgboost_regressor       = xgb.XGBRegressor()
lightgbm_regressor      = lgb.LGBMRegressor()
catboost_regressor      = CatBoostRegressor(verbose=0)

models = {
    "Linear Regression"       : linear_regression,
    "KNN Regression"          : knn_regressor,
    "Decision Tree Regression": decision_tree_regressor,
    "Random Forest Regression": random_forest_regressor,
    "SVM RBF Regression"      : svm_rbf_regressor,
    "MLP Regression"          : mlp_regressor,
    "XGBoost Regression"      : xgboost_regressor,
    "LightGBM Regression"     : lightgbm_regressor,
    "CatBoost Regression"     : catboost_regressor
}

# Train
print("*" * 46, "Performa Model ML pada Data Train", "*" * 46, "\n")

for nama_model, model in models.items():

    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)

    print(f"{nama_model} Train: ")
    regression_model_report(y_train, y_pred_train)
    print("=" * 25 + "\n")

# Test
print("*"*50, "Performa Testing Model ML", "*"*50, "\n")

for nama_model, model in models.items():

    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    print(f"{nama_model} Test: ")
    regression_model_report(y_test, y_pred_test)

    print("="*25 + "\n")

# fungsi untuk evaluasi model prediksi dan visualisasi residual
def regression_model_report_viz(y_true, y_pred):
    y_true = np.ravel(y_true)  # array 1D
    y_pred = np.ravel(y_pred)  # array 1D

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Mencari smape dengan rumus
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

    print("MAE: {:.4f}".format(mae))  # Mean Absolute Error
    print("MSE: {:.4f}".format(mse))  # Mean Squared Error
    print("RMSE: {:.4f}".format(rmse))  # Root Mean Squared Error
    print("sMAPE: {:.4f}%".format(smape))  # Symmetric Mean Absolute Percentage Error

    # Visualisasi Scatter Plot Actual vs. Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, c='blue', label='Prediksi', alpha=0.5)
    plt.scatter(y_true, y_true, c='green', label='Aktual', alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Ideal')
    plt.xlabel('Nilai Aktual')
    plt.ylabel('Nilai Prediksi')
    plt.title('Scatter Plot Aktual vs. Prediksi')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Residuals
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, c='purple', label='Residuals', alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Nilai Prediksi')
    plt.ylabel('Residual')
    plt.title('Residuals Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

# Hyperparameter Tuning

# Cat Boost
cat_cv = CatBoostRegressor()

parameter_cat = {
    'learning_rate': np.linspace(0.01, 0.3),
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'iterations': randint(50, 200)
}

grid_cv_cat = RandomizedSearchCV(cat_cv,
                                 param_distributions=parameter_cat,
                                 n_iter=50,
                                 scoring='neg_mean_squared_error',
                                 cv=5,
                                 verbose=0)

grid_cv_cat.fit(X_train, y_train)
grid_cv_cat.best_params_
cat_best = CatBoostRegressor(learning_rate=0.09285714285714285,
                             l2_leaf_reg=1,
                             depth=6,
                             iterations=190)

cat_best.fit(X_train, y_train)
y_tt_cat_best = cat_best.predict(X_test)
regression_model_report_viz(y_test, y_tt_cat_best)

# Random Forest
rf_cv = RandomForestRegressor()

parameter_rf = {
    'n_estimators': randint(50, 200),
    'max_depth': [0, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

grid_cv_rf = RandomizedSearchCV(rf_cv,
                                param_distributions=parameter_rf,
                                n_iter=50,
                                scoring='neg_mean_squared_error',
                                cv=5,
                                verbose=0)

grid_cv_rf.fit(X_train, y_train)

grid_cv_rf.best_params_

rf_best = RandomForestRegressor(max_depth=50,
                                max_features='sqrt',
                                min_samples_leaf=1,
                                min_samples_split=5,
                                n_estimators=191)

rf_best.fit(X_train, y_train)

y_tt_rf_best = rf_best.predict(X_test)
regression_model_report_viz(y_test, y_tt_rf_best)

xgb_cv = xgb.XGBRegressor()

parameter_xgb = {
    'learning_rate': np.linspace(0.01, 0.3, 30),
    'max_depth': [4, 6, 8, 10],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3],
    'colsample_bytree': np.linspace(0.3, 0.7, 5),
    'n_estimators': randint(50, 200)
}

grid_cv_xgb = RandomizedSearchCV(xgb_cv,
                                 param_distributions=parameter_xgb,
                                 n_iter=50,
                                 scoring='neg_mean_squared_error',
                                 cv=5,
                                 verbose=0)

grid_cv_xgb.fit(X_train, y_train)

grid_cv_xgb.best_params_
xgb_best = xgb.XGBRegressor(colsample_bytree=0.5,
                            gamma=0,
                            learning_rate=0.89,
                            max_depth=4,
                            min_child_weight=3,
                            n_estimators=191)

xgb_best.fit(X_train, y_train)
y_tt_xgb_best = xgb_best.predict(X_test)
regression_model_report_viz(y_test, y_tt_xgb_best)

# Analsis insight
X_test_with_predictions = X_test.copy()

# Tambahkan hasil prediksi sebagai kolom baru
X_test_with_predictions['predicted_productivity'] = y_tt_rf_best

# Membuat visualisasi korelasi
sns.set(font_scale=1)
plt.figure(figsize=(20, 20))
sns.set_style("white")
cmap = sns.light_palette("green", as_cmap=True)

# Hitung korelasi
corr = X_test_with_predictions.corr()

# Buat mask untuk segitiga atas
mask = np.triu(np.ones_like(corr, dtype=bool))

# Buat heatmap
sns.heatmap(corr, annot=True, cmap=cmap, fmt='.2f', mask=mask)

plt.show()

def scatter_plot(df, numerik_vars, target_var):
    num = df[numerik_vars]
    n = len(num.columns)
    rows = n // 2 + n % 2
    cols = 2
    sns.set(font_scale=1)
    fig, ax = plt.subplots(rows, cols, figsize=(20, rows * 5))

    for i in range(rows):
        for j in range(cols):
            index = i * cols + j

            if index < n:
                col = num.columns[index]
                sns.scatterplot(ax=ax[i, j], data=df, x=col, y=target_var)

                ax[i, j].set_title(f'Hubungan antara {col} terhadap {target_var}', fontdict={'fontsize': 15})
                ax[i, j].set_xlabel(col)
                ax[i, j].set_ylabel(target_var)

    plt.tight_layout()
    plt.show()

# List of numerical variables
numerical_vars = X_test.columns

scatter_plot(X_test_with_predictions, numerical_vars, 'predicted_productivity')

# analisi kategorik
categorical_vars = ['team', 'no_of_style_change', 'department_sweing', 'department_finishing']

# Definisikan fungsi untuk membuat bar plot
def bar_plot(df, kategorik_vars, target_var):
    cat = df[kategorik_vars]
    n = len(cat.columns)
    rows = n // 2 + n % 2
    cols = 2
    sns.set(style="whitegrid", font_scale=1)
    fig, ax = plt.subplots(rows, cols, figsize=(20, rows * 5))

    for i in range(rows):
        for j in range(cols):
            index = i * cols + j

            if index < n:
                col = cat.columns[index]
                order = df.groupby(col)[target_var].mean().sort_values().index  # rata-rata

                unique_values = df[col].nunique()
                palette = sns.color_palette("Blues", unique_values)

                sns.barplot(ax=ax[i, j],
                            data=df,
                            x=col,
                            y=target_var,
                            order=order,
                            palette=palette)

                ax[i, j].set_title(f'Hubungan antara {col} terhadap {target_var}', fontdict={'fontsize': 15})
                ax[i, j].set_xlabel(col)
                ax[i, j].set_ylabel(target_var)

    plt.tight_layout()
    plt.show()

bar_plot(X_test_with_predictions, categorical_vars, 'predicted_productivity')


# Save Model
path2 = '/content/drive/MyDrive/Kode/ML/DBS-Expert/RandomForest_Productivity_4.pkl'

with open(path2, 'wb') as file:
    pickle.dump(rf_best, file)