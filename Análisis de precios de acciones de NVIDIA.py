
import numpy as np 
import pandas as pd 
from plotly import express
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose


DATA = 'nvidia_stock_2015_to_2024.csv'
print(DATA)

df = pd.read_csv(filepath_or_buffer=DATA, parse_dates=['date'])
df['year'] = df['date'].dt.year
print(df.head())

#Veamos primero el historial de precios
express.line(data_frame=df, x='date', y=['open', 'high', 'low', 'close'], log_y=True, template = "plotly_dark").show()
express.line(data_frame=df, x='date', y=['close', 'adjclose'], log_y=True, template = "plotly_dark").show()
express.line(data_frame=df, x='date', y='volume', log_y=True, template = "plotly_dark").show()

#Lowess (o loess)
#Suavizado de diagrama de dispersión ponderado localmente #"Loess significa suavizado de diagrama de dispersión estimado localmente 
# (lowess significa suavizado de diagrama de dispersión ponderado localmente) y es una de las muchas técnicas de regresión no 
# paramétrica, pero posiblemente la más flexible".
express.scatter(data_frame=df, x='date', y='volume', trendline='lowess', log_y=True, color='year', template = "plotly_dark").show()

#"Nuestra línea de tendencia de precios es muy suave incluso cuando utilizamos valores de cierre ajustados diariamente"
express.scatter(data_frame=df, x='date', y='adjclose', trendline='lowess', color='year', log_y=True, template = "plotly_dark").show()

#"Aquí hemos filtrado los valores atípicos de volumen y obtenemos casi una capa anual de precios de cierre por año, ya que el precio 
# de las acciones # de Zynex ha subido cada vez más incluso cuando el volumen de operaciones ha disminuido".
#Original era 2000000 Reduje los ceros para trazar algo
express.scatter(data_frame=df[df['volume'] > 20000], x='volume', y='adjclose', color='year', log_x=True, log_y=True, template = "plotly_dark").show()

df.isnull().sum()

df['date'] = pd.to_datetime(df['date'])
plt.figure(figsize=(14, 7))
plt.plot(df['adjclose'], label='Precio de cierre ajustado')
plt.title('Precio de cierre ajustado de NVIDIA a lo largo del tiempo\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Fecha\n')
plt.ylabel('Precio de cierre ajustado\n')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(df['volume'], label='Volumen', color='orange')
plt.title('Volumen de operaciones de NVIDIA a lo largo del tiempo\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Fecha\n')
plt.ylabel('Volumen\n')
plt.legend()
plt.grid(True)
plt.show()

df['MA50'] = df['adjclose'].rolling(window=50).mean()
df['MA200'] = df['adjclose'].rolling(window=200).mean()

plt.figure(figsize=(14, 7))
plt.plot(df['adjclose'], label='Precio de cierre ajustado', color= "red")
plt.plot(df['MA50'], label='Promedio móvil de 50 días', color = "mediumspringgreen")
plt.plot(df['MA200'], label='Promedio móvil de 200 días', color = "blue")
plt.title('Precio de cierre ajustado de NVIDIA con promedios móviles\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Fecha\n')
plt.ylabel('Precio de cierre ajustado\n')
plt.legend()
plt.grid(True)
plt.show()

df['Daily Return'] = df['adjclose'].pct_change()
plt.figure(figsize=(14, 7))
plt.plot(df['Daily Return'], label='Retorno diario')
plt.title('Rendimientos diarios de NVIDIA a lo largo del tiempo\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Fecha\n')
plt.ylabel('Retorno diario\n')
plt.legend()
plt.grid(True)
plt.show()

df['date'] = pd.to_datetime(df['date'])
fig, ax1 = plt.subplots(figsize=(14, 7))
color = 'tab:blue'
ax1.set_xlabel('Fecha\n')
ax1.set_ylabel('Precio de cierre ajustado\n', color=color)
ax1.plot(df['date'], df['adjclose'], color=color, label='Precio de cierre ajustado')
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Volume', color=color)
ax2.plot(df['date'], df['volume'], color=color, label='Volumen')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Precio de cierre ajustado y volumen de operaciones de NVIDIA a lo largo del tiempo\n', fontsize = '16', fontweight = 'bold')

fig.tight_layout()
fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)
ax1.grid(True)
plt.show()

X = df[['open', 'high', 'low', 'close', 'volume']]
y = df['adjclose']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
lr_model.fit(X_train, y_train)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
e_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
cat_model = CatBoostRegressor(iterations=1000, depth=10, learning_rate=0.1, loss_function='MAE', verbose=0)
cat_model.fit(X_train, y_train)

y_pred_cat = cat_model.predict(X_test)
mae_cat = mean_absolute_error(y_test, y_pred_cat)
mse_cat = mean_squared_error(y_test, y_pred_cat)
r2_cat = r2_score(y_test, y_pred_cat)

print("\nCatBoost - MAE:", mae_cat, "\nMSE:", mse_cat, "\nR2:", r2_cat)

plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred_lr, label='Predicho (Regresión lineal)')
plt.title('Real vs. Predicho (Regresión Lineal)\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Fecha\n')
plt.ylabel('Precio de cierre ajustado\n')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred_cat, label='Predicho (CatBoost)')
plt.title('Real vs. Predicho (CatBoost)\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Fecha\n')
plt.ylabel('Precio de cierre ajustado\n')
plt.legend()
plt.grid(True)
plt.show()
