# 📈Análisis y Predicción del Precio de las Acciones de NVIDIA (NVDA)

Este proyecto realiza un análisis exploratorio, técnico y predictivo del precio de las acciones de NVIDIA (NVDA) utilizando datos históricos entre 2015 y 2024.
Se combinan técnicas de visualización, series temporales y modelos de Machine Learning para estudiar el comportamiento del precio y evaluar distintos enfoques de predicción.

# 🎯Objetivos del proyecto

- Analizar la evolución histórica del precio y volumen de NVIDIA.
- Identificar tendencias mediante medias móviles y suavizado LOWESS.
- Estudiar retornos diarios y relación precio–volumen.
- Construir y comparar modelos de Machine Learning para predecir el precio de cierre ajustado.
- Evaluar el desempeño de los modelos con métricas estándar.

# 📁Descripción de los datos

El dataset contiene información bursátil diaria de NVIDIA:

- date: fecha de negociación
- open: precio de apertura
- high: precio máximo
- low: precio mínimo
- close: precio de cierre
- adjclose: precio de cierre ajustado
- volume: volumen negociado

Se agregan variables derivadas como:
- Año
- Retornos diarios
- Medias móviles (50 y 200 días)

# 📊Análisis exploratorio (EDA)

- Visualizaciones interactivas (Plotly):
  -- Evolución temporal de precios (open, high, low, close).
  -- Comparación entre close vs adjclose en escala logarítmica.
  -- Análisis de volumen.
  -- Suavizado de tendencias mediante LOWESS.
  -- Relación volumen–precio con escalas log–log.
  -- Visualizaciones estáticas (Matplotlib / Seaborn)
  -- Precio de cierre ajustado a lo largo del tiempo.
  -- Volumen de operaciones.
  -- Medias móviles de 50 y 200 días.
  -- Retornos diarios.
  -- Gráfico combinado precio–volumen con doble eje.

# 🤖Modelos de Machine Learning

Se entrenan y comparan distintos modelos supervisados para predecir el precio de cierre ajustado:
- Regresión Lineal
- Decision Tree Regressor
- Random Forest Regressor
- CatBoost Regressor
- Métricas de evaluación
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- R² Score
- Se comparan valores reales vs. predichos mediante gráficos temporales.

# 🛠️Tecnologías utilizadas

- Python
- pandas / numpy
- Plotly
- Matplotlib / `Seaborn`
- scikit-learn
- `CatBoost`
- TensorFlow / Keras (preparado para modelos secuenciales)
- statsmodels


# 📂Estructura del proyecto

├── Análisis de precios de acciones de NVIDIA.py
├── nvidia_stock_2015_to_2024.csv
└── README.md


# 📌Resultados principales

- Fuerte tendencia alcista de largo plazo en NVIDIA.
- Alta relación entre crecimiento del precio y cambios estructurales en el volumen.
- Las medias móviles capturan claramente cambios de tendencia.
- Modelos no lineales (especialmente CatBoost) muestran mejor desempeño que la regresión lineal.
- Buen potencial del enfoque ML para modelado financiero exploratorio.

# ⚠️Disclaimer

Este proyecto tiene fines educativos y analíticos.
No constituye asesoramiento financiero ni recomendaciones de inversión.

👤 Autor

Flavia Hepp
Data Science en formación· Machine Learning · Análisis Financiero
