#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Funciones Generales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns

# Funciones de Calendario y Tiempo
import holidays
import time
from datetime import datetime
from pandas.tseries.offsets import DateOffset
from dateutil.relativedelta import relativedelta

# Funciones de Modelos de Series de Tiempo
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Funciones de Scikit Learn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# Funciones de Feature Engine
from feature_engine.creation import CyclicalFeatures
from feature_engine.datetime import DatetimeFeatures
from feature_engine.imputation import DropMissingData
from feature_engine.selection import DropFeatures
from feature_engine.timeseries.forecasting import (
     LagFeatures,WindowFeatures)

# Algoritmos de Machine Learning
import xgboost as xgb 
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Otras Funciones
from tqdm import tqdm #Barras de Progreso
import warnings # Eliminar de advertencias
import sys 
from IPython.display import clear_output # Complemento barra de progreso
import math

from scipy.stats import norm
import streamlit as st

# Funcion para exportar a excel
import io


# ## 1. Cargar Datos Historicos

# In[2]:


# Función de carga y listado de hojas
def cargar_data_nv(ruta):
    # Leer el archivo y listar las hojas
    excel_file = pd.ExcelFile(ruta)
    print("Hojas disponibles en el archivo:")
    for hoja in excel_file.sheet_names:
        print(f"- {hoja}")    
    # Cargar la hoja seleccionada
    df = excel_file.parse('Historial campañas Novaventa')
    return df


# In[3]:


def preprocesar_datos_1_nv(df):
    df['Periodo_Campaña'] = df['Año'].astype(str) + '-' + df['Campaña'].astype(str).str.zfill(2)
    df_orig = df[['Periodo_Campaña','Referencia Novaventa',	'Unds Brutas']].copy()
    df_orig = df_orig.rename(columns={'Referencia Novaventa':'CODIGO','Unds Brutas':'DEMANDA','Periodo_Campaña':'FECHA'})
    
    return df_orig


# In[4]:


def llenar_nan(df_orig):
    df_horiz = df_orig.pivot_table(index='CODIGO', columns='FECHA', values='DEMANDA', fill_value=0, observed=True)
    # Reset the index of the pivot table so 'CODIGO' becomes a column again
    df_reset = df_horiz.reset_index()
        # Melt the DataFrame to convert it back to a vertical format
    df_vertical = df_reset.melt(id_vars=['CODIGO'], var_name='FECHA', value_name='DEMANDA')
    df_vertical['CODIGO'] = df_vertical['CODIGO'].astype('str')
    return df_vertical


# In[5]:


def eliminar_ceros_iniciales_nv(df):
    # Lista para almacenar DataFrames válidos
    lista_df = []

    # Obtener códigos únicos (SKU-cliente)
    codigos_unicos = df['CODIGO'].unique()
    #clientes_unicos = df['CLIENTE'].unique()
    
    for codigo in codigos_unicos:
       
        # Filtrar los datos para cada código y cliente
        df_codigo = df[df['CODIGO'] == codigo]
            # Verificar si el DataFrame no está vacío
        if not df_codigo.empty:
            # Verificar si hay valores no cero en DEMANDA
            if df_codigo['DEMANDA'].ne(0).any():
                # Encontrar la primera fila donde la demanda no es cero
                indice_primer_no_cero = df_codigo['DEMANDA'].ne(0).idxmax()

                # Recortar la serie temporal desde el primer valor no cero
                df_codigo_recortado = df_codigo.loc[indice_primer_no_cero:]
                
                # Agregar a la lista si no está vacío
                if not df_codigo_recortado.empty:
                    lista_df.append(df_codigo_recortado)

    # Concatenar todos los DataFrames válidos de una vez
    if lista_df:
        df_resultado = pd.concat(lista_df)
    else:
        df_resultado = pd.DataFrame(columns=df.columns)

    return df_resultado


# In[6]:


def reemplazar_ceros_nv(df_mes_orig):
    
    # Generar copia de trabajo
    df_mes_ceros = df_mes_orig.copy()
    # Reemplazar los valores de demanda iguales a 0 por la mediana correspondiente
    # df_mes_ceros['DEMANDA'] = df_mes_orig.groupby('CODIGO')['DEMANDA'].transform(
    #     lambda x: x.replace(0, x.median())
    #     )
    df_mes_ceros['DEMANDA'] = df_mes_orig.groupby('CODIGO')['DEMANDA'].transform(
        lambda x: x.where(x >= 70, x.median())
    )
    df_mes_ceros['CONSECUTIVO'] = df_mes_ceros.groupby('CODIGO').cumcount() + 1
    df_mes_ceros = df_mes_ceros.set_index('FECHA')
    
    return df_mes_ceros


# In[7]:


def graficar_demanda_codigo_nv(df_mes_orig):
    # Obtener los códigos únicos
    codigos_unicos = df_mes_orig['CODIGO'].unique()
    
    # Crear la figura de subplots
    fig = make_subplots(
        rows=(len(codigos_unicos) + 2) // 3,  # Distribuir en 3 columnas
        cols=3,
        shared_yaxes=False,  # No compartir el eje Y
        subplot_titles=[f"Código: {codigo}" for codigo in codigos_unicos],
        vertical_spacing=0.06,  # Reducir el espaciado entre subplots
    )
    
    # Iterar sobre cada código para agregar subplots
    for i, codigo in enumerate(codigos_unicos, start=1):
        # Filtrar los datos para el código actual
        df_codigo = df_mes_orig[df_mes_orig['CODIGO'] == codigo].copy()
        # Asegurar que el índice es tratado como string
        if 'FECHA' in df_codigo.columns:
            df_codigo = df_codigo.set_index('FECHA')
        df_codigo['FECHA'] = df_codigo.index.astype('str')
        # Agregar una traza al subplot
        fig.add_trace(
            go.Scatter(
                x=df_codigo.index,  # Usar el índice de fecha
                y=df_codigo['DEMANDA'],
                mode='lines',
                name=codigo,
                line=dict(width=2, color = "#4682B4"),  # Personalizar el ancho de la línea
            ),
            row=(i - 1) // 3 + 1,  # Fila en la que estará el subplot
            col=(i - 1) % 3 + 1,   # Columna en la que estará el subplot
        )
    
    # Ajustar el diseño general
    fig.update_layout(
        height=220 * ((len(codigos_unicos) + 2) // 3),  # Altura ajustada según el número de subplots        
        title_text="Demanda por Código",
        title_x=0.5,  # Centrar el título
        title_font=dict(size=14),  # Tamaño de fuente del título principal
        showlegend=False,  # Eliminar la leyenda global
        font=dict(size=10),  # Tamaño de fuente general
        margin=dict(l=50, r=50, t=50, b=50),  # Ajustar márgenes
        template="ggplot2",
    )
    # Ajustar los títulos de los subplots
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=11)  # Reducir tamaño del texto de los títulos de subplots
        
     # Ajustar los títulos de los subplots y ejes

    fig.update_xaxes(
        title_font=dict(size=9),  # Tamaño de fuente para los títulos del eje X
        type='category',  # Especificar que el eje X es categórico
        tickmode='array',  # Asegurar que las etiquetas del eje X no sean interpretadas como fechas
    )
    fig.update_yaxes(title_font=dict(size=9))  # Tamaño de fuente para los títulos del eje Y
    # Mostrar la figura
    #fig.show()
    st.plotly_chart(fig, use_container_width=True)


# In[8]:


def imputar_outliers(df_outliers, sup, inf, n):
    
    # Generar un pronostico Ingenuo
    df_outliers['FORECAST'] = df_outliers['DEMANDA'].rolling(window=n, min_periods=1).mean().shift(1)
    
    # Calcular la mediana de la columna DEMANDA
    mediana_demanda = df_outliers['DEMANDA'].median()
    
    # Reemplazar los valores NaN en la columna FORECAST con la mediana de DEMANDA
    df_outliers['FORECAST'] = df_outliers['FORECAST'].fillna(mediana_demanda)

    # Calular error
    df_outliers['ERROR'] = df_outliers['DEMANDA'] - df_outliers['FORECAST']

    # Calcular Promedio y desviacion
    m = df_outliers['ERROR'].mean()
    s = df_outliers['ERROR'].std()

    # Aplicar Percentil sup e inf
    prob = norm.cdf(df_outliers['ERROR'],m,s)

    # Marcar principales Outliers
    outliers = (prob > sup) | (prob < inf)

    # Recalcular promedio y desviacion SIN principales outliers
    m2 = df_outliers.loc[~outliers,'ERROR'].mean()
    s2 = df_outliers.loc[~outliers,'ERROR'].std()

    # Calcular limite superior e inferior
    df_outliers['LIM_SUP'] = norm.ppf(sup,m2,s2) + df_outliers['FORECAST']
    df_outliers['LIM_INF'] = norm.ppf(inf,m2,s2) + df_outliers['FORECAST']

    # Usar .clip para imputar los valores por fuera de los limites
    df_outliers['NUEVA_DEM'] = df_outliers['DEMANDA'].clip(lower = df_outliers['LIM_INF'], upper= df_outliers['LIM_SUP'])

    # Señalar los valores imputados
    df_outliers['IS_OUTLIER'] = (df_outliers['DEMANDA'] != df_outliers['NUEVA_DEM'])
    
    return df_outliers


# In[9]:


def eliminar_outliers(df_mes_ceros, sup, inf, n):
    
    # Inicializar el DataFrame acumulado vacío
    df_acumulado = pd.DataFrame()
    
    # Aplicar la función imputar_outliers a cada grupo (SKU)
    for sku, df_sku in df_mes_ceros.groupby('CODIGO'):
        # Asegurarse de que 'FECHA' sea el índice
        
        # Imputar outliers para cada SKU
        df_imputado = imputar_outliers(df_sku.copy(), sup, inf, n)
        
        # Agregar el resultado al DataFrame acumulado
        df_acumulado = pd.concat([df_acumulado, df_imputado])

    # Crear copia de trabajo
    df_outliers = df_acumulado.copy()

    if 'CONSECUTIVO' in df_acumulado.columns: 
        df_acumulado = df_acumulado[['CODIGO',	
                             'CONSECUTIVO', 
                             'NUEVA_DEM']]
    else:
        df_acumulado = df_acumulado[['CODIGO',	
                               'NUEVA_DEM']]
        
    df_mes = df_acumulado.rename(columns={'NUEVA_DEM':'DEMANDA'})
    
    # Mostrar el DataFrame acumulado
    reporte_outliers = df_outliers[df_outliers['IS_OUTLIER'] == True].reset_index()
    
    return df_mes, df_outliers, reporte_outliers


# In[41]:


def graficar_outliers_subplots(df_mes_ceros, df_outliers, sup, inf, n):
    # Lista de SKUs únicos
    lista_skus = df_mes_ceros['CODIGO'].unique()
    
    # Calcular número de filas necesarias para 3 columnas
    n_cols = 3
    n_rows = -(-len(lista_skus) // n_cols)  # Redondeo hacia arriba

    # Crear los subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"SKU: {sku}" for sku in lista_skus],
        horizontal_spacing=0.05,
        vertical_spacing=0.06
    )
    
    # Definir colores
    color_demanda = "#FF6347"  # Rojo tomate
    color_forecast = "#FFA500"  # Naranja
    color_nueva_dem = "#4682B4" # Azul Acero
    
    # Iterar por cada SKU
    for idx, sku in enumerate(lista_skus):
        # Filtrar por SKU
        df_outliers = df_mes_ceros[df_mes_ceros['CODIGO'] == sku][['DEMANDA']].copy()

        # Aplicar función imputar_outliers
        df_outliers = imputar_outliers(df_outliers, sup, inf, n)

        # Calcular la posición en la cuadrícula
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1

        # Agregar trazas al subplot
        fig.add_trace(go.Scatter(
            x=df_outliers.index,
            y=df_outliers["DEMANDA"],
            name=f"Demanda - {sku}",
            mode='lines',
            marker=dict(size=6),
            line=dict(width=2, color=color_demanda)
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=df_outliers.index,
            y=df_outliers["FORECAST"],
            mode='lines',
            name=f"Forecast - {sku}",
            marker=dict(size=6),
            line=dict(width=2.2, dash='solid', color=color_forecast)
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=df_outliers.index,
            y=df_outliers["NUEVA_DEM"],
            mode='lines',
            name=f"Nueva Demanda - {sku}",
            marker=dict(size=6),
            line=dict(width=2, dash='solid', color=color_nueva_dem)
        ), row=row, col=col)

        # Límites superior e inferior
        fig.add_trace(go.Scatter(
            x=df_outliers.index,
            y=df_outliers["LIM_SUP"],
            mode='lines',
            name=f"Límite Superior - {sku}",
            line=dict(color='black', width=1, dash='dot'),
            opacity=0.5
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=df_outliers.index,
            y=df_outliers["LIM_INF"],
            mode='lines',
            name=f"Límite Inferior - {sku}",
            line=dict(color='black', width=1, dash='dot'),
            opacity=0.5
        ), row=row, col=col)

        # Outliers
        if df_outliers["IS_OUTLIER"].any():
            fig.add_trace(go.Scatter(
                x=df_outliers.index[df_outliers["IS_OUTLIER"]],
                y=df_outliers["DEMANDA"].loc[df_outliers["IS_OUTLIER"]],
                mode='markers',
                name=f"Outliers - {sku}",
                marker=dict(color=color_demanda, size=8, symbol='circle')
            ), row=row, col=col)

    # Actualizar diseño global
    fig.update_layout(
        title="Outliers vs Ventas por SKU",
        title_x=0.5,  # Centrar el título
        title_font=dict(size=14),  # Tamaño de fuente del título principal
        font=dict(size=10),  # Tamaño de fuente general
        margin=dict(l=50, r=50, t=50, b=50),  # Ajustar márgenes
        template="ggplot2",
        height=200 * n_rows,  # Ajustar altura según filas
        #width=900,  # Ancho fijo
        showlegend=False  # Ocultar leyenda global
    )
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=11)  # Reducir tamaño del texto de los títulos de subplots
    if 'CONSECUTIVO' in df_mes_ceros.columns:
        fig.update_xaxes(
            title_font=dict(size=9),  # Tamaño de fuente para los títulos del eje X
            type='category',  # Especificar que el eje X es categórico
            tickmode='array',  # Asegurar que las etiquetas del eje X no sean interpretadas como fechas
        )         
    # Mostrar la figura
    #fig.show()
    st.plotly_chart(fig, use_container_width=True)


# In[11]:


def crear_lista_skus(df_mes):
    lista_skus = df_mes['CODIGO'].unique()
    return lista_skus


# In[43]:


def calcular_meses_a_evaluar(df_sku, periodo_max_evaluacion, porc_eval):
       
    #Calculo del largo de cada serie de tiempo
    largo_serie_tiempo = len(df_sku)
    
    # Calculo del numero de meses a usar como testeo de acuerdo con porc_eval
    meses_evaluar = min(periodo_max_evaluacion, math.ceil(largo_serie_tiempo * porc_eval))

    return meses_evaluar


# In[60]:


def crear_rango_fechas(df_sku, meses_evaluar):

    if 'CONSECUTIVO' in df_sku.columns:
        # Seleccionar la fecha mas reciente de los datos originales
        ultima_fecha = df_sku['CONSECUTIVO'].max()
        # Definimos la fecha inicial de corte - meses_evaluar -1 
        inicio = ultima_fecha - meses_evaluar
        # Creamos un rango de fechas comenzando en inicio y terminando en ultina_fecha, con frecuencia mensual inicio MS
        rango_fechas = range(inicio, ultima_fecha+1)

    else:
        # Seleccionar la fecha mas reciente de los datos originales
        ultima_fecha = df_sku.index.max()
    
        # Definimos la fecha inicial de corte - meses_evaluar -1 
        inicio = ultima_fecha - pd.DateOffset(months=meses_evaluar)
  
        # Creamos un rango de fechas comenzando en inicio y terminando en ultina_fecha, con frecuencia mensual inicio MS
        rango_fechas = pd.date_range(start=inicio, end=ultima_fecha, freq='MS')

    return rango_fechas


# In[14]:


def crear_columnas_error(df):
    
    df['ERROR'] = df['DEMANDA'] - df['FORECAST'] # Error
    df['ABS_ERROR'] = df['ERROR'].abs() # Error Absoluto
    df['ERROR_PORC'] = np.where(df['DEMANDA'] == 0, 2, df['ABS_ERROR'] / df['DEMANDA']) # Error porcentual, devuelve 200% si la demanda es 0
    df['ERROR_CUADRADO'] = df['ERROR'] ** 2 # Error al cuadrado
    
    return df


# In[15]:


def metricas_error(df, imprimir):
     
    # Verificar si el total de la demanda es 0
    if df['DEMANDA'].sum() == 0:
        sesgo_porc = 2
        mae_porc = 2
        score = 2
    else:
        sesgo_porc = df['ERROR'].sum() / df['DEMANDA'].sum()
        mae_porc = df['ABS_ERROR'].sum() / df['DEMANDA'].sum()
        score = mae_porc + abs(sesgo_porc)
    
    rmse = np.sqrt(df['ERROR_CUADRADO'].mean())
        # Muestra los resultados formateados
    if imprimir == 1:
        print('MAE% modelo: {:.2%}'.format(mae_porc))
        print('Sesgo% modelo: {:.2%}'.format(sesgo_porc))
        print('Score modelo: {:.2%}'.format(score))
        print('RMSE modelo: {:.1f}'.format(rmse))
   
    return sesgo_porc, mae_porc, rmse, score


# In[45]:


def kpi_error_sku(df):
    
    if df is None:
        return None, None, None
        
    # Definicion de fechas de testeo
    if 'CONSECUTIVO' in df.columns:
        fecha_fin_testeo = df['CONSECUTIVO'].max()
        fecha_inicio_testeo = df['CONSECUTIVO'].min()
    else:    
        fecha_fin_testeo = df.index.max()
        fecha_inicio_testeo = df.index.min()

    # Crear columnas de error para cada pronostico generado
    df_test = crear_columnas_error(df)

    # Imprimir informacion de los periodos evaluados
    print('Periodo de Evaluacion desde:')
    if 'CONSECUTIVO' in df.columns:
        print(f"\033[1m{df_test['CONSECUTIVO'].min()} hasta {df_test['CONSECUTIVO'].max()}\033[0m") #\033[1m{}\033[0m muestra la linea en negrilla
    else:
        print(f"\033[1m{df_test.index.min().strftime('%Y-%m')} hasta {df_test.index.max().strftime('%Y-%m')}\033[0m") #\033[1m{}\033[0m muestra la linea en negrilla
    
    # Calcular metricas de error
    sesgo_porc, mae_porc, rmse, score = metricas_error(df_test, imprimir=1)
    
    # Agrupar df por sku
    grupo_sku_error = df_test.groupby(['CODIGO'], observed=True).agg({
                                                            'DEMANDA': 'sum',
                                                            'ERROR': 'sum',
                                                            'ABS_ERROR': 'sum',
                                                            'ERROR_CUADRADO': ['sum', 'count'],
                                                            }).reset_index()

    # Renombrar columnas
    grupo_sku_error.columns = ['CODIGO', 'DEMANDA', 'ERROR', 'ABS_ERROR', 
                             'ERROR_CUADRADO_suma', 'ERROR_CUADRADO_cuenta']
    
    # Calcular MAE% y Sesgo% de datos agregados por sku
    grupo_sku_error = calcular_error(grupo_sku_error)
    
    # Ordenar el DataFrame por 'SCORE%' en orden ascendente
    grupo_sku_error = grupo_sku_error.sort_values(by='SCORE%')
    
    # Aplicar formato porcentaje
    formatted_columns = grupo_sku_error[['MAE%', 'SESGO%', 'SCORE%']].map(lambda x: f'{x * 100:.2f}%')
    
    # Concatenar la columna "Codigo" sin formatear con las columnas formateadas
    grupo_sku_error_formato = pd.concat([grupo_sku_error[['CODIGO']], formatted_columns], axis=1)
    
    # Mostrar el resultado
    #display(grupo_sku_error_formato)

    # Agrupar por codigo y por Lag para almacenar RMSE
    grupo_sku_lag_error = df_test.groupby(['CODIGO', 'LAG'], observed=True).agg({
                                                            'DEMANDA': 'sum',
                                                            'ERROR': 'sum',
                                                            'ABS_ERROR': 'sum',
                                                            'ERROR_CUADRADO': ['sum', 'count'],
                                                            }).reset_index()

    # Renombrar columnas
    grupo_sku_lag_error.columns = ['CODIGO','LAG', 'DEMANDA', 'ERROR', 'ABS_ERROR', 
                             'ERROR_CUADRADO_suma', 'ERROR_CUADRADO_cuenta']
    
    # Calcular MAE% y Sesgo% de datos agregados por lag
    grupo_sku_lag_error = calcular_error(grupo_sku_lag_error)

    # Calcular error rmse por lag
    rmse_sku_lag = grupo_sku_lag_error[['CODIGO','LAG','RMSE']]
    
    # Agrupar por codigo para almacenar RMSE
    #df_test['Mes'] = df_test.index.month
    grupo_sku_mes_error = df_test.groupby(['CODIGO', 
                                           #'Mes'
                                          ], observed=True).agg({
                                                            'DEMANDA': 'sum',
                                                            'ERROR': 'sum',
                                                            'ABS_ERROR': 'sum',
                                                            'ERROR_CUADRADO': ['sum', 'count'],
                                                            }).reset_index()
    
    # Renombrar columnas
    grupo_sku_mes_error.columns = ['CODIGO',
                                   #'Mes', 
                                   'DEMANDA', 'ERROR', 'ABS_ERROR', 
                             'ERROR_CUADRADO_suma', 'ERROR_CUADRADO_cuenta']

    # Calcular error rmse por codigo
    grupo_sku_mes_error = calcular_error(grupo_sku_mes_error)

    # Filtrar las columnas para mejor visualizacion
    rmse_sku_mes = grupo_sku_mes_error[['CODIGO',
                                        #'Mes',
                                        'RMSE']]
    
    return grupo_sku_error_formato, rmse_sku_lag, rmse_sku_mes


# In[17]:


def calcular_error(df):
    df['MAE%'] = df['ABS_ERROR']/df['DEMANDA']
    df['SESGO%'] = df['ERROR']/df['DEMANDA']
    df['SCORE%'] = df['MAE%'] + df['SESGO%'].abs()
    if 'ERROR_CUADRADO_suma' in df.columns:
        df['RMSE'] = np.sqrt(df['ERROR_CUADRADO_suma'] / df['ERROR_CUADRADO_cuenta'])
    return df


# In[18]:


def evaluar_lags(df):
    # Calcular los scores por lag
    df_lags = df.groupby('LAG')[['ERROR', 'ABS_ERROR', 'DEMANDA']].sum()
        
    # Calcular los scores por lag evitando la división cuando DEMANDA es cero                  
    df_lags['SCORE%'] = np.where(df_lags['DEMANDA'] == 0, 2,
                            (df_lags['ABS_ERROR'] / df_lags['DEMANDA']) + abs(df_lags['ERROR'] / df_lags['DEMANDA'])
                            )
    return df_lags


# In[48]:


def kpi_error_lag(df):
    
    if df is None:
        return None, None
    # Definicion de fechas de testeo
    if 'CONSECUTIVO' in df.columns:
        fecha_fin_testeo = df['CONSECUTIVO'].max()
        fecha_inicio_testeo = df['CONSECUTIVO'].min()
    else:    
        fecha_fin_testeo = df.index.max()
        fecha_inicio_testeo = df.index.min()
    
    # Crear columnas de error  
    df_test = crear_columnas_error(df)
    print('Periodo de Evaluacion desde:')   
    if 'CONSECUTIVO' in df.columns:
        print(f"\033[1m{df_test['CONSECUTIVO'].min()} hasta {df_test['CONSECUTIVO'].max()}\033[0m") #\033[1m{}\033[0m muestra la linea en negrilla
    else:
        print(f"\033[1m{df_test.index.min().strftime('%Y-%m')} hasta {df_test.index.max().strftime('%Y-%m')}\033[0m") #\033[1m{}\033[0m muestra la linea en negrilla

    # Calcular loas metricas de error
    sesgo_porc, mae_porc, rmse, score = metricas_error(df_test, imprimir=1)
    
    # Agrupar df por mes
    grupo_mes_error = df_test.groupby(['LAG']).agg({
                                                            'DEMANDA': 'sum',
                                                            'ERROR': 'sum',
                                                            'ABS_ERROR': 'sum',
                                                            'ERROR_CUADRADO': ['sum', 'count'],
                                                            }).reset_index()

    # Renombrar columnas
    grupo_mes_error.columns = ['LAG', 'DEMANDA', 'ERROR', 'ABS_ERROR', 
                             'ERROR_CUADRADO_suma', 'ERROR_CUADRADO_cuenta']
    
    # Calcular MAE% y Sesgo% de datos agregados por mes
    grupo_mes_error = calcular_error(grupo_mes_error)
    
    # Aplicar formato porcentaje
    formatted_columns = grupo_mes_error[['MAE%', 'SESGO%', 'SCORE%']].map(lambda x: f'{x * 100:.2f}%')
    
    # Concatenar la columna "Lag" sin formatear con las columnas formateadas
    grupo_mes_error_formato = pd.concat([grupo_mes_error[['LAG']], formatted_columns], axis=1)
    
    # Mostrar el resultado
    #display(grupo_mes_error_formato)

    # Agrupar por codigo y por Lag para almacenar RMSE
    grupo_sku_lag_error = df_test.groupby(['CODIGO', 'LAG'], observed=True).agg({
                                                            'DEMANDA': 'sum',
                                                            'ERROR': 'sum',
                                                            'ABS_ERROR': 'sum',
                                                            'ERROR_CUADRADO': ['sum', 'count'],
                                                            }).reset_index()

    # Renombrar columnas
    grupo_sku_lag_error.columns = ['CODIGO', 'LAG', 'DEMANDA', 'ERROR', 'ABS_ERROR', 
                             'ERROR_CUADRADO_suma', 'ERROR_CUADRADO_cuenta']

    # Calcular columnas de error por lag
    grupo_sku_lag_error = calcular_error(grupo_sku_lag_error)

    # Filtrar columnas para mejor visualizacion
    rmse_sku_lag = grupo_sku_lag_error[['CODIGO', 'LAG','RMSE']]
    
    return grupo_mes_error_formato, df_test


# In[50]:


# Agrupar df por sku
def agrupar_por_codigo(df):
    grupo_sku_error = df.groupby(['CODIGO'], observed=True).agg({
                                                                'DEMANDA': 'sum',
                                                                'ERROR': 'sum',
                                                                'ABS_ERROR': 'sum',
                                                                'ERROR_CUADRADO': ['sum', 'count'],
                                                                }).reset_index()
    grupo_sku_error.columns = ['CODIGO', 'DEMANDA', 'ERROR', 'ABS_ERROR', 
                                 'ERROR_CUADRADO_suma', 'ERROR_CUADRADO_cuenta']
    # Calcular MAE% y Sesgo% de datos agregados por sku
    grupo_sku_error = calcular_error(grupo_sku_error)
    grupo_sku_error = grupo_sku_error[['CODIGO','MAE%',	'SESGO%',	'SCORE%',	'RMSE']]
    
    # Agrupar por codigo y por Lag 
    grupo_sku_lag_error = df.groupby(['CODIGO', 'LAG'], observed=True).agg({
                                                            'DEMANDA': 'sum',
                                                            'ERROR': 'sum',
                                                            'ABS_ERROR': 'sum',
                                                            'ERROR_CUADRADO': ['sum', 'count'],
                                                            }).reset_index()
    
    grupo_sku_lag_error.columns = ['CODIGO','LAG', 'DEMANDA', 'ERROR', 'ABS_ERROR', 
                             'ERROR_CUADRADO_suma', 'ERROR_CUADRADO_cuenta']
    
    grupo_sku_lag_error = calcular_error(grupo_sku_lag_error)
    grupo_sku_lag_error = grupo_sku_lag_error[['CODIGO','LAG','MAE%',	'SESGO%',	'SCORE%',	'RMSE']]

    # Pivotear el DataFrame de lag
    pivoted_lags = grupo_sku_lag_error.pivot(index='CODIGO', columns='LAG', values='SCORE%')
    pivoted_lags.columns = [f"score_{col}" for col in pivoted_lags.columns]
    
    # Unir con el DataFrame principal
    tabla_final = grupo_sku_error.merge(pivoted_lags, on='CODIGO', how='left')
    
    # Renombrar columnas para cumplir con el formato
    tabla_final = tabla_final.rename(columns={'MAE%': 'mae_porc', 'SESGO%': 'sesgo_porc', 'SCORE%': 'score', 'RMSE': 'rmse'})
    
    return tabla_final


# In[53]:


def construir_pronostico_pms_nv(df_mejor, df_mes, meses_a_pronosticar_produccion, nombre_modelo):

    # Crear un nuevo DataFrame para almacenar los resultados
    data = []
    
    # Iterar por cada fila de df_mejor
    for _, row in df_mejor.iterrows():
        codigo = row["CODIGO"]
        ultimo_forecast = row["ultimo_forecast"]
        # Generar las fechas para los meses pronosticados
        fechas = [df_mes['CONSECUTIVO'].max() + i for i in range(1, meses_a_pronosticar_produccion + 1)]
    
        # Generar las filas para los meses pronosticados
        for i, fecha in enumerate(fechas, start=1):
            data.append({
                "CONSECUTIVO": fecha,
                "CODIGO": codigo,
                "FORECAST": ultimo_forecast,
                "LAG": f"Lag_{i}"
            })
    
    # Crear el nuevo DataFrame
    df_forecast = pd.DataFrame(data)
    #df_forecast = df_forecast.set_index('FECHA')
    df_forecast['MODELO'] = nombre_modelo
    
    # Visualizar el resultado
    return df_forecast


# In[55]:


def adicionar_nombre_modelo_serie_tiempo(df, nombre_modelo):
    if df is None:
        return None
    df['MODELO'] = nombre_modelo
    if 'CONSECUTIVO' in df.columns:
        df = df[['CODIGO','CONSECUTIVO','FORECAST','LAG','MODELO']]
    else:
        df = df[['CODIGO','FORECAST','LAG','MODELO']]
    df.index.name = 'FECHA'
    return df


# In[57]:


def evaluar_y_generar_pms_nv(df_mes, df_mes_ceros, lista_skus, 
                          periodo_max_evaluacion, 
                          porc_eval, 
                          meses_a_pronosticar_evaluacion,
                         barra_progreso,
                         status_text):
    
    mejor_n = [] # Bolsa para guardar resultados por cada sku
    acumulado_forecast = [] # Bolsa para guardar los resultados de todos los pronosticos
    total_series = len(lista_skus)  
    
    for i, sku in enumerate(tqdm(lista_skus, desc="Procesando SKUs")):
        # Actualizar barra de progreso y mensaje de estado
        barra_progreso.progress((i + 1) / total_series)
        status_text.text(f"Evaluando PMS para SKU N° {i + 1} de {total_series}...")
        
        resultados_n = [] # Bolsa para guardar resultados evalaudos por cada n
        resultados_datos_evaluacion = [] # Bolsa para guardar datos sin evaluar por cada n
        
        df_sku_fecha = df_mes[df_mes['CODIGO'] == sku].copy() # Filtrar df_mes por cada Sku
        df_sku_fecha_ceros = df_mes_ceros[df_mes_ceros['CODIGO'] == sku].copy() # Filtrar df_mes_ceros por cada Sku

        # Evaluar el largo de la serie de tiempo y calcular meses a evaluar aper cada sku
        meses_evaluar = calcular_meses_a_evaluar(df_sku_fecha, periodo_max_evaluacion, porc_eval)
        #print('meses_evaluar:',meses_evaluar)
        # Crear el rango de fechas para cortar el set de datos de acuerdo con meses a evaluar
        rango_fechas = crear_rango_fechas(df_sku_fecha, meses_evaluar)
        #print('rango_fechas:',rango_fechas)  
        # Tamaño de histórico n maximo y rango   
        n_max = max(2, len(df_sku_fecha) - meses_evaluar)        
        rango_n = range(1, n_max)
        #print('n, rango_n:',n_max, rango_n) 
        # Iterar por cada posible tamaño de n
        for n in rango_n: 
            
            datos_evaluacion = []  # Bolsa para guardar resultados evaluados
            ultimo_forecast_n = None  # Variable para almacenar el último forecast de cada n
        
            for fecha_corte in rango_fechas:
                # Filtrar datos hasta la fecha de corte
                df_sku_fecha_temp = df_sku_fecha[df_sku_fecha['CONSECUTIVO'] <= fecha_corte].copy()
                
                if len(df_sku_fecha_temp['DEMANDA']) > 1:
                    # Calcular el forecast usando una media móvil con ventana n
                    #print(len(df_sku_fecha_temp['DEMANDA']))
                    df_sku_fecha_temp['FORECAST'] = df_sku_fecha_temp['DEMANDA'].rolling(window=n, min_periods=1).mean()
 
                    forecast = [df_sku_fecha_temp['FORECAST'].iloc[-1]]
                    
                else:
                    forecast = df_sku_fecha_temp['DEMANDA'].iloc[-1]
                
                # Generar los próximos lags para el forecast actual            
                datos_forecast = pd.DataFrame({                
                    'fecha':fecha_corte,
                    'n':n,
                    'CODIGO': sku,
                    'FORECAST': forecast,
                    'LAG': [f'Lag_{i}' for i in range(1, meses_a_pronosticar_evaluacion + 1)]}, 
                    index=[df_sku_fecha_temp['CONSECUTIVO'].iloc[-1] 
                                      + i for i in range(1, meses_a_pronosticar_evaluacion + 1)]) # Genera titulo Lags dinamicamente
                datos_forecast = datos_forecast.reset_index()

                # Step 2: Rename the index column to match the key in df_sku_fecha_ceros
                datos_forecast = datos_forecast.rename(columns={'index': 'CONSECUTIVO'})
                           
                datos_forecast_demanda = datos_forecast.merge(
                    df_sku_fecha_ceros[['DEMANDA', 'CONSECUTIVO']],
                    how='left',
                    on='CONSECUTIVO'
                )
  
                # Eliminar NaN si se esta pronosticando
                if porc_eval != 0:
                    datos_forecast_demanda = datos_forecast_demanda.dropna()
                    
                # Acumular data frames por cada fecha
                datos_evaluacion.append(datos_forecast_demanda)
                
                # Guardar el último forecast de esta iteración de fecha_corte
                ultimo_forecast_n = datos_forecast  # Se actualiza en cada fecha_corte               
                       
            # Concatenar todos los DataFrames de la evaluación
            df_evaluacion_final = pd.concat(datos_evaluacion)#.dropna()
                     
            # Calcular columnas de error
            df_columnas_error = crear_columnas_error(df_evaluacion_final)
            
            # Calcular métricas de error
            sesgo_porc, mae_porc, rmse, score = metricas_error(df_columnas_error, imprimir=0)
              
            # Calcular los scores por lag
            df_lags = evaluar_lags(df_columnas_error)
            
            # Agregar resultados y el último forecast de cada n a la lista resultados_n
            resultados_n.append({
                'CODIGO':sku,
                'parametro': n,         
                'sesgo_porc': sesgo_porc,        
                'mae_porc': mae_porc,        
                'rmse': rmse,
                'score': score,
                **{f'score_{lag}': score_value for lag, score_value in zip(df_lags.index, df_lags['SCORE%'])},  # Agrega dinámicamente scores por lag
                'ultimo_forecast': forecast[0],  # Guarda el último forecast de n                
            })                  

            # Acumula los datos de pronostico para evaluacion aparte
            resultados_datos_evaluacion.append({                
                'score': score,
                'datos_evaluacion': datos_evaluacion
            })
           
        # Crear el DataFrame final 
        df_kpi = pd.DataFrame(resultados_n)
        #display(df_kpi) 
        # Crear df con info del n con menor score
        df_min_score = df_kpi[df_kpi['score'] == df_kpi['score'].min()]
        # Seleccionar el forecast correspondiente al mejor score de df_min_score
        forecast_optimo = df_min_score['ultimo_forecast'].iloc[0]
        #Acumular df con resultados por sku
        mejor_n.append(df_min_score)
        
        # Crear un data frame con los datos de evaluacion 
        df_kpi_datos_evaluacion = pd.DataFrame(resultados_datos_evaluacion) 
        # Seleccionar el n con menor score
        mejor_n_row = df_kpi_datos_evaluacion[df_kpi_datos_evaluacion['score'] == df_kpi_datos_evaluacion['score'].min()].iloc[0]
        # Recuperar los datos del mejor n
        mejor_n_datos = mejor_n_row['datos_evaluacion']  
       # Almacenar los datos del mejor n para este SKU
        acumulado_forecast.extend(mejor_n_datos)     
       
    # Concatenar mejor_n para obtener un solo df
    df_mejor_n = pd.concat(mejor_n)
    # Eliminar duplicados conservando solo el primer codigo
    df_mejor_n = df_mejor_n.drop_duplicates(subset='CODIGO', keep='first')
    # Mover ultimo_forecast a la ultima columna
    columnas = [col for col in df_mejor_n.columns if col != 'ultimo_forecast'] + ['ultimo_forecast']
    df_mejor_n = df_mejor_n[columnas]
    
    df_forecast_pms = pd.concat(acumulado_forecast, ignore_index=False)
    status_text.text("")
    barra_progreso.empty()  
    return df_mejor_n, df_forecast_pms 


# In[58]:


def encontrar_mejor_se_nv(df_mes, df_mes_ceros, lista_skus, periodo_max_evaluacion, porc_eval, 
                        meses_a_pronosticar_evaluacion,
                        barra_progreso,
                        status_text):
    
    mejor_se = [] # Bolsa para guardar resultados por cada sku
    resultados_se = []  # Bolsa para guardar resultados por cada n
    acumulado_df_evaluacion_final = [] # Bolsa para guardar los resultados de todos los pronosticos
    total_series = len(lista_skus)  
    
    for i, sku in enumerate(tqdm(lista_skus, desc="Procesando SKUs")):    
    #for sku in tqdm(lista_skus, desc="Procesando SKUs"):
        # Actualizar barra de progreso y mensaje de estado
        barra_progreso.progress((i + 1) / total_series)
        status_text.text(f"Evaluando SE para SKU N° {i + 1} de {total_series}...")
        
        datos_evaluacion_se = []  # Bolsa para guardar resultados de evaluación por fecha
                
        df_sku_fecha = df_mes[df_mes['CODIGO'] == sku].copy() # Filtrar df_almacen_semana por cada SKU
        df_sku_fecha_ceros = df_mes_ceros[df_mes_ceros['CODIGO'] == sku].copy() # Filtrar df_mes_ceros por cada Sku            
        
        # Evaluar el largo de la serie de tiempo y calcular meses a evaluar para cada sku
        #meses_evaluar = calcular_meses_a_evaluar(df_sku_fecha, periodo_max_evaluacion, porc_eval)
        meses_evaluar = calcular_meses_a_evaluar(df_sku_fecha, periodo_max_evaluacion, porc_eval)
        # Crear el rango de fechas para corar el set de datos de acuerdo con meses a evaluar
        #rango_fechas = crear_rango_fechas(df_sku_fecha, meses_evaluar)
        rango_fechas = crear_rango_fechas(df_sku_fecha, meses_evaluar)
        # Iterar por fecha
        for fecha_corte in rango_fechas:
            
            # Filtrar datos hasta la fecha de corte
            df_sku_fecha_temp = df_sku_fecha[df_sku_fecha['CONSECUTIVO'] <= fecha_corte].copy()                 
    
            # Extraer la demanda como un array
            demanda = df_sku_fecha_temp['DEMANDA'].values

            # Chequeo de la longitud de cada serie de tiempo
            if len(demanda) < 2:
                print(f'El sku {sku} no tiene suficientes datos para la fecha de corte {fecha_corte}') # Ver SKUs con datos insuficientes y su fecha de corte
                forecast = np.NaN
            else:                            
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model = SimpleExpSmoothing(demanda).fit(smoothing_level=None, optimized=True)
   
                    # Calcular pronostico para los proximos periodos
                    forecast = model.forecast(steps=meses_a_pronosticar_evaluacion)
                           
                    # Generar los próximos lags para el forecast actual            
                    datos_forecast = pd.DataFrame({
                        'FECHA':fecha_corte,
                        'CODIGO': sku,
                        'FORECAST': forecast,
                        'LAG': [f'Lag_{i}' for i in range(1, meses_a_pronosticar_evaluacion + 1)]
                                }, 
                        index=[df_sku_fecha_temp['CONSECUTIVO'].iloc[-1]
                                          + i for i in range(1,  meses_a_pronosticar_evaluacion + 1)])
                    datos_forecast = datos_forecast.reset_index()
                    datos_forecast = datos_forecast.rename(columns={'index': 'CONSECUTIVO'})
                    # Unir forecast con la demanda real para evaluar
                    datos_forecast_demanda = datos_forecast.merge(
                                        df_sku_fecha_ceros[['DEMANDA', 'CONSECUTIVO']],
                                        how='left',
                                        on='CONSECUTIVO'
                                    )

                    if porc_eval != 0:
                        datos_forecast_demanda = datos_forecast_demanda.dropna()
                                        
                    # Acumular datos para evaluar el pronostico
                    datos_evaluacion_se.append(datos_forecast_demanda)
                    
                    # Guardar el último alfa de esta iteración de fecha_corte
                    ultimo_forecast_alfa = model.params['smoothing_level']  # Se actualiza en cada fecha_corte

       # Concatenar df_evaluacion para obtener un solo df
        df_evaluacion = pd.concat(datos_evaluacion_se)

        # Generar una copia de los datos de evaluacion para analizar aparte
        df_evaluacion_final = df_evaluacion.copy()

        # Acumular los datos de evaluacion por fecha
        acumulado_df_evaluacion_final.append(df_evaluacion_final)
                     
        # Calcular columnas de error sobre df inicial de evaluacion
        df_columnas_error = crear_columnas_error(df_evaluacion)
        
        # Calcular métricas de error
        sesgo_porc, mae_porc, rmse, score = metricas_error(df_columnas_error, imprimir=0)
        
        # Calcular los scores por lag
        df_lags = evaluar_lags(df_columnas_error)
       
        # Agregar resultados y el último forecast de cada n a la lista resultados_n
        resultados_se.append({
            'CODIGO' : sku,
            'parametro': ultimo_forecast_alfa,            
            'sesgo_porc': sesgo_porc,            
            'mae_porc': mae_porc,            
            'rmse': rmse,
            'score': score,
            **{f'score_{lag}': score_value for lag, score_value in zip(df_lags.index, df_lags['SCORE%'])},  # Agrega dinámicamente scores por lag               
            'ultimo_forecast': forecast[0]  # Guarda el último forecast            
        })
           
    # Crear el DataFrame con los resultados
    df_kpi = pd.DataFrame(resultados_se)

    # Acumular por cada sku
    mejor_se.append(df_kpi)

    # Concatenar para obtener un solo df
    df_mejor_se = pd.concat(mejor_se)

    # Concatener acumulado de matriz de datos de evaluacion para obtener un solo df
    df_forecast_se = pd.concat(acumulado_df_evaluacion_final)
    status_text.text("")
    barra_progreso.empty()
    return df_mejor_se,  df_forecast_se


# In[66]:


def aplicar_regresion_lineal_simple_nv(lista_skus, df_mes, df_mes_ceros, 
                                    periodo_max_evaluacion, porc_eval, 
                                    meses_a_pronosticar_evaluacion,
                                    barra_progreso,
                                    status_text):
   
    resultados_datos_forecast_demanda_lineal = []
    resultados_datos_forecast_demanda_estacional = []
    resultados_rl_lineal = []
    resultados_rl_estacional = []
    total_series = len(lista_skus)  
    
    for i, sku in enumerate(tqdm(lista_skus, desc="Procesando SKUs")):    
    #for sku in tqdm(lista_skus, desc="Procesando SKUs"):
        # Actualizar barra de progreso y mensaje de estado
        barra_progreso.progress((i + 1) / total_series)
        status_text.text(f"Evaluando RL para SKU N° {i + 1} de {total_series}...")
        
        resultados_regresion_lineal = []  
        resultados_regresion_estacional = []  
                    
        df_sku_fecha = df_mes[df_mes['CODIGO'] == sku].copy() # Filtrar df_almacen_semana por cada SKU
        df_sku_fecha_ceros = df_mes_ceros[df_mes_ceros['CODIGO'] == sku].copy() # Filtrar df_mes_ceros por cada Sku            
        
        # Evaluar el largo de la serie de tiempo y calcular meses a evaluar para cada sku
        #meses_evaluar = calcular_meses_a_evaluar(df_sku_fecha, periodo_max_evaluacion, porc_eval)
        meses_evaluar = calcular_meses_a_evaluar(df_sku_fecha, periodo_max_evaluacion, porc_eval)
 
        rango_fechas = crear_rango_fechas(df_sku_fecha, meses_evaluar)
        # Iterar por fecha
        for fecha_corte in rango_fechas:
            # Filtrar datos hasta la fecha de corte
            df_sku_fecha_temp = df_sku_fecha[df_sku_fecha['CONSECUTIVO'] <= fecha_corte].copy()                 
            
            # Extraer la demanda como un array
            demanda = df_sku_fecha_temp['DEMANDA'].values
            
            if len(demanda) >= 4:
                # Generar y adecuar la variable independiente tiempo
                X = np.arange(1, len(demanda)+1).reshape(-1, 1)
                y = demanda
                # Modelo de Regresión Lineal
                model = LinearRegression()
                model.fit(X, y)
                
                # Pronóstico para los próximos periodos
                limite_sup_pronost = len(demanda)+1 + meses_a_pronosticar_evaluacion
                X_futuro = np.arange(len(demanda)+1, limite_sup_pronost).reshape(-1, 1)
                y_futuro_lineal = model.predict(X_futuro)

                if len(demanda) >= 19:
                    factores_estacionales_mes = demanda[-19:] / demanda[-19:].mean() 
                    y_futuro_estacional =  y_futuro_lineal * factores_estacionales_mes[:len(y_futuro_lineal)]
                else:    
                    y_futuro_estacional = [np.NaN] * meses_a_pronosticar_evaluacion 
            
            else:
                print(f"Sin datos: SKU={sku}, Fecha={fecha_corte}, Datos disponibles={len(demanda)}") 
                y_futuro_lineal = [np.NaN] * meses_a_pronosticar_evaluacion 
            
            # Almacenar los resultados lineales con código y periodo
            for periodo, prediccion in zip(range(len(demanda)+1, limite_sup_pronost), y_futuro_lineal):
                fecha_pronostico = df_sku_fecha_temp['CONSECUTIVO'].max() + periodo-len(demanda)
                lag = f"Lag_{periodo - len(demanda)}"
                resultados_regresion_lineal.append({'CODIGO': sku, 
                                                    'CONSECUTIVO': periodo, 
                                                    'FECHA': fecha_corte, 
                                                    'FORECAST': prediccion, 
                                                    'LAG': lag})
                #display(resultados_regresion_lineal)
            # Almacenar los resultados estacionales con código y periodo
            for periodo, prediccion in zip(range(len(demanda)+1, limite_sup_pronost), y_futuro_estacional):
                fecha_pronostico = df_sku_fecha_temp['CONSECUTIVO'].max() + periodo-len(demanda)
                lag = f"Lag_{periodo - len(demanda)}"
                resultados_regresion_estacional.append({'CODIGO': sku, 
                                                        'CONSECUTIVO': periodo, 
                                                        'FECHA': fecha_corte, 
                                                        'FORECAST': prediccion, 
                                                        'LAG': lag})
        
        df_forecast_regresion_lineal = pd.DataFrame(resultados_regresion_lineal)#.set_index('FECHA')
        #df_forecast_regresion_lineal = df_forecast_regresion_lineal.set_index('FECHA')
        #display( df_forecast_regresion_lineal)
        df_forecast_regresion_estacional = pd.DataFrame(resultados_regresion_estacional)#.set_index('FECHA')
        
        # Unir forecast con la demanda real para evaluar
        datos_forecast_demanda_lineal = df_forecast_regresion_lineal.merge(
            df_sku_fecha_ceros[['DEMANDA', 'CONSECUTIVO']], 
            how='left', 
            on='CONSECUTIVO', 
            
        )
        # Unir forecast con la demanda real para evaluar
        datos_forecast_demanda_estacional = df_forecast_regresion_estacional.merge(
            df_sku_fecha_ceros[['DEMANDA', 'CONSECUTIVO']], 
            how='left', 
            on='CONSECUTIVO', 
        )
        # Condicionar eliminacion de NaN a si es evaluacion o generacion de pronostico
        if porc_eval != 0:
            datos_forecast_demanda_lineal = datos_forecast_demanda_lineal.dropna()
            datos_forecast_demanda_estacional = datos_forecast_demanda_estacional.dropna()           
    
        # Generar una copia de los datos de evaluacion para analizar aparte
        datos_forecast_demanda_lineal_final = datos_forecast_demanda_lineal.copy() 
        datos_forecast_demanda_estacional_final = datos_forecast_demanda_estacional.copy() 
        
        resultados_datos_forecast_demanda_lineal.append(datos_forecast_demanda_lineal_final)
        resultados_datos_forecast_demanda_estacional.append(datos_forecast_demanda_estacional_final)
        
        # Calcular columnas de error sobre df inicial de evaluación
        df_columnas_error_lineal = crear_columnas_error(datos_forecast_demanda_lineal)
        df_columnas_error_estacional = crear_columnas_error(datos_forecast_demanda_estacional)
        
        # Calcular métricas de error
        sesgo_porc_lineal, mae_porc_lineal, rmse_lineal, score_lineal = metricas_error(df_columnas_error_lineal, imprimir=0)
        sesgo_porc_estacional, mae_porc_estacional, rmse_estacional, score_estacional = metricas_error(df_columnas_error_estacional, imprimir=0)
        
        # Calcular los scores por lag
        df_lags_lineal = evaluar_lags(df_columnas_error_lineal)
        df_lags_estacional = evaluar_lags(df_columnas_error_estacional)
        
        # Guardar KPIs lineales
        resultados_rl_lineal.append({
            'CODIGO': sku,
            'sesgo_porc': sesgo_porc_lineal,
            'mae_porc': mae_porc_lineal,
            'rmse': rmse_lineal,
            'score': score_lineal,
            **{f'score_{lag}': score_value for lag, score_value in zip(df_lags_lineal.index, df_lags_lineal['SCORE%'])},              
        })
        
        # Guardar KPIs estacionales
        resultados_rl_estacional.append({
            'CODIGO': sku,
            'sesgo_porc': sesgo_porc_estacional,
            'mae_porc': mae_porc_estacional,
            'rmse': rmse_estacional,
            'score': score_estacional,
            **{f'score_{lag}': score_value for lag, score_value in zip(df_lags_estacional.index, df_lags_estacional['SCORE%'])},              
        })
    # Concatenar resultados acumulados
    df_forecast_rl_lineal = pd.concat(resultados_datos_forecast_demanda_lineal, ignore_index=False)
    df_forecast_rl_estacional = pd.concat(resultados_datos_forecast_demanda_estacional, ignore_index=False) 
    
    # Crear el DataFrame con los resultados de KPIs
    df_mejor_rl_lineal = pd.DataFrame(resultados_rl_lineal)
    df_mejor_rl_estacional = pd.DataFrame(resultados_rl_estacional)
    status_text.text("")
    barra_progreso.empty()                                     
    # Visualizar resultados
    return df_mejor_rl_lineal, df_mejor_rl_estacional, df_forecast_rl_lineal, df_forecast_rl_estacional


# In[67]:


def aplicar_mstl_nv(lista_skus, df_mes, df_mes_ceros, 
                 periodo_max_evaluacion, porc_eval, 
                 meses_a_pronosticar_evaluacion, peso_ult_data,
                 barra_progreso, status_text):

    # Listas para acumular resultados
    df_lag_forecasts = []
    total_series = len(lista_skus)
    
    for i, sku in enumerate(tqdm(lista_skus, desc="Procesando SKUs")):
        # Actualizar barra de progreso y mensaje de estado
        barra_progreso.progress((i + 1) / total_series)
        status_text.text(f"Evaluando MSTL para SKU N° {i + 1} de {total_series}...")
    
        # Filtrar datos por SKU
        df_sku_fecha = df_mes[df_mes['CODIGO'] == sku].copy()
        df_sku_fecha_ceros = df_mes_ceros[df_mes_ceros['CODIGO'] == sku].copy()

        # Calcular campañas a evaluar y rango de fechas
        meses_evaluar = calcular_meses_a_evaluar(df_sku_fecha, periodo_max_evaluacion, porc_eval)
        rango_fechas = crear_rango_fechas(df_sku_fecha, meses_evaluar)

        forecasts = []

        for fecha_corte in rango_fechas:
            # Filtrar datos hasta la fecha de corte
            df_sku_fecha_temp = df_sku_fecha[df_sku_fecha['CONSECUTIVO'] <= fecha_corte].copy()
            date = df_sku_fecha_temp['CONSECUTIVO']
            demanda = df_sku_fecha_temp['DEMANDA'].values
            demand_series = pd.Series(demanda, index=date)
            #print(sku, 'largo demanda:',len(demand_series))
            if len(demand_series) > 38:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    try:
                        # Aplicar descomposición MSTL
                        mstl_model = MSTL(demand_series, periods=19, stl_kwargs={'seasonal_deg': 0})
                        descomposicion = mstl_model.fit()

                        # Componentes de descomposición
                        tendencia = descomposicion.trend
                        seasonal = descomposicion.seasonal
                        indice_tiempo = pd.to_numeric(demand_series.index)
                        
                        # Pesos exponenciales para regresión
                        pesos = np.exp(peso_ult_data * np.arange(len(tendencia)))
                        
                        # Regresión polinómica
                        poly = PolynomialFeatures(degree=2)
                        X_poly = poly.fit_transform(indice_tiempo.values.reshape(-1, 1))
                        model = LinearRegression()
                        model.fit(X_poly, tendencia, sample_weight=pesos)
                        
                        # Proyección de tendencia
                        fechas_futuras = [demand_series.index[-1] + i + 1 for i in range(meses_a_pronosticar_evaluacion)]
                        indice_fechas_futuras = pd.to_numeric(pd.Index(fechas_futuras))
                        X_poly_futura = poly.transform(indice_fechas_futuras.values.reshape(-1, 1))
                        pronostico_tendencia = model.predict(X_poly_futura)
                        
                        # Proyección de estacionalidad
                        estacionalidad_promedio = seasonal.groupby(seasonal.index).mean()
                        mes_inicial = fechas_futuras[0]
                        pronostico_estacional = [
                            estacionalidad_promedio[(mes_inicial + i - 1) % 12 + 1] for i in range(meses_a_pronosticar_evaluacion)
                        ]
                        
                        # Pronóstico final
                        pronostico_final = pronostico_tendencia + pronostico_estacional
                        pronostico_final_series = pd.Series(pronostico_final, index=fechas_futuras)
                        
                        forecasts.append((sku, date, pronostico_final_series))
                       
                    except Exception as e:
                        print(f"Error al ajustar MSTL para {sku}: {e}")
                        continue

        # Procesar resultados de los pronósticos
        for sku, fechas_originales, forecast_series in forecasts:
            ultima_fecha = fechas_originales[-1]
            fechas_pronosticos = [ultima_fecha + i + 1 for i in range(len(forecast_series))]
            lags = [f"Lag_{i}" for i in range(1, len(forecast_series) + 1)]
            temp_df = pd.DataFrame({
                'FECHA': fechas_pronosticos,
                'LAG': lags,
                'CODIGO': [sku] * len(forecast_series),
                'FORECAST': forecast_series.values,
            })
            df_lag_forecasts.append(temp_df)
    
    if df_lag_forecasts:
        df_forecasts_mstl = pd.concat(df_lag_forecasts, ignore_index=True)
    else:
        print("No hay SKUs con suficientes campañas para un análisis estacional")
        return None, None

    # Unir forecast con la demanda real para evaluar
    datos_forecast_mstl = df_forecasts_mstl.merge(
        df_mes_ceros[['CODIGO', 'DEMANDA']], 
        how='left', 
        left_on=['CODIGO', 'FECHA'], 
        right_on=['CODIGO', df_mes_ceros.index]
    )
    if porc_eval != 0:
        datos_forecast_mstl = datos_forecast_mstl.dropna()

    df_forecast_mstl = datos_forecast_mstl.rename(columns={'key_1': 'FECHA'}).set_index('FECHA').copy()
    df_columnas_error_mstl = crear_columnas_error(datos_forecast_mstl)
    df_mejor_mstl = agrupar_por_codigo(df_columnas_error_mstl)

    status_text.text("")
    barra_progreso.empty()
    return df_mejor_mstl, df_forecast_mstl


# ## 5. Comparacion Modelos

# In[27]:


def generar_reporte_error_skus(modelos):
    return {modelo: globals()[f'grupo_sku_error_formato_{modelo}'] for modelo in modelos}


# In[28]:


def concatenar_rmse(modelos):
    # Obtener los DataFrames dinámicamente usando la lista de modelos
    dfs_error = []
    
    for modelo in modelos:
        # Obtener el DataFrame para cada modelo
        df = globals().get(f'rmse_sku_mes_{modelo}')
        
        # Verificar si el DataFrame es None o está vacío
        if df is None or df.empty:
            print(f"El modelo {modelo} fue ignorado porque no tiene datos.")
            continue
        
        # Añadir una columna 'MODELO' con el nombre del modelo
        df['MODELO'] = modelo
        df['RMSE'] = np.ceil(df['RMSE']).astype(int)
        
        # Añadir el DataFrame a la lista
        dfs_error.append(df)
    
    # Verificar si hay DataFrames para concatenar
    if not dfs_error:
        print("No hay datos para concatenar.")
        return pd.DataFrame()  # Devuelve un DataFrame vacío
    
    # Concatenar todos los DataFrames en uno solo
    df_todos_rmse = pd.concat(dfs_error, ignore_index=True)
    
    # Asegurar que la columna 'CODIGO' sea de tipo string
    df_todos_rmse['CODIGO'] = df_todos_rmse['CODIGO'].astype(str)

    return df_todos_rmse


# def comparar_y_graficar_modelos_nv(reporte_error_skus):
#     # Crear el DataFrame base con la columna 'Codigo'
#     df_final = reporte_error_skus['pms'][['CODIGO']].copy()
#     
#     # Iterar sobre los modelos para combinarlos en df_final
#     for nombre_modelo, df in reporte_error_skus.items():
#         df_final = df_final.merge(
#             df[['CODIGO', 'SCORE%']].rename(columns={'SCORE%': nombre_modelo}), 
#             on='CODIGO', 
#             how='left'
#         )
#         df['MODELO'] = nombre_modelo
#         
#     # Remover simbolos de porcentaje y convertir columnas a valores numericos
#     modelos_cols = list(reporte_error_skus.keys())
#     df_final[modelos_cols] = df_final[modelos_cols].apply(lambda col: abs(col.str.rstrip('%').astype(float)))
#     
#     # Identificar la columna con el valor minimo para cada fila
#     df_final['MEJOR_MODELO'] = df_final[modelos_cols].idxmin(axis=1)
#     #dejar una copia sin formato porcentaje
#     df_minimos = df_final.copy()
#     # Dar formato a las columnas con un decimal y agregar el simbolo %
#     df_final[modelos_cols] = df_final[modelos_cols].apply(lambda x: x.map('{:.1f}%'.format))
#     
#     # Contar cuantas veces el modelo es el mejor
#     report = df_final['MEJOR_MODELO'].value_counts()
#     
#     # Preparar y crear la grafica de dona
#     fig1 = go.Figure(data=[go.Pie(
#         labels=report.index, 
#         values=report.values, 
#         hole=0.4,  
#         textinfo='percent+label',  
#         marker=dict(colors=px.colors.qualitative.Plotly)  
#     )])
#     
#     # Actualizar Layout de la grafica
#     fig1.update_layout(
#         title='Distribucion de Mejor Modelo por SKUs',
#         title_x=0.5,  
#         template='plotly_white'  
#     )
#    
# 
# 
#     # Concatenar todos los DataFrames en uno solo
#     df_errores_totales = pd.concat(reporte_error_skus.values(), ignore_index=True) 
#     
#     return df_minimos, df_final, reporte_error_skus, fig1, df_errores_totales

# In[62]:


def comparar_y_graficar_modelos_nv(reporte_error_skus):
    # Crear el DataFrame base con la columna 'CODIGO'
    df_final = None

    # Filtrar los DataFrames válidos (no None y no vacíos)
    reporte_error_skus_validos = {nombre: df for nombre, df in reporte_error_skus.items() if df is not None and not df.empty}

    if not reporte_error_skus_validos:
        print("No hay modelos válidos para procesar.")
        return None, None, None, None, None

    # Usar el primer DataFrame válido como base para 'CODIGO'
    for nombre_modelo, df in reporte_error_skus_validos.items():
        if df_final is None:
            df_final = df[['CODIGO']].copy()
        break

    # Iterar sobre los modelos válidos para combinarlos en df_final
    for nombre_modelo, df in reporte_error_skus_validos.items():
        df_final = df_final.merge(
            df[['CODIGO', 'SCORE%']].rename(columns={'SCORE%': nombre_modelo}),
            on='CODIGO',
            how='left'
        )
        df['MODELO'] = nombre_modelo

    # Remover símbolos de porcentaje y convertir columnas a valores numéricos
    modelos_cols = list(reporte_error_skus_validos.keys())
    df_final[modelos_cols] = df_final[modelos_cols].apply(lambda col: abs(col.str.rstrip('%').astype(float)))

    # Identificar la columna con el valor mínimo para cada fila
    df_final['MEJOR_MODELO'] = df_final[modelos_cols].idxmin(axis=1)
    # Dejar una copia sin formato porcentaje
    df_minimos = df_final.copy()
    # Dar formato a las columnas con un decimal y agregar el símbolo %
    df_final[modelos_cols] = df_final[modelos_cols].apply(lambda x: x.map('{:.1f}%'.format))

    # Contar cuántas veces el modelo es el mejor
    report = df_final['MEJOR_MODELO'].value_counts()

    # Preparar y crear la gráfica de dona
    fig1 = go.Figure(data=[go.Pie(
        labels=report.index,
        values=report.values,
        hole=0.4,
        textinfo='percent+label',
        marker=dict(colors=px.colors.qualitative.Plotly)
    )])

    # Actualizar Layout de la gráfica
    fig1.update_layout(
        title='Distribución de Mejor Modelo por SKUs',
        title_x=0.5,
        template='plotly_white'
    )

    # Concatenar todos los DataFrames válidos en uno solo
    df_errores_totales = pd.concat(reporte_error_skus_validos.values(), ignore_index=True)

    return df_minimos, df_final, reporte_error_skus_validos, fig1, df_errores_totales


# In[30]:


def generar_periodos_futuros(df_periodo, n):
    # Separar los elementos en año y número de campaña
    periodos  = df_periodo.index.unique()
    periodos_split = [p.split('-') for p in periodos]
    periodos_df = pd.DataFrame(periodos_split, columns=['AÑO', 'CAMPAÑA']).astype(int)
    
    # Identificar el año máximo
    max_año = periodos_df['AÑO'].max()
    
    # Filtrar los elementos del año máximo
    periodos_año_max = periodos_df[periodos_df['AÑO'] == max_año]
    
    # Identificar la campaña máxima dentro del año máximo
    max_campaña = periodos_año_max['CAMPAÑA'].max()
    
    # Generar el período máximo
    periodo_max = f"{max_año:04d}-{max_campaña:02d}"
    
    # Generar los períodos futuros
    futuros = []
    año_actual = max_año
    campaña_actual = max_campaña
    
    for _ in range(n):
        campaña_actual += 1
        if campaña_actual > 19:  # Reiniciar campañas después de la 19
            campaña_actual = 1
            año_actual += 1
        futuros.append(f"{año_actual:04d}-{campaña_actual:02d}")
    
    return periodo_max, futuros


# In[63]:


def agregar_fecha_a_grupo(df_todos_pronosticos, futuros):
    # Agrupar por CODIGO y MODELO
    grouped = df_todos_pronosticos.groupby(['CODIGO', 'MODELO'])
    
    # Crear una lista para almacenar los DataFrames con la columna FECHA añadida
    dfs_con_fecha = []
    
    # Iterar sobre los grupos
    for (codigo, modelo), group in grouped:
        # Asegurar que la longitud de futuros sea la misma que la del grupo
        if len(futuros) == len(group):
            # Crear la nueva columna FECHA
            group['FECHA'] = futuros
        else:
            raise ValueError(f"La longitud de 'futuros' no coincide con la longitud del grupo para CODIGO: {codigo} y MODELO: {modelo}")
        
        # Añadir el grupo modificado a la lista
        dfs_con_fecha.append(group)
    
    # Concatenar todos los DataFrames de nuevo en uno solo
    df_todos_pronosticos_fecha = pd.concat(dfs_con_fecha, ignore_index=True)
    
    return df_todos_pronosticos_fecha



# In[32]:


def concatenar_forecasts_pronosticos(modelos):
    # Filtrar los DataFrames válidos (no None y no vacíos)
    dfs_validos = [
        globals()[f'df_forecast_final_{modelo}']
        for modelo in modelos
        if globals()[f'df_forecast_final_{modelo}'] is not None and not globals()[f'df_forecast_final_{modelo}'].empty
    ]
    
    # Verificar si hay DataFrames válidos
    if not dfs_validos:
        print("No hay pronósticos válidos para concatenar.")
        return None

    # Concatenar todos los DataFrames válidos en uno solo
    df_todos_pronosticos = pd.concat(dfs_validos)

    # Asegurar que la columna 'CODIGO' sea de tipo string
    df_todos_pronosticos['CODIGO'] = df_todos_pronosticos['CODIGO'].astype(str)

    return df_todos_pronosticos


# In[73]:


def obtener_mejor_pronostico_nv(df_minimos, df_todos_pronosticos_fecha):
    # Crear una lista para almacenar los DataFrames filtrados
    lista_filtrados = [
        df_todos_pronosticos_fecha[
            (df_todos_pronosticos_fecha['CODIGO'] == row['CODIGO']) & 
            (df_todos_pronosticos_fecha['MODELO'] == row['MEJOR_MODELO'])
        ]
        for _, row in df_minimos.iterrows()
    ]
    
    # Concatenar todos los DataFrames filtrados
    df_pronosticos_mejor_modelo = pd.concat(lista_filtrados)
    
    # Pivotear el resultado para mostrar el forecast por Código, Modelo y Fecha
    df_pronosticos_12_meses = df_pronosticos_mejor_modelo.pivot_table(index=["CODIGO", "MODELO"], columns="FECHA", values="FORECAST")
    
    return df_pronosticos_mejor_modelo, df_pronosticos_12_meses


# In[75]:


def crear_grafica_pronostico_nv(df_mes, df_todos_pronosticos, df_pronosticos_mejor_modelo):
    # Obtener modelos únicos
    modelos_unicos = df_todos_pronosticos['MODELO'].unique()
    
    # Generar una paleta de colores en seaborn
    dark_colors = sns.color_palette("muted", n_colors=len(modelos_unicos)).as_hex()
    
    # Crear un diccionario para asignar colores a cada modelo
    color_mapping = {modelo: dark_colors[i] for i, modelo in enumerate(modelos_unicos)}

    # Lista de códigos únicos
    codigos_unicos = df_mes["CODIGO"].unique()

    # Crear una figura
    fig = go.Figure()

    # Crear todas las trazas (una por cada Código y Modelo) y agregar al gráfico
    for codigo in codigos_unicos:
        # Filtrar df_mes por Codigo (para graficar la demanda)
        df_mes_filtrado = df_mes[df_mes["CODIGO"] == codigo]
        
        # Filtrar df_todos_pronosticos por Codigo (para graficar todos los pronósticos de modelos)
        df_todos_pronosticos_filtrado = df_todos_pronosticos[df_todos_pronosticos["CODIGO"] == codigo]

        # Filtrar df_pronosticos_mejor_modelo para obtener el mejor modelo de ese código
        df_pronosticos_filtrado = df_pronosticos_mejor_modelo[df_pronosticos_mejor_modelo["CODIGO"] == codigo]
        mejor_modelo = df_pronosticos_filtrado["MODELO"].values[0]  # Extraer el mejor modelo para ese código

        # Agregar la traza de DEMANDA para este código (inicialmente invisible)
        fig.add_trace(go.Scatter(
            x=df_mes_filtrado.index, 
            y=df_mes_filtrado["DEMANDA"], 
            mode='lines',
            name=f'{codigo}',
            line=dict(color='navy'),
            visible=False  # Inicialmente invisible
        ))

        # Agregar una traza para cada modelo en df_todos_pronosticos_filtrado
        for modelo in modelos_unicos:
            # Filtrar por el modelo específico dentro del código seleccionado
            df_modelo_filtrado = df_todos_pronosticos_filtrado[df_todos_pronosticos_filtrado["MODELO"] == modelo]

            # Determinar el estilo de la línea
            if modelo == mejor_modelo:
                line_style = dict(dash='solid', color='#FF4500', width=2.5)  # Continua para el mejor modelo
            else:
                line_style = dict(dash='dot', color=color_mapping[modelo])  # Punteada para los demás modelos

            # Agregar la traza de FORECAST de este modelo (inicialmente invisible)
            fig.add_trace(go.Scatter(
                x=df_modelo_filtrado['FECHA'], 
                y=df_modelo_filtrado["FORECAST"], 
                mode='lines',
                name=f'{modelo}',
                line=line_style,
                visible=False  # Inicialmente invisible
            ))

    # Crear botones para el dropdown del primer menú (Códigos)
    dropdown_buttons_codigo = []
    for i, codigo in enumerate(codigos_unicos):
        # Visibilidad de DEMANDA y todas las trazas de pronósticos para este código
        visibility = [False] * len(fig.data)  # Inicializar todas las trazas como invisibles

        # Mostrar DEMANDA
        visibility[i * (len(modelos_unicos) + 1)] = True  

        # Mostrar todas las trazas de los modelos para el código seleccionado
        for j in range(len(modelos_unicos)):
            visibility[i * (len(modelos_unicos) + 1) + j + 1] = True

        # Botón para seleccionar el código
        dropdown_buttons_codigo.append(
            dict(
                args=[{"visible": visibility}],  # Cambiar la visibilidad de las trazas
                label=str(codigo),  # Etiqueta del código
                method="update"
            )
        )

    # Mostrar la primera DEMANDA y todos los modelos del primer código por defecto
    fig.data[0].visible = True
    for j in range(len(modelos_unicos)):
        fig.data[j + 1].visible = True

    # Configurar el layout con el menú dropdown
    fig.update_layout(
        template="ggplot2",
        updatemenus=[
            # Menú desplegable para seleccionar el Código
            dict(
                buttons=dropdown_buttons_codigo,
                direction="down",
                showactive=True,
                x=-0.05, y=1.2,  # Posición del dropdown
                xanchor="left",
                yanchor="top"
            )
        ],
        title="Demanda vs Pronóstico por Código",
        xaxis_title="Campaña",
        yaxis_title="Demanda",
        showlegend=True,
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        legend=dict(
            orientation="v",
            yanchor="middle",
            xanchor="left",
            x=1.05,
            y=0.5
        ),
        height=400,
        plot_bgcolor='#F0F0F0',  # Set the plot background color to a light gray
    )
    fig.update_xaxes(        
        type='category',  # Especificar que el eje X es categórico
        tickmode='array',  # Asegurar que las etiquetas del eje X no sean interpretadas como fechas
        tickangle=-45,  # Rotar las etiquetas del eje X a 45 grados
        tickfont=dict(size=9)  # Reducir el tamaño de la fuente en un 25%
    ) 
    return fig


# In[35]:


def validar_pronosticos(sku, modelo, df_todos_pronosticos):
    
    # Filtrar DataFrame basado en selección
    df_filtrado = df_todos_pronosticos[
                (df_todos_pronosticos['CODIGO'] == sku) & 
                (df_todos_pronosticos['MODELO'] == modelo)
            ]     
    df_filtrado['FORECAST'] = np.ceil(df_filtrado['FORECAST']).astype(int)  
    return df_filtrado


# ## Script de prueba

# # Ruta ubicacion de archivo fuente
# ruta_demanda = r'dataset/Consolidado_datos_5.xlsx'
# df = cargar_data_nv(ruta_demanda)
# df_orig = preprocesar_datos_1_nv(df)
# df_vertical = llenar_nan(df_orig)
# df_resultado = eliminar_ceros_iniciales_nv(df_vertical)
# df_ceros = reemplazar_ceros_nv(df_resultado)
# graficar_demanda_codigo_nv(df_resultado)
# sup = 0.98
# inf = 0.02
# n = 6
# df_periodo, df_outliers, reporte_outliers = eliminar_outliers(df_ceros, sup, inf, n)
# graficar_outliers_subplots(df_ceros, df_outliers, sup, inf, n)
# lista_skus = crear_lista_skus(df_periodo) # Crear lista de skus
# meses_a_pronosticar_evaluacion = 6 # Numero de meses a pronosticar para evaluar y seleccionar el modelo
# periodo_max_evaluacion = 12 # Numero de periodos maximos de evaluacion de cada serie de tiempo
# porc_eval = 0.35 # Porcentaje de meses para evaluar el modelo
# barra_progreso = st.progress(0)
# status_text = st.text("Iniciando Evaluación...")
# 
# df_mejor_n, df_forecast_pms = evaluar_y_generar_pms_nv(df_periodo, df_ceros, lista_skus, 
#                                                     periodo_max_evaluacion, 
#                                                     porc_eval, 
#                                                     meses_a_pronosticar_evaluacion,
#                                                    barra_progreso,
#                                                    status_text)
# grupo_mes_error_formato_pms, df_test_pms = kpi_error_lag(df_forecast_pms) # Reporte global
# grupo_sku_error_formato_pms, rmse_sku_lag_pms, rmse_sku_mes_pms = kpi_error_sku(df_forecast_pms) # Reporte por sku
# 
# # Generar Pronosticos finales con PMS
# meses_a_pronosticar_produccion = 12 # Numero de meses finales a pronosticar
# df_forecast_final_pms = construir_pronostico_pms_nv(df_mejor_n, df_periodo, meses_a_pronosticar_produccion, 'pms')
# 
# df_mejor_se,  df_forecast_se = encontrar_mejor_se_nv(df_periodo, df_ceros, lista_skus, periodo_max_evaluacion, porc_eval, 
#                         meses_a_pronosticar_evaluacion,
#                         barra_progreso,
#                         status_text)
# grupo_mes_error_formato_se, df_test_se = kpi_error_lag(df_forecast_se) # Reporte global
# grupo_sku_error_formato_se, rmse_sku_lag_se, rmse_sku_mes_se = kpi_error_sku(df_forecast_se)
# 
# porc_eval_pronost = 0
# meses_a_pronosticar_produccion = 12
# df_mejor_se_final,  df_forecast_final_se = encontrar_mejor_se_nv(df_periodo, df_ceros, lista_skus, periodo_max_evaluacion, porc_eval_pronost, 
#                         meses_a_pronosticar_produccion,
#                         barra_progreso,
#                         status_text)
# # Adicionar nombre a los pronosticos de SE
# df_forecast_final_se = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_se, 'se')
# 
# porc_eval = 0.35
# df_mejor_rl_lineal, df_mejor_rl_estacional, df_forecast_rl_lineal, df_forecast_rl_estacional = aplicar_regresion_lineal_simple_nv(lista_skus, df_periodo, df_ceros, 
#                                     periodo_max_evaluacion, porc_eval, 
#                                     meses_a_pronosticar_evaluacion,
#                                     barra_progreso,
#                                     status_text)
# 
# grupo_mes_error_formato_rl_lineal, df_test_rl_lineal = kpi_error_lag(df_forecast_rl_lineal) # Reporte global
# grupo_sku_error_formato_rl_lineal, rmse_sku_lag_rl_lineal, rmse_sku_mes_rl_lineal = kpi_error_sku(df_forecast_rl_lineal)
# 
# grupo_mes_error_formato_rl_estacional, df_test_rl_estacional = kpi_error_lag(df_forecast_rl_estacional) # Reporte global
# grupo_sku_error_formato_rl_estacional, rmse_sku_lag_rl_estacional, rmse_sku_mes_rl_estacional = kpi_error_sku(df_forecast_rl_estacional)
# 
# 
# df_final_mejor_rl_lineal, df_final_mejor_rl_estacional, df_forecast_final_rl_lineal, df_forecast_final_rl_estacional = aplicar_regresion_lineal_simple_nv(lista_skus, df_periodo, df_ceros, 
#                                     periodo_max_evaluacion, porc_eval_pronost, 
#                                     meses_a_pronosticar_produccion,
#                                     barra_progreso,
#                                     status_text)
# 
# # Adicionar nombre a los pronosticos de RL
# df_forecast_final_rl_lineal = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_rl_lineal, 'rl_lineal')
# df_forecast_final_rl_estacional = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_rl_estacional, 'rl_estacional')
# 
# # Modelo de descomposicion MSTL
# peso_ult_data = 0.08
# df_mejor_mstl, df_forecast_mstl = aplicar_mstl_nv(lista_skus, df_periodo, df_ceros, 
#                                     periodo_max_evaluacion, porc_eval, 
#                                     meses_a_pronosticar_evaluacion, peso_ult_data,
#                                     barra_progreso,
#                                     status_text)
#  # Reportes de error MSTL
# grupo_mes_error_formato_mstl, df_test_mstl = kpi_error_lag(df_forecast_mstl) # Reporte golbal
# grupo_sku_error_formato_mstl, rmse_sku_lag_mstl, rmse_sku_mes_mstl = kpi_error_sku(df_forecast_mstl) # Reporte por sku
# 
# # Generar Pronosticos finales con MSTL
# tabla_final_pronost, df_forecast_final_mstl = aplicar_mstl_nv(lista_skus, df_periodo, df_ceros, 
#                                     periodo_max_evaluacion, porc_eval_pronost, 
#                                     meses_a_pronosticar_produccion, 
#                                     peso_ult_data,
#                                     barra_progreso,
#                                     status_text)                      
# 
# # Adicionar nombre a los pronosticos de MSTL                                                         
# df_forecast_final_mstl = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_mstl, 'mstl')
# modelos = ['pms', 'se', 'rl_lineal', 'rl_estacional', 'mstl']
# reporte_error_skus = generar_reporte_error_skus(modelos)
# df_todos_rmse = concatenar_rmse(modelos)
# df_minimos, df_final, reporte_error_skus, fig1, df_errores_totales = comparar_y_graficar_modelos_nv(reporte_error_skus)
# fig1.show()
# n = 12
# periodo_max, futuros = generar_periodos_futuros(df_periodo, n)
# df_todos_pronosticos = concatenar_forecasts_pronosticos(modelos)
# df_todos_pronosticos_fecha = agregar_fecha_a_grupo(df_todos_pronosticos, futuros)
# df_pronosticos_mejor_modelo, df_pronosticos_12_meses = obtener_mejor_pronostico_nv(df_minimos, df_todos_pronosticos_fecha )
# fig = crear_grafica_pronostico_nv(df_periodo, df_todos_pronosticos_fecha, df_pronosticos_mejor_modelo)
# fig.show()
# 

# # Front end streamlit

# In[37]:


# Configurar el layout de Streamlit
st.set_page_config(layout="wide")

# Título de la aplicación
st.title("Pronósticos por campaña MILAGROS para NOVAVENTA")

# Variables globales
session_vars = ['df', 'df_vertical', 'df_mes_cliente', 'df_mes_orig', 
                'sup', 'inf', 'n',
                'meses_a_pronosticar_evaluacion', 'meses_a_pronosticar_produccion', 'periodo_max_evaluacion',
                'porc_eval', 'porc_eval_pronost', 'df_resultado',
                'df_ceros', 'df_periodo', 'df_outliers', 'df_todos_pronosticos', 'df_todos_pronosticos_fecha',
                'codigo_seleccionado', 'modelo_seleccionado', 'mostrar_grafica', 
                'df_pronosticos_12_meses', 'reporte_outliers', 'fig', 'mostrar_grafica_outliers'
               ]
# Inicializar session_state si no existe
for var in session_vars:
    if var not in st.session_state:
        st.session_state[var] = None

tabs = st.tabs(['📂 Cargar datos Novaventa', 
                '📊 Outliers Novaventa',  
                '🔮 Evaluar y generar pronósticos Novaventa',
                '🛠️ Herramientas de Análisis Novaventa'
               ])

with tabs[0]:
    # Estado inicial de la variable de gráfica
    if st.session_state.mostrar_grafica is None:
        st.session_state.mostrar_grafica = "ninguna"
    
    # Subida y procesamiento de datos
    st.header("Cargar Datos")
    
    if st.session_state.df is not None:
        st.success('Datos ya cargados previamente')
        st.write("Datos cargados:")
        st.write(st.session_state.df.head())
        if st.button('Cargar Nuevos Datos'):
            for var in session_vars:
                st.session_state[var] = None
            st.experimental_rerun()
    else:
        ruta_demanda = st.file_uploader("Sube el archivo de demanda en formato Excel", type=['xlsx'])
        if ruta_demanda is not None:        
            df = cargar_data_nv(ruta_demanda)
            st.success("Archivo histórico cargado correctamente.")
            st.session_state.df = df
            st.write("Datos cargados:")
            st.write(st.session_state.df.head())
            df_orig_nv = preprocesar_datos_1_nv(df)
            df_vertical_nv = llenar_nan(df_orig_nv)
            st.session_state.df_resultado = eliminar_ceros_iniciales_nv(df_vertical_nv)
            st.session_state.df_ceros = reemplazar_ceros_nv(st.session_state.df_resultado)
            st.success("Datos preprocesados correctamente.")
    
    # Opciones de gráficas
    if st.session_state.df_ceros is not None:
        st.header("Ver Gráficas de Demanda por Campaña")
        col1, col2, col3 = st.columns(3)
        
        with col1:
           if st.button("Graficar demanda original con ceros"):
                st.session_state.mostrar_grafica = "con_ceros"
        with col2:        
            if st.button("Graficar demanda sin ceros"):            
                st.session_state.mostrar_grafica = "sin_ceros"
        with col3:
            if st.button("Cerrar gráfica"):
                st.session_state.mostrar_grafica = "ninguna"
                
        # Mostrar las gráficas según la selección
        if st.session_state.mostrar_grafica == "con_ceros":
            st.subheader("Gráfica: Demanda Original con Ceros")
            graficar_demanda_codigo_nv(st.session_state.df_resultado)
        elif st.session_state.mostrar_grafica == "sin_ceros":
            st.subheader("Gráfica: Demanda Sin Ceros")
            graficar_demanda_codigo_nv(st.session_state.df_ceros)

        with tabs[1]:
            # Parámetros de configuración
            if "df_ceros" in st.session_state and st.session_state.df_ceros is not None:
                st.header("Manejo de Outliers")
                sup = st.number_input("Límite Superior", min_value=0.0, max_value=1.0, value=0.98, step=0.01)
                inf = st.number_input("Límite Inferior", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
                n = st.number_input("Número de Periodos para Pronóstico Ingenuo", min_value=1, max_value=12, value=6, step=1)
                df_periodo, df_outliers, reporte_outliers = eliminar_outliers(st.session_state.df_ceros, sup, inf, n)
                st.session_state.df_periodo = df_periodo
                st.session_state.df_outliers = df_outliers
                st.session_state.reporte_outliers = reporte_outliers
                st.session_state.sup = sup
                st.session_state.inf = inf
                st.session_state.n = n  
                st.success("Outliers Imputados correctamente.")
                    
            if "df_periodo" in st.session_state and "df_outliers" in st.session_state:
            
                # Exportar a excel outliers
                if df_outliers is not None:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        st.session_state.reporte_outliers.to_excel(writer, index=False, sheet_name='Outliers')
                        
                    excel_data = output.getvalue()
                else:
                    st.warning("No hay datos procesados aun para exportar.")
                # Boton de descarga a excel
                st.download_button(
                    label="📥 Descargar Outliers (Excel)",
                    data=excel_data,
                    file_name="df_outliers.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.header("Generar Gráfica Manejo Ouliers")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Mostrar gráfica de outliers"):
                        st.session_state.mostrar_grafica_outliers = "mostrar"
                with col2:
                    if st.button("Cerrar gráfica de outliers"):
                        st.session_state.mostrar_grafica_outliers = "cerrar"
            
                # Mostrar o cerrar gráfica
                if st.session_state.mostrar_grafica_outliers == "mostrar":
                    st.subheader("Gráfica: Imputación de Outliers")
                    graficar_outliers_subplots(st.session_state.df_ceros, 
                                               st.session_state.df_outliers, 
                                               sup=st.session_state.sup, 
                                               inf=st.session_state.inf, 
                                               n=st.session_state.n)
        with tabs[2]:
             # Chequear si los pronosticos ya han sido generados
            if st.session_state.df_pronosticos_12_meses is not None and st.session_state.fig is not None and st.session_state.df_todos_pronosticos is not None:
                st.success("Pronósticos ya generados previamente.")
                # Boton para regenerar pronosticos
                if st.button('Regenerar Pronósticos'):
                    # Limpiar los pronosticos previos y volver a correr script de la seccion
                    st.session_state.df_pronosticos_12_meses = None
                    st.session_state.fig = None
                    st.session_state.df_todos_pronosticos = None
                    st.experimental_rerun()  # Volver a correr el script
                else: 
                    st.dataframe(st.session_state.df_pronosticos_12_meses)
                    st.plotly_chart(st.session_state.fig)
            
            else:    
                if "df_periodo" in st.session_state and "df_ceros" in st.session_state:
                    st.header("Parámetros de evaluación de los modelos")
                    meses_a_pronosticar_evaluacion = st.number_input("Meses a Pronosticar para Evaluación", 
                                                                     min_value=1, max_value=24, value=6, step=1)
                    meses_a_pronosticar_produccion = st.number_input("Meses a Pronosticar para Produccion", 
                                                                     min_value=1, max_value=24, value=12, step=1)
                    periodo_max_evaluacion = 12
                    porc_eval = st.number_input("Porcentaje de Evaluación", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
                    porc_eval_pronost = 0
                
                    st.session_state.meses_a_pronosticar_evaluacion = meses_a_pronosticar_evaluacion
                    st.session_state.meses_a_pronosticar_produccion = meses_a_pronosticar_produccion
                    st.session_state.periodo_max_evaluacion = periodo_max_evaluacion
                    st.session_state.porc_eval = porc_eval
                    st.session_state.porc_eval_pronost = porc_eval_pronost
            
                # Evaluar y generar pronósticos
                if "df_periodo" in st.session_state and "df_ceros" in st.session_state:
                    st.header("Evaluar y Generar Pronósticos")
                    
                    if st.button("Evaluar y Generar"):
                              
                        st.session_state.pronosticos_generados = True
                        lista_skus = crear_lista_skus(st.session_state.df_periodo)
                        barra_progreso = st.progress(0)
                        status_text = st.text("Iniciando Evaluación PMS...")
                        with st.spinner('Evaluando Promedio Móvil Simple:'):
                            df_mejor_n, df_forecast_pms = evaluar_y_generar_pms_nv(st.session_state.df_periodo,
                                                                                st.session_state.df_ceros, 
                                                                                lista_skus, 
                                                                                st.session_state.periodo_max_evaluacion, 
                                                                                st.session_state.porc_eval, 
                                                                                st.session_state.meses_a_pronosticar_evaluacion,
                                                                               barra_progreso,
                                                                               status_text)
                            
                            grupo_mes_error_formato_pms, df_test_pms = kpi_error_lag(df_forecast_pms) # Reporte global
                            grupo_sku_error_formato_pms, rmse_sku_lag_pms, rmse_sku_mes_pms = kpi_error_sku(df_forecast_pms) 
                            
                        # Generar pronósticos finales
                        with st.spinner('Generando Promedio Móvil Simple:'):
                            df_forecast_final_pms = construir_pronostico_pms_nv(df_mejor_n, 
                                                                             st.session_state.df_periodo,
                                                                             st.session_state.meses_a_pronosticar_produccion, 
                                                                             'pms')
                        barra_progreso = st.empty()
                        status_text = st.empty()
                        barra_progreso = st.progress(0)
                        status_text = st.text("Iniciando Evaluación SE...") 
                        with st.spinner('Evaluando Suavizacion Exponencial Simple:'):
                            df_mejor_se,  df_forecast_se = encontrar_mejor_se_nv(st.session_state.df_periodo,
                                                                              st.session_state.df_ceros, 
                                                                              lista_skus, 
                                                                              st.session_state.periodo_max_evaluacion, 
                                                                              st.session_state.porc_eval, 
                                                                              st.session_state.meses_a_pronosticar_evaluacion,
                                                                               barra_progreso,
                                                                               status_text)
                            
                            grupo_mes_error_formato_se, df_test_se = kpi_error_lag(df_forecast_se) 
                            grupo_sku_error_formato_se, rmse_sku_lag_se, rmse_sku_mes_se = kpi_error_sku(df_forecast_se)
                        
                        barra_progreso = st.empty()
                        status_text = st.empty()                
                        barra_progreso = st.progress(0)
                        status_text = st.text("Pronosticando con SE...")     
                        with st.spinner('Generando Suavización Exponencial:'):
                            
                            df_mejor_se_final,  df_forecast_final_se = encontrar_mejor_se_nv(st.session_state.df_periodo, 
                                                                                          st.session_state.df_ceros, 
                                                                                          lista_skus, 
                                                                                          st.session_state.periodo_max_evaluacion,
                                                                                          st.session_state.porc_eval_pronost,                                                                           
                                                                                          st.session_state.meses_a_pronosticar_produccion,
                                                                                          barra_progreso,
                                                                                          status_text)
                            
                            df_forecast_final_se = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_se, 'se')
                        
                        barra_progreso = st.empty()
                        status_text = st.empty()   
                        barra_progreso = st.progress(0)
                        status_text = st.text("Iniciando Evaluación RL...") 
                        with st.spinner('Evaluando Regresion Lineal Simple y "Estacional":'):
                            df_mejor_rl_lineal, df_mejor_rl_estacional, df_forecast_rl_lineal, df_forecast_rl_estacional = aplicar_regresion_lineal_simple_nv(lista_skus, 
                                                                                                                                                           st.session_state.df_periodo, 
                                                                                                                                                           st.session_state.df_ceros,
                                                                                                                                                           st.session_state.periodo_max_evaluacion,
                                                                                                                                                           st.session_state.porc_eval, 
                                                                                                                                                           st.session_state.meses_a_pronosticar_evaluacion,
                                                                                                                                                            barra_progreso,
                                                                                                                                                           status_text)
            
                            grupo_mes_error_formato_rl_lineal, df_test_rl_lineal= kpi_error_lag(df_forecast_rl_lineal) # Reporte global RL simple
                            grupo_sku_error_formato_rl_lineal, rmse_sku_lag_rl_lineal, rmse_sku_mes_rl_lineal = kpi_error_sku(df_forecast_rl_lineal) # Reporte por sku RL simple
                            grupo_mes_error_formato_rl_estacional, df_test_rl_estacional= kpi_error_lag(df_forecast_rl_estacional) # Reporte global RL estacional
                            grupo_sku_error_formato_rl_estacional, rmse_sku_lag_rl_estacional, rmse_sku_mes_rl_estacional = kpi_error_sku(df_forecast_rl_estacional) # Reporte por sku RL estaciona
            
                        barra_progreso = st.empty()
                        status_text = st.empty() 
                        barra_progreso = st.progress(0)
                        status_text = st.text("Pronosticando con RL...") 
                        with st.spinner('Generando Regresion Lineal Simple y "Estacional":'):
                            df_final_mejor_rl_lineal, df_final_mejor_rl_estacional, df_forecast_final_rl_lineal, df_forecast_final_rl_estacional = aplicar_regresion_lineal_simple_nv(lista_skus, 
                                                                                                                                                                                    st.session_state.df_periodo, 
                                                                                                                                                                                    st.session_state.df_ceros,
                                                                                                                                                                                    st.session_state.periodo_max_evaluacion, 
                                                                                                                                                                                    st.session_state.porc_eval_pronost, 
                                                                                                                                                                                    st.session_state.meses_a_pronosticar_produccion,
                                                                                                                                                                                    barra_progreso,
                                                                                                                                                                                    status_text)
                            
                            df_forecast_final_rl_lineal = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_rl_lineal, 'rl_lineal')
                            df_forecast_final_rl_estacional = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_rl_estacional, 'rl_estacional')
                        barra_progreso = st.empty()
                        status_text = st.empty() 
                        barra_progreso = st.progress(0)
                        status_text = st.text("Iniciando Evaluación MSTL...") 
                        with st.spinner('Evaluando MSTL:'):
                            peso_ult_data = 0.08 
                            df_mejor_mstl, df_forecast_mstl = aplicar_mstl_nv(lista_skus, st.session_state.df_periodo, 
                                                                           st.session_state.df_ceros, 
                                                                           st.session_state.periodo_max_evaluacion, 
                                                                           st.session_state.porc_eval, 
                                                                           st.session_state.meses_a_pronosticar_evaluacion, 
                                                                           peso_ult_data, 
                                                                           barra_progreso, 
                                                                           status_text)
                            
                            grupo_mes_error_formato_mstl, df_test_mstl = kpi_error_lag(df_forecast_mstl) # Reporte golbal
                            grupo_sku_error_formato_mstl, rmse_sku_lag_mstl, rmse_sku_mes_mstl = kpi_error_sku(df_forecast_mstl) # Reporte por sku
                        barra_progreso = st.empty()
                        status_text = st.empty() 
                        barra_progreso = st.progress(0)
                        status_text = st.text("Pronosticando con MSTL...") 
                        with st.spinner('Generando MSTL:'):
            
                            tabla_final_pronost, df_forecast_final_mstl = aplicar_mstl_nv(lista_skus, 
                                                                                       st.session_state.df_periodo, 
                                                                                       st.session_state.df_ceros, 
                                                                                       st.session_state.periodo_max_evaluacion, 
                                                                                       st.session_state.porc_eval_pronost, 
                                                                                       st.session_state.meses_a_pronosticar_produccion, 
                                                                                       peso_ult_data, 
                                                                                       barra_progreso, 
                                                                                       status_text)
                            
                            df_forecast_final_mstl = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_mstl, 'mstl')
                        
                        barra_progreso = st.empty()
                        status_text = st.empty()  
            
                        st.success("✅✨ ¡Modelos Calculados Correctamente! 🎯🚀")
                        st.balloons()
                        
                        modelos = ['pms', 'se', 'rl_lineal', 'rl_estacional', 'mstl']
                        
                        reporte_error_skus = generar_reporte_error_skus(modelos)
                        df_todos_rmse = concatenar_rmse(modelos)
                        df_minimos, df_final, reporte_error_skus, fig1, df_errores_totales = comparar_y_graficar_modelos_nv(reporte_error_skus)
                        with st.expander("Mostrar Estadisticas de Modelos de Pronosticos"):
                            st.plotly_chart(fig1)
                        periodo_max, futuros = generar_periodos_futuros(df_periodo, st.session_state.meses_a_pronosticar_produccion)            
                        df_todos_pronosticos = concatenar_forecasts_pronosticos(modelos)
                        df_todos_pronosticos_fecha = agregar_fecha_a_grupo(df_todos_pronosticos, futuros)
                        st.session_state.df_todos_pronosticos = df_todos_pronosticos
                        st.session_state.df_todos_pronosticos_fecha  = df_todos_pronosticos_fecha 
                        
                        df_pronosticos_mejor_modelo, df_pronosticos_12_meses = obtener_mejor_pronostico_nv(df_minimos, 
                                                                                                        st.session_state.df_todos_pronosticos_fecha, 
                                                                                                        #df_errores_totales, 
                                                                                                        #df_todos_rmse
                                                                                                       )
                                                                                                       
                        st.session_state.df_pronosticos_12_meses = df_pronosticos_12_meses
                        # Mostrar resultados
                        st.subheader("Pronósticos proximas campañas")
                        st.write(df_pronosticos_12_meses)
                
                        # Mostrar gráfica final
                        fig = crear_grafica_pronostico_nv(st.session_state.df_periodo, st.session_state.df_todos_pronosticos_fecha, df_pronosticos_mejor_modelo)
                        st.session_state.fig = fig
                        st.plotly_chart(fig)

        with tabs[3]:
            # Sección de filtrado (solo se muestra si df_todos_pronosticos existe)
            if 'df_todos_pronosticos_fecha' in st.session_state and st.session_state.df_todos_pronosticos_fecha is not None:
                st.header("Filtrar Pronósticos")
                st.text('Desea usar otro pronóstico diferente al sugerido estadísticamente?')
                
                # Menú desplegable para seleccionar el código
                codigo_seleccionado = st.selectbox(
                    "Seleccione el Código:",
                    options=st.session_state.df_todos_pronosticos_fecha['CODIGO'].unique()
                )
                
                # Menú desplegable para seleccionar el modelo
                modelo_seleccionado = st.selectbox(
                    "Seleccione el Modelo:",
                    options=st.session_state.df_todos_pronosticos_fecha['MODELO'].unique()
                )
                if st.button('Validar Series de Tiempo'):
                    df_filtrado = validar_pronosticos(codigo_seleccionado, 
                                                  modelo_seleccionado, 
                                                  st.session_state.df_todos_pronosticos_fecha)
                    
                    st.write('Datos de pronostico para codigo y modelo seleccionado:')
                    st.dataframe(df_filtrado)

