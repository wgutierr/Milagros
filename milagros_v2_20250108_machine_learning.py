#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print(f"Streamlit está usando Python desde: {sys.executable}")
# Funciones Generales de Analisis de datos
import pandas as pd
import numpy as np

# Funciones para graficas
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Funciones estadisticas y matematicas
from scipy.stats import norm
import math

# Libreria Statsmodels
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.seasonal import MSTL

# Libreria de Scikitlearn
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

# Funciones de Skforecast
#from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.recursive import ForecasterRecursive
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster
#from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries
from skforecast.model_selection import backtesting_forecaster_multiseries
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection import bayesian_search_forecaster_multiseries
from skforecast.preprocessing import RollingFeatures

# Algoritmos de Machine Learning
import xgboost as xgb 
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Funciones de Calendario y Tiempo
import holidays
import time
from datetime import datetime
from pandas.tseries.offsets import DateOffset
from dateutil.relativedelta import relativedelta

# Barra de progreso
from tqdm import tqdm 
#from IPython.display import clear_output # Complemento barra de progreso

# Manejo de advertencias
import warnings

# Libreria de Streamlit para front-end
import streamlit as st

# Funcion para exportar a excel
import io


# # Cargar Datos Historicos

# In[2]:


# Función de carga pronosticos por mes
def cargar_data(ruta):
    df = pd.read_excel(ruta)
    return df

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


# # Preprocesar Datos Parte 1

# In[3]:


# Convierte los datos originales a una matriz vertical pronostico en meses
def convertir_a_df_vertical(df):
    # Cambiar nombres de columnas
    columnas_fechas = [col for col in df.columns if col not in ['CODIGO', 'DESCRIPCION', 'CLIENTE']]
    #df = df.rename(columns={'CÓDIGO': 'CODIGO', 'REFERENCIA':'CLIENTE'})
    #columnas_fechas = df.columns[2:]    
    # Realizar el melt para transformar el DataFrame en formato largo
    df_vertical = df.melt(
        id_vars=['CODIGO','DESCRIPCION', 'CLIENTE'], 
        value_vars=columnas_fechas,
        var_name='FECHA', 
        value_name='DEMANDA'
    )
    return df_vertical
    
# Solo para Novaventa
def preprocesar_datos_1_nv(df):
    df['Periodo_Campaña'] = df['Año'].astype(str) + '-' + df['Campaña'].astype(str).str.zfill(2)
    df_orig = df[['Periodo_Campaña','Referencia Novaventa',	'Descripción producto','Unds Brutas']].copy()
    df_orig = df_orig.rename(columns={'Referencia Novaventa':'CODIGO','Unds Brutas':'DEMANDA','Periodo_Campaña':'FECHA','Descripción producto':'DESCRIPCION'})
    
    return df_orig

# Solo para Novaventa
def llenar_nan(df_orig):
    df_horiz = df_orig.pivot_table(index=['CODIGO','DESCRIPCION'], columns='FECHA', values='DEMANDA', fill_value=0, observed=True)
    # Reset the index of the pivot table so 'CODIGO' becomes a column again
    df_reset = df_horiz.reset_index()
        # Melt the DataFrame to convert it back to a vertical format
    df_vertical = df_reset.melt(id_vars=['CODIGO','DESCRIPCION'], var_name='FECHA', value_name='DEMANDA')
    df_vertical['CODIGO'] = df_vertical['CODIGO'].astype('str')
    return df_vertical


# ## Mapear nombre de los meses

# In[4]:


# Mapear los nombres de las columnas a fechas en formato 'YYYY-MM-DD'
meses = {
    "ENE": "01", "FEB": "02", "MAR": "03", "ABR": "04", "MAY": "05", 
    "JUN": "06", "JUL": "07", "JUl":"07","AGO": "08", "SEPT": "09", "OCT": "10", 
    "NOV": "11", "DIC": "12", 
}


# ## Convertir Texto a Fecha

# In[5]:


# Para pronostico por meses
def convertir_texto_a_fecha(df_vertical, meses):
    # Normalizar texto en la columna 'FECHA'
    df_vertical['FECHA'] = df_vertical['FECHA'].str.upper().str.strip()

    # Extraer mes y año usando regex
    extract = df_vertical['FECHA'].str.extract(r'([A-Z]+) (\d{4})')
    extract.columns = ['mes', 'año']  # Renombrar columnas para claridad

    # Manejar valores no coincidentes
    if extract.isnull().any().any():
        no_validos = df_vertical.loc[extract.isnull().any(axis=1), 'FECHA'].unique()
        raise ValueError(f"Valores no coincidentes en 'FECHA': {no_validos}")

    # Formatear y convertir a fecha
    df_vertical['FECHA'] = pd.to_datetime(
        extract.apply(lambda x: f"{x['año']}-{meses.get(x['mes'], '01')}-01", axis=1)
    )
    
    # Establecer 'FECHA' como índice
    df_vertical_fecha = df_vertical.set_index('FECHA')
    
    return df_vertical_fecha


# ## Eliminar Ceros iniciales en las Series de Tiempo

# In[6]:


def eliminar_ceros_iniciales(df):
    # Lista para almacenar DataFrames válidos
    lista_df = []

    # Obtener códigos únicos (SKU-cliente)
    codigos_unicos = df['CODIGO'].unique()
    clientes_unicos = df['CLIENTE'].unique()
    
    for codigo in codigos_unicos:
        for cliente in clientes_unicos:
            
            # Filtrar los datos para cada código y cliente
            df_codigo = df[(df['CODIGO'] == codigo) & (df['CLIENTE'] == cliente)]

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

# Solo para novaventa
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


# # Preprocesar Datos Parte 2

# In[7]:


def preprocesar_tabla_2(df_resultado):
    
    # Crear columna CODIGO_CLIENTE concatenado ambas columnas
    df_resultado['CODIGO_CLIENTE'] = df_resultado['CODIGO'] + "_" + df_resultado['CLIENTE']
    
    # Seleccionar y copiar las columnas relevantes
    df_mes_cliente = df_resultado[['CODIGO_CLIENTE', 'CODIGO', 'DESCRIPCION','CLIENTE', 'DEMANDA']].copy()
    
    return df_mes_cliente


# ## Grafica Demanda Original por Cliente

# In[8]:


def graficar_demanda_codigo_cliente(df_mes_cliente):
    
    # Obtener los códigos únicos
    codigos_unicos = df_mes_cliente['CODIGO'].unique()
    descripciones_unicas = df_mes_cliente['DESCRIPCION'].unique()
    # Crear la figura de subplots
    fig = make_subplots(
        rows=(len(codigos_unicos) + 2) // 3,  # Para distribuir los subplots en 3 columnas
        cols=3,
        shared_yaxes=False,  # No Compartir el eje Y
        subplot_titles = [
            f"Código: {codigo}<br><span style='font-size:10px;'>{descripcion}</span>"
            for codigo, descripcion in zip(codigos_unicos, descripciones_unicas)
        ],
        vertical_spacing=0.025,  # Reducir el espaciado vertical entre los subplots
    )
    
    # Colores específicos para clientes
    cliente_colores = {
                'NOVAVENTA': '#FFA500',  # Naranja
                'DISTRIBUIDORES': "#4682B4",  # Azul Acero
                'FARMATODO': '#8B0000',  # Rojo oscuro
                'E-COMMERCE': '#77DD77',  # Verde pastel
            }
    
    # Asignar un color único para los otros clientes, si los hay
    clientes_unicos = df_mes_cliente['CLIENTE'].unique()
    
    for i, cliente in enumerate(clientes_unicos):
        if cliente not in cliente_colores:
            # Asignar un color diferente para los otros clientes
            cliente_colores[cliente] = f"rgb({(i * 50) % 256}, {(i * 100) % 256}, {(i * 150) % 256})"
    
    # Iterar sobre cada código para agregar los subplots
    for i, codigo in enumerate(codigos_unicos, start=1):
        # Filtrar los datos para el código actual
        df_codigo = df_mes_cliente[df_mes_cliente['CODIGO'] == codigo]
        
        # Iterar sobre los clientes y agregar una traza por cliente
        for cliente in clientes_unicos:
            df_cliente = df_codigo[df_codigo['CLIENTE'] == cliente]
            fig.add_trace(
                go.Scatter(
                    x=df_cliente.index,  # Usar el índice de fecha
                    y=df_cliente['DEMANDA'],
                    mode='lines',
                    name=cliente,
                    line=dict(color=cliente_colores[cliente]),  # Asignar color específico por cliente
                    hovertemplate=(
                        "<b>Fecha:</b> %{x}<br>"
                        "<b>Demanda:</b> %{y}<br>"                        
                        f"<b>Descripción:</b> {df_codigo['DESCRIPCION'].iloc[0]}<br>"
                        #"<extra></extra>"  # Ocultar información adicional de Plotly
                    )
                ),
                row=(i - 1) // 3 + 1,  # Fila en la que estará el subplot
                col=(i - 1) % 3 + 1,   # Columna en la que estará el subplot
            )
    
    # Ajustar la altura total para todos los subplots y actualizar la disposición
    fig.update_layout(
        height=220 * ((len(codigos_unicos) + 2) // 3),  # Aumentar la altura total de los subplots
        title_text="Demanda por Código y Cliente",
        title_x=0.5,  # Centrar el título
        title_font=dict(size=14),  # Tamaño de fuente del título principal
        showlegend=False,  # Eliminar la leyenda
        font=dict(size=10),  # Tamaño de fuente general
        margin=dict(l=50, r=50, t=80, b=50),  # Ajustar márgenes
        template="ggplot2",
    )
    # Ajustar los títulos de los subplots
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=11)  # Reducir tamaño del texto de los títulos de subplots
        
    # Ajustar los títulos de los subplots y ejes
    fig.update_xaxes(title_font=dict(size=9))  # Tamaño de fuente para los títulos del eje X
    fig.update_yaxes(title_font=dict(size=9))  # Tamaño de fuente para los títulos del eje Y
    
    # Mostrar la figura
    #fig.show()
    st.plotly_chart(fig, use_container_width=True)


# ## Consolidar Demanda por Codigo

# In[9]:


def agrupar_demanda(df_mes_cliente):
    # Agrupar por FECHA y CÓDIGO, sumar DEMANDA
    df_mes_orig = (
        df_mes_cliente
        .groupby(['FECHA', 'CODIGO','DESCRIPCION'])
        .agg({'DEMANDA': 'sum'})
        .reset_index()
        .set_index('FECHA')
    )
    
    # Calcular el largo de cada serie de tiempo por CODIGO
    series_length = df_mes_orig.groupby('CODIGO').size()
    
    # Filtrar los códigos con menos de 2 registros
    codigos_cortos = series_length[series_length < 2].index.tolist()
    
    # Generar un reporte de estos códigos
    reporte_codigos = df_mes_orig[df_mes_orig['CODIGO'].isin(codigos_cortos)]
    
    # Eliminar los códigos con menos de 2 registros del DataFrame original
    df_mes_orig = df_mes_orig[~df_mes_orig['CODIGO'].isin(codigos_cortos)]

    print('No se pronosticaran las siguientes referencias debido a que tienen muy pocos datos:')
    print(reporte_codigos.groupby('CODIGO').size())
    
    return df_mes_orig, reporte_codigos


# ## Grafica Consolidada por Codigo

# In[10]:


# Para pronostico por meses
def graficar_demanda_codigo(df_mes_orig):
    # Obtener los códigos únicos
    codigos_unicos = df_mes_orig['CODIGO'].unique()
    descripciones_unicas = df_mes_orig['DESCRIPCION'].unique()
    # Crear la figura de subplots
    fig = make_subplots(
        rows=(len(codigos_unicos) + 2) // 3,  # Distribuir en 3 columnas
        cols=3,
        shared_yaxes=False,  # No compartir el eje Y
        #subplot_titles=[f"Código: {codigo}" for codigo in codigos_unicos],
        subplot_titles = [
            f"Código: {codigo}<br><span style='font-size:10px;'>{descripcion}</span>"
            for codigo, descripcion in zip(codigos_unicos, descripciones_unicas)
        ],
        vertical_spacing=0.02,  # Reducir el espaciado entre subplots
    )
    
    # Iterar sobre cada código para agregar subplots
    for i, codigo in enumerate(codigos_unicos, start=1):
        
        # Filtrar los datos para el código actual
        df_codigo = df_mes_orig[df_mes_orig['CODIGO'] == codigo]
        
        # Agregar una traza al subplot
        fig.add_trace(
            go.Scatter(
                x=df_codigo.index,  # Usar el índice de fecha
                y=df_codigo['DEMANDA'],
                mode='lines',
                name=codigo,
                line=dict(width=2, color = "#4682B4"),  # Personalizar el ancho de la línea
                hovertemplate=(
                "<b>Fecha:</b> %{x}<br>"
                "<b>Demanda:</b> %{y}<br>"
                f"<b>Descripción:</b> {df_codigo['DESCRIPCION'].iloc[0]}<br>"
                "<extra></extra>"
                ),
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
        margin=dict(l=50, r=50, t=80, b=50),  # Ajustar márgenes
        template="ggplot2",
    )
    # Ajustar los títulos de los subplots
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=11)  # Reducir tamaño del texto de los títulos de subplots
        
     # Ajustar los títulos de los subplots y ejes
    fig.update_xaxes(title_font=dict(size=9))  # Tamaño de fuente para los títulos del eje X
    fig.update_yaxes(title_font=dict(size=9))  # Tamaño de fuente para los títulos del eje Y
    
    # Mostrar la figura
    #fig.show()
    st.plotly_chart(fig, use_container_width=True)

# Solo para pronosticos de novaventa
def graficar_demanda_codigo_nv(df_mes_orig):
    # Obtener los códigos únicos
    codigos_unicos = df_mes_orig['CODIGO'].unique()
    descripciones_unicas = df_mes_orig['DESCRIPCION'].unique()
    # Crear la figura de subplots
    fig = make_subplots(
        rows=(len(codigos_unicos) + 2) // 3,  # Distribuir en 3 columnas
        cols=3,
        shared_yaxes=False,  # No compartir el eje Y
        #subplot_titles=[f"Código: {codigo}" for codigo in codigos_unicos],
        subplot_titles = [
            f"Código: {codigo}<br><span style='font-size:10px;'>{descripcion}</span>"
            for codigo, descripcion in zip(codigos_unicos, descripciones_unicas)
        ],
        vertical_spacing=0.08,  # Reducir el espaciado entre subplots
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
                name=df_codigo['DESCRIPCION'].iloc[0],#codigo,
                line=dict(width=2, color = "#4682B4"),  # Personalizar el ancho de la línea                
                hovertemplate=(
                "<b>Fecha:</b> %{x}<br>"
                "<b>Demanda:</b> %{y}<br>"
                #f"<b>Descripción:</b> {df_codigo['DESCRIPCION'].iloc[0]}<br>"
                #"<extra></extra>"
                ),
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
        margin=dict(l=50, r=50, t=80, b=50),  # Ajustar márgenes
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


# In[11]:


def preprocesar_demanda_cliente(df_mes_cliente):
    
    # Generar copia de trabajo
    df_mes_orig = df_mes_cliente.copy()

    # Eliminar columnas codigo, cliente
    df_mes_orig = df_mes_orig.drop(columns=['CODIGO','CLIENTE'])

    # Renombrar columna codigo como codigo_cliente
    df_mes_orig = df_mes_orig.rename(columns={'CODIGO_CLIENTE':'CODIGO'})
    
    # Calcular el largo de cada serie de tiempo por CODIGO
    series_length = df_mes_orig.groupby('CODIGO').size()
    
    # Filtrar los códigos con menos de 2 registros
    codigos_cortos = series_length[series_length < 2].index.tolist()
    
    # Generar un reporte de estos códigos
    reporte_codigos = df_mes_orig[df_mes_orig['CODIGO'].isin(codigos_cortos)]
    
    # Eliminar los códigos con menos de 2 registros del DataFrame original
    df_mes_orig = df_mes_orig[~df_mes_orig['CODIGO'].isin(codigos_cortos)]

    print('No se pronosticaran las siguientes referencias debido a que tienen muy pocos datos:')
    print(reporte_codigos.groupby('CODIGO').size())

    return df_mes_orig, reporte_codigos


# In[12]:


# Selecciona si se quiere pronosticar por codigo consolidado (suministro) o por codigo_cliente
def seleccionar_tipo_pronostico(opcion, df_mes_cliente):

    # Conndicional para selecccionar tipo de pronostico
    if opcion == 'POR CODIGO CLIENTE':
        df_mes_orig, reporte_codigos = preprocesar_demanda_cliente(df_mes_cliente)
    elif opcion == 'POR CODIGO AGREGADO':
        df_mes_orig, reporte_codigos = agrupar_demanda(df_mes_cliente)
    
    return df_mes_orig, reporte_codigos


# ## Reemplazar los ceros por la mediana

# In[13]:


# Para pronosticos por mes
def reemplazar_ceros(df_mes_orig):
    
    # Generar copia de trabajo
    df_mes_ceros = df_mes_orig.copy()

    # Reemplaza los datos menores a 70 unidades (tambien considerados 0s) con la mediana de la serie de tiempo
    df_mes_ceros['DEMANDA'] = df_mes_orig.groupby('CODIGO')['DEMANDA'].transform(
        lambda x: x.where(x >= 70, x.median())
    )
    
    return df_mes_ceros

# Solo para pronosticos de novaventa:
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


# ## Imputar Outliers con Lim Sup - Lim Inf

# * Se crea un pronostico con n=6, para los primeros periodos se va promediando los datos disponibles hasta que tenga 6 datos
# * Se definen limites superior e inferior con base en la distribucion normal, 98% y 2% (ajustable)
# * Se aplica distribucion normal, lo que quede por fuera de limites se marca como outlier, se recalcula promedio y desviacion
# * Se establece lim_sup como pronostico + percentil 98% y lim_inf como pronostico - percentil 2%

# In[14]:


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
    


# In[15]:


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
        df_acumulado = df_acumulado[['CODIGO','DESCRIPCION',	
                             'CONSECUTIVO', 
                             'NUEVA_DEM']]
    else:
        df_acumulado = df_acumulado[['CODIGO','DESCRIPCION',	
                               'NUEVA_DEM']]
        
    df_mes = df_acumulado.rename(columns={'NUEVA_DEM':'DEMANDA'})
    
    # Mostrar el DataFrame acumulado
    reporte_outliers = df_outliers[df_outliers['IS_OUTLIER'] == True].reset_index()
    
    return df_mes, df_outliers, reporte_outliers


# In[16]:


def graficar_outliers_subplots(df_mes_ceros, df_outliers, sup, inf, n):
    
    # Lista de SKUs únicos
    lista_skus = df_mes_ceros['CODIGO'].unique()
    descripciones_unicas = df_mes_ceros['DESCRIPCION'].unique()
    # Calcular número de filas necesarias para 3 columnas
    n_cols = 3
    n_rows = -(-len(lista_skus) // n_cols)  # Redondeo hacia arriba

    # Ajustar el espaciado vertical según la columna 'CONSECUTIVO'
    vertical_spacing = 0.085 if 'CONSECUTIVO' in df_mes_ceros.columns else 0.032
    
    # Crear los subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        #subplot_titles=[f"SKU: {sku}" for sku in lista_skus],
        subplot_titles = [
            f"Código: {sku}<br><span style='font-size:10px;'>{descripcion}</span>"
            for sku, descripcion in zip(lista_skus, descripciones_unicas)
        ],
        horizontal_spacing=0.08,
        vertical_spacing=vertical_spacing
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
        margin=dict(l=50, r=50, t=80, b=40),  # Ajustar márgenes
        template="ggplot2",
        height=200 * n_rows,  # Ajustar altura según filas
        #width=900,  # Ancho fijo
        showlegend=False  # Ocultar leyenda global
    )
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=11)  # Reducir tamaño del texto de los títulos de subplots

    if 'CONSECUTIVO' in df_mes_ceros.columns:
        fig.update_xaxes(
            title_font=dict(size=6),  # Tamaño de fuente para los títulos del eje X
            type='category',  # Especificar que el eje X es categórico
            tickmode='array',  # Asegurar que las etiquetas del eje X no sean interpretadas como fechas
            tickangle=-35,
        )
    else:
        fig.update_xaxes(
            title_font=dict(size=6),  # Tamaño de fuente para los títulos del eje X
            tickangle=-25,
        )
    # Mostrar la figura
    #fig.show()
    st.plotly_chart(fig, use_container_width=True)


# # Funciones generales de ayuda para procesar modelos

# ## Funcion para crear la lista de sku's

# In[17]:


def crear_lista_skus(df_mes):
    lista_skus = df_mes['CODIGO'].unique()
    return lista_skus


# ## Funcion para calcular el numero de meses a evaluar por cada SKU

# In[18]:


def calcular_meses_a_evaluar(df_sku, periodo_max_evaluacion, porc_eval):
       
    #Calculo del largo de cada serie de tiempo
    largo_serie_tiempo = len(df_sku)
    
    # Calculo del numero de meses a usar como testeo de acuerdo con porc_eval
    meses_evaluar = min(periodo_max_evaluacion, math.ceil(largo_serie_tiempo * porc_eval))

    return meses_evaluar


# ## Funcion para crear el rango de fechas para iterar

# In[19]:


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


# ## Funciones para calculo y medicion de metricas de error

# ### Función para crear columnas de error, error absoluto, error porcentual y error cuadrado

# In[20]:


def crear_columnas_error(df):
    
    df['ERROR'] = df['DEMANDA'] - df['FORECAST'] # Error
    df['ABS_ERROR'] = df['ERROR'].abs() # Error Absoluto
    df['ERROR_PORC'] = np.where(df['DEMANDA'] == 0, 2, df['ABS_ERROR'] / df['DEMANDA']) # Error porcentual, devuelve 200% si la demanda es 0
    df['ERROR_CUADRADO'] = df['ERROR'] ** 2 # Error al cuadrado
    
    return df


# ### Funcion para calcular las metricas totales dado un df con columnas de error

# In[21]:


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


# ### Funcion para evaluar el error por sku

# In[22]:


def kpi_error_sku(df):
    
    if df is None or df.empty:
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
    grupo_sku_mes_error = df_test.groupby(['CODIGO'], observed=True).agg({
                                                            'DEMANDA': 'sum',
                                                            'ERROR': 'sum',
                                                            'ABS_ERROR': 'sum',
                                                            'ERROR_CUADRADO': ['sum', 'count'],
                                                            }).reset_index()
    
    # Renombrar columnas
    grupo_sku_mes_error.columns = ['CODIGO',
                            'DEMANDA', 'ERROR', 'ABS_ERROR', 
                             'ERROR_CUADRADO_suma', 'ERROR_CUADRADO_cuenta']

    # Calcular error rmse por codigo
    grupo_sku_mes_error = calcular_error(grupo_sku_mes_error)

    # Filtrar las columnas para mejor visualizacion
    rmse_sku_mes = grupo_sku_mes_error[['CODIGO','RMSE']]
    
    return grupo_sku_error_formato, rmse_sku_lag, rmse_sku_mes


# ### Funcion para calcular metricas por una sola linea

# In[23]:


def calcular_error(df):
    df['MAE%'] = df['ABS_ERROR']/df['DEMANDA']
    df['SESGO%'] = df['ERROR']/df['DEMANDA']
    df['SCORE%'] = df['MAE%'] + df['SESGO%'].abs()
    if 'ERROR_CUADRADO_suma' in df.columns:
        df['RMSE'] = np.sqrt(df['ERROR_CUADRADO_suma'] / df['ERROR_CUADRADO_cuenta'])
    return df


# ### Funcion para calcular errores por LAG para el promedio

# In[24]:


def evaluar_lags(df):
    
    # Calcular los scores por lag
    df_lags = df.groupby('LAG')[['ERROR', 'ABS_ERROR', 'DEMANDA']].sum()
    df_lags['MAE%'] = np.where(df_lags['DEMANDA'] == 0, 2,
                            (df_lags['ABS_ERROR'] / df_lags['DEMANDA']))
    df_lags['SESGO%'] =  np.where(df_lags['DEMANDA'] == 0, 2,
                            (abs(df_lags['ERROR'] / df_lags['DEMANDA']))
                                 )
    # Calcular los scores por lag evitando la división cuando DEMANDA es cero                  
    df_lags['SCORE%'] = np.where(df_lags['DEMANDA'] == 0, 2,
                            (df_lags['ABS_ERROR'] / df_lags['DEMANDA']) + abs(df_lags['ERROR'] / df_lags['DEMANDA'])
                            )
    return df_lags


# In[25]:


def kpi_error_lag(df):
    
    if df is None or df.empty:
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


# In[26]:


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


# ## Funciones para formatear y graficar

# ### Funcion para construir pronostico final para el promedio movil simple

# In[27]:


# Para pronosticos por mes
def construir_pronostico_pms(df_mejor, df_mes, meses_a_pronosticar_produccion, nombre_modelo, barra_progreso, status_text):

    # Crear un nuevo DataFrame para almacenar los resultados
    data = []
    
    # Iterar por cada fila de df_mejor
    for _, row in df_mejor.iterrows():
        codigo = row["CODIGO"]
        ultimo_forecast = row["ultimo_forecast"]
        # Generar las fechas para los meses pronosticados
        fechas = [df_mes.index.max() + pd.DateOffset(months=i) for i in range(1, meses_a_pronosticar_produccion + 1)]
    
        # Generar las filas para los meses pronosticados
        for i, fecha in enumerate(fechas, start=1):
            data.append({
                "FECHA": fecha,
                "CODIGO": codigo,
                "FORECAST": ultimo_forecast,
                "LAG": f"Lag_{i}"
            })
    
    # Crear el nuevo DataFrame
    df_forecast = pd.DataFrame(data)
    df_forecast = df_forecast.set_index('FECHA')
    df_forecast['MODELO'] = nombre_modelo
    status_text.success("Pronósticos PMS Generados")
    barra_progreso.empty() 
    # Visualizar el resultado
    return df_forecast

# Solo para novaventa:
def construir_pronostico_pms_nv(df_mejor, df_mes, meses_a_pronosticar_produccion, nombre_modelo, barra_progreso, status_text):

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
    status_text.success("Pronósticos PMS Generados")
    barra_progreso.empty() 
    # Visualizar el resultado
    return df_forecast


# ### Funcion para adicionar nombre del modelo

# In[28]:


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


# # Modelos de Pronosticos de Series de Tiempo

# ## Simulacion promedio movil simple PMS

# Evalúa y genera pronósticos PMS para un grupo de SKUs, seleccionando el mejor n con base en el score.
# 
# Args:
# * df_mes (DataFrame): DataFrame con datos de demanda imputando outliers por SKU.
# * df_mes_ceros (Data Frame): DataFrame con datos de demanda original (sin imputar outliers), pero con ceros reemplazados por la mediana
# * lista_skus (list): Lista de SKUs a evaluar.
# * periodo_max_evaluacion (int): Máximo numero de meses a evaluar - se mantendra en 12.
# * porc_eval (float): Porcentaje de datos (meses) a usar para evaluación.
# * meses_a_pronosticar_evaluacion (int): Número de meses a pronosticar para efectos de seleccion de modelo
# 
# Devuelve:
# * DataFrame df_mejor_n: Información del mejor n, metricas, pronostico por SKU.
# * DataFrame df_forecast_pms: Datos de pronósticos seleccionados.
# 

# In[29]:


# Para pronosticos por mes:
def evaluar_y_generar_pms(df_mes, df_mes_ceros, lista_skus, 
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
    
        # Crear el rango de fechas para cortar el set de datos de acuerdo con meses a evaluar
        rango_fechas = crear_rango_fechas(df_sku_fecha, meses_evaluar)
       
        # Tamaño de histórico n maximo y rango   
        n_max = max(2, len(df_sku_fecha) - meses_evaluar)        
        rango_n = range(1, n_max)
        
        # Iterar por cada posible tamaño de n
        for n in rango_n: 
            
            datos_evaluacion = []  # Bolsa para guardar resultados evaluados
            ultimo_forecast_n = None  # Variable para almacenar el último forecast de cada n
        
            for fecha_corte in rango_fechas:
                # Filtrar datos hasta la fecha de corte
                df_sku_fecha_temp = df_sku_fecha[df_sku_fecha.index <= fecha_corte].copy()
                
                if len(df_sku_fecha_temp['DEMANDA']) > 1:
                    # Calcular el forecast usando una media móvil con ventana n
                    #print(len(df_sku_fecha_temp['DEMANDA']))
                    df_sku_fecha_temp['FORECAST'] = df_sku_fecha_temp['DEMANDA'].rolling(window=n, min_periods=1).mean()
 
                    forecast = [df_sku_fecha_temp['FORECAST'].iloc[-1]]

                else:
                    #forecast = [np.NaN]
                    forecast = df_sku_fecha_temp['DEMANDA'].iloc[-1]
                # Generar los próximos lags para el forecast actual            
                datos_forecast = pd.DataFrame({                
                    'fecha':fecha_corte,
                    'n':n,
                    'CODIGO': sku,
                    'FORECAST': forecast,
                    'LAG': [f'Lag_{i}' for i in range(1, meses_a_pronosticar_evaluacion + 1)]}, index=[df_sku_fecha_temp.index[-1] 
                                      + pd.DateOffset(months=i) for i in range(1, meses_a_pronosticar_evaluacion + 1)]) # Genera titulo Lags dinamicamente
                 
                # Unir forecast con la demanda real para evaluar
                datos_forecast_demanda = datos_forecast.merge(df_sku_fecha_ceros[['DEMANDA']], 
                                                how='left', left_index=True, right_index=True)            

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
    status_text.text("Backtesting PMS Terminado")
    barra_progreso.empty()  
    return df_mejor_n, df_forecast_pms 

# Solo para novaventa:
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
    status_text.text("Backtesting PMS Terminado")
    barra_progreso.empty()  
    return df_mejor_n, df_forecast_pms 


# ## Suavizacion Exponencial Simple

# In[30]:


def encontrar_mejor_se(df_mes, df_mes_ceros, lista_skus, periodo_max_evaluacion, porc_eval, 
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
        meses_evaluar = calcular_meses_a_evaluar(df_sku_fecha, periodo_max_evaluacion, porc_eval)
    
        # Crear el rango de fechas para corar el set de datos de acuerdo con meses a evaluar
        rango_fechas = crear_rango_fechas(df_sku_fecha, meses_evaluar)

        # Iterar por fecha
        for fecha_corte in rango_fechas:
            
            # Filtrar datos hasta la fecha de corte
            df_sku_fecha_temp = df_sku_fecha[df_sku_fecha.index <= fecha_corte].copy()                 
    
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
                        'fecha':fecha_corte,
                        'CODIGO': sku,
                        'FORECAST': forecast,
                        'LAG': [f'Lag_{i}' for i in range(1, meses_a_pronosticar_evaluacion + 1)]
                                }, 
                        index=[df_sku_fecha_temp.index[-1] 
                                          + pd.DateOffset(months=i) for i in range(1,  meses_a_pronosticar_evaluacion + 1)])
                    
                    # Unir forecast con la demanda real para evaluar
                    datos_forecast_demanda = datos_forecast.merge(df_sku_fecha_ceros[['DEMANDA']], 
                                                    how='left', left_index=True, right_index=True)

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
    if porc_eval == 0:
        status_text.success("Pronósticos Suavización Exponencial Generados")
    else:
        status_text.text("Backtesting Suavización Exponencial Terminado")
        
    barra_progreso.empty()
    
    return df_mejor_se,  df_forecast_se

# Solo para novaventa
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
    if porc_eval == 0:
        status_text.success("Pronósticos Suavización Exponencial Generados")
    else:
        status_text.text("Backtesting Suavización Exponencial Terminado")
    barra_progreso.empty()
    return df_mejor_se,  df_forecast_se


# ## Regresion lineal simple y estacional

# In[31]:


# Para pronosticos por mes
def aplicar_regresion_lineal_simple(lista_skus, df_mes, df_mes_ceros, 
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
    
    #for sku in tqdm(lista_skus, desc="Procesando SKUs"):
        #resultados_rl_lineal = []
        #resultados_rl_estacional = []
        resultados_regresion_lineal = []  
        resultados_regresion_estacional = []  
                    
        df_sku_fecha = df_mes[df_mes['CODIGO'] == sku].copy() # Filtrar df_almacen_semana por cada SKU
        df_sku_fecha_ceros = df_mes_ceros[df_mes_ceros['CODIGO'] == sku].copy() # Filtrar df_mes_ceros por cada Sku            
        
        # Evaluar el largo de la serie de tiempo y calcular meses a evaluar para cada sku
        meses_evaluar = calcular_meses_a_evaluar(df_sku_fecha, periodo_max_evaluacion, porc_eval)
        
        # Crear el rango de fechas para cortar el set de datos de acuerdo con meses a evaluar
        rango_fechas = crear_rango_fechas(df_sku_fecha, meses_evaluar)
        
        # Iterar por fecha
        for fecha_corte in rango_fechas:
            # Filtrar datos hasta la fecha de corte
            df_sku_fecha_temp = df_sku_fecha[df_sku_fecha.index <= fecha_corte].copy()                 
            
            # Extraer la demanda como un array
            demanda = df_sku_fecha_temp['DEMANDA'].values
            
            if len(demanda) >= 4:
                # Generar y adecuar la variable independiente tiempo
                X = np.arange(1, len(demanda)+1).reshape(-1, 1)
                y = demanda
                # Modelo de Regresión Lineal
                model = LinearRegression()
                model.fit(X, y)
                
                # Pronóstico para los próximos 6 periodos
                limite_sup_pronost = len(demanda)+1 + meses_a_pronosticar_evaluacion
                X_futuro = np.arange(len(demanda)+1, limite_sup_pronost).reshape(-1, 1)
                y_futuro_lineal = model.predict(X_futuro)

                if len(demanda) >= 12:
                    factores_estacionales_mes = demanda[-12:] / demanda[-12:].mean() 
                    y_futuro_estacional =  y_futuro_lineal * factores_estacionales_mes[:len(y_futuro_lineal)]
                else:    
                    y_futuro_estacional = [np.NaN] * meses_a_pronosticar_evaluacion 
            
            else:
                print(f"Sin datos: SKU={sku}, Fecha={fecha_corte}, Datos disponibles={len(demanda)}") 
                y_futuro_lineal = [np.NaN] * meses_a_pronosticar_evaluacion 
            
            # Almacenar los resultados lineales con código y periodo
            for periodo, prediccion in zip(range(len(demanda)+1, limite_sup_pronost), y_futuro_lineal):
                fecha_pronostico = df_sku_fecha_temp.index.max() + pd.DateOffset(months=(periodo-len(demanda)))
                lag = f"Lag_{periodo - len(demanda)}"
                resultados_regresion_lineal.append({'CODIGO': sku, 'PERIODO': periodo, 'FECHA': fecha_pronostico, 'FORECAST': prediccion, 'LAG': lag})

            # Almacenar los resultados estacionales con código y periodo
            for periodo, prediccion in zip(range(len(demanda)+1, limite_sup_pronost), y_futuro_estacional):
                fecha_pronostico = df_sku_fecha_temp.index.max() + pd.DateOffset(months=(periodo-len(demanda)))
                lag = f"Lag_{periodo - len(demanda)}"
                resultados_regresion_estacional.append({'CODIGO': sku, 'PERIODO': periodo, 'FECHA': fecha_pronostico, 'FORECAST': prediccion, 'LAG': lag})
        
        df_forecast_regresion_lineal = pd.DataFrame(resultados_regresion_lineal)#.set_index('FECHA')
        df_forecast_regresion_lineal = df_forecast_regresion_lineal.set_index('FECHA')

        df_forecast_regresion_estacional = pd.DataFrame(resultados_regresion_estacional).set_index('FECHA')
        
        # Unir forecast con la demanda real para evaluar
        datos_forecast_demanda_lineal = df_forecast_regresion_lineal.merge(
            df_mes_ceros[['CODIGO', 'DEMANDA']], 
            how='left', 
            left_on=['CODIGO', df_forecast_regresion_lineal.index], 
            right_on=['CODIGO', df_mes_ceros.index]
        )
        # Unir forecast con la demanda real para evaluar
        datos_forecast_demanda_estacional = df_forecast_regresion_estacional.merge(
            df_mes_ceros[['CODIGO', 'DEMANDA']], 
            how='left', 
            left_on=['CODIGO', df_forecast_regresion_estacional.index], 
            right_on=['CODIGO', df_mes_ceros.index]
        )
        # Condicionar eliminacion de NaN a si es evaluacion o generacion de pronostico
        if porc_eval != 0:
            datos_forecast_demanda_lineal = datos_forecast_demanda_lineal.dropna()
            datos_forecast_demanda_estacional = datos_forecast_demanda_estacional.dropna()
            
        # Renombrar la columna 'key_1' a 'FECHA' y Establecer 'FECHA' como el índice
        datos_forecast_demanda_lineal = datos_forecast_demanda_lineal.rename(columns={'key_1': 'FECHA'}).set_index('FECHA')
        # Renombrar la columna 'key_1' a 'FECHA' y Establecer 'FECHA' como el índice
        datos_forecast_demanda_estacional = datos_forecast_demanda_estacional.rename(columns={'key_1': 'FECHA'}).set_index('FECHA')
    
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
    
    if porc_eval == 0:
        status_text.success("Pronósticos Regresión Lineal Generados")
    else:
        status_text.text("Backtesting Regresión Lineal Terminado")
    
    barra_progreso.empty()                                     
    # Visualizar resultados
    return df_mejor_rl_lineal, df_mejor_rl_estacional, df_forecast_rl_lineal, df_forecast_rl_estacional

# Solo para novaventa:
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
    if porc_eval == 0:
        status_text.success("Pronósticos Regresión Lineal Generados")
    else:
        status_text.text("Backtesting Regresión Lineal Terminado")
    barra_progreso.empty()                                     
    # Visualizar resultados
    return df_mejor_rl_lineal, df_mejor_rl_estacional, df_forecast_rl_lineal, df_forecast_rl_estacional


# ## MSTL con Regresion Polinomica y Mayor Peso a la Ultima Tendencia

# In[32]:


# Para pronosticos por meses
def aplicar_mstl(lista_skus, df_mes, df_mes_ceros, 
                                    periodo_max_evaluacion, porc_eval, 
                                    meses_a_pronosticar_evaluacion, peso_ult_data,
                                    barra_progreso,
                                    status_text):

    df_lag_forecasts = []
    resultados_datos_forecast_demanda_mstl = []
    resultados_mstl = []
    total_series = len(lista_skus)  
    for i, sku in enumerate(tqdm(lista_skus, desc="Procesando SKUs")):    
#for sku in tqdm(lista_skus, desc="Procesando SKUs"):
        # Actualizar barra de progreso y mensaje de estado
        barra_progreso.progress((i + 1) / total_series)
        status_text.text(f"Evaluando MSTL para SKU N° {i + 1} de {total_series}...")
    
    #for sku in tqdm(lista_skus, desc="Procesando SKUs"):           
                       
        df_sku_fecha = df_mes[df_mes['CODIGO'] == sku].copy() # Filtrar df_mes por cada SKU
        df_sku_fecha_ceros = df_mes_ceros[df_mes_ceros['CODIGO'] == sku].copy() # Filtrar df_mes_ceros por cada Sku            
        
        # Evaluar el largo de la serie de tiempo y calcular meses a evaluar para cada sku
        meses_evaluar = calcular_meses_a_evaluar(df_sku_fecha, periodo_max_evaluacion, porc_eval)
        #print(sku, 'meses_evaluar:',meses_evaluar)
        # Crear el rango de fechas para corar el set de datos de acuerdo con meses a evaluar
        rango_fechas = crear_rango_fechas(df_sku_fecha, meses_evaluar)
        #print(rango_fechas)
        
        forecasts = []
        # Iterar por fecha
        for fecha_corte in rango_fechas:
            
            # Filtrar datos hasta la fecha de corte
            df_sku_fecha_temp = df_sku_fecha[df_sku_fecha.index <= fecha_corte].copy()                 
    
            # Extraer la demanda y la fecha como un array
            date = df_sku_fecha_temp.index
            demanda = df_sku_fecha_temp['DEMANDA'].values
            demand_series = pd.Series(demanda, index=date)                      
    
            if len(demand_series) > 24:
                #print(f"sku: {sku}, longitud: {len(demand_series)}")
                  
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    try:
                        # Aplicar descomposición MSTL
                        mstl_model = MSTL(demand_series, periods=12, stl_kwargs={'seasonal_deg': 0})
                        descomposicion = mstl_model.fit()
                        
                        # Extraer componentes
                        tendencia = descomposicion.trend
                        seasonal = descomposicion.seasonal
                        # Usar el indice en formato tiempo como indice de tiempo para la regresion
                        indice_tiempo = pd.to_numeric(demand_series.index)  # Convertir DatetimeIndex a formato numero para la regresion
                        peso_ult_data = peso_ult_data  # Ajustar para incrementar o disminuir la importancia de los datos mas recientes
                        pesos = np.exp(peso_ult_data * np.arange(len(tendencia)))
                        # Aplicar regresion polinomica a los datos de tendencia
                        poly = PolynomialFeatures(degree=2)  # Aplicar grado 2 para obtener una proyeccion no lineal
                        X_poly = poly.fit_transform(indice_tiempo.values.reshape(-1, 1))
                        
                        model = LinearRegression()
                        model.fit(X_poly, tendencia, sample_weight=pesos)
    
                     # Proyectar la tendencia para los proximos 7 periodos
                        fechas_futuras = [demand_series.index[-1] + DateOffset(months=i+1) for i in range(meses_a_pronosticar_evaluacion)]
                        indice_fechas_futuras = pd.to_numeric(pd.Index(fechas_futuras))
                        X_poly_futura = poly.transform(indice_fechas_futuras.values.reshape(-1, 1))
                        pronostico_tendencia = model.predict(X_poly_futura)
    
                         # Calcular el indice estacional para cada mes
                        estacionalidad_promedio = seasonal.groupby(seasonal.index.month).mean()
                        
                        # Determinar el mes de inicio de fecha pronostico 
                        mes_inicial = fechas_futuras[0].month
                    
                        # Proyectar el componente estacional por los proximos  periodos
                        pronostico_estacional = [estacionalidad_promedio[(mes_inicial + i - 1) % 12 + 1] for i in range(meses_a_pronosticar_evaluacion)]
                    
                        # Combinar tendencia y componente estacional en un solo pronostico
                        pronostico_final = pronostico_tendencia + pronostico_estacional
                    
                        # Crear una serie de pandas para mejor manejo
                        pronostico_final_series = pd.Series(pronostico_final, index=fechas_futuras)
                        
                        forecasts.append((sku, date, pronostico_final))
                       
                    except Exception as e:
                        print(f"Error al ajustar MSTL para {sku}: {e}")
                        continue

    
        for sku, fechas_originales, forecast_series in forecasts:
            #print('fechas originales:', fechas_originales, 'forecast_series:', forecast_series)
            ultima_fecha = fechas_originales[-1]
            
            # Crear lista de fechas, de a mes a partir de ultima fecha + 1 mes
            fechas_pronosticos = [ultima_fecha + DateOffset(months=i+1) for i in range(len(forecast_series))]
            
            # Crear lista de valores de lags
            lags = [f"Lag_{i}" for i in range(1, len(forecast_series) + 1)]
            # Crear df temporal por sku
            temp_df = pd.DataFrame({
                'FECHA': fechas_pronosticos,
                'LAG':lags,
                'CODIGO': [sku] * len(forecast_series),
                'FORECAST': forecast_series,
                
            })
            
            # Acumular el df temporal al df principal
            df_lag_forecasts.append(temp_df)
    
        
        df_forecasts_mstl = pd.concat(df_lag_forecasts, ignore_index=True)
        df_forecasts_mstl = df_forecasts_mstl.set_index('FECHA')
        df2 = df_forecasts_mstl.copy()
        # Unir forecast con la demanda real para evaluar
        datos_forecast_mstl = df_forecasts_mstl.merge(
            df_mes_ceros[['CODIGO', 'DEMANDA']], 
            how='left', 
            left_on=['CODIGO', df_forecasts_mstl.index], 
            right_on=['CODIGO', df_mes_ceros.index]
        )
        if porc_eval != 0:
            datos_forecast_mstl = datos_forecast_mstl.dropna()
        
         # Renombrar la columna 'key_1' a 'FECHA' y Establecer 'FECHA' como el índice
        df_forecast_mstl = datos_forecast_mstl.rename(columns={'key_1': 'FECHA'}).set_index('FECHA').copy()
                      
        df_columnas_error_mstl = crear_columnas_error(datos_forecast_mstl)

        df_mejor_mstl = agrupar_por_codigo(df_columnas_error_mstl)   
    
    #df_mejor_mstl = pd.DataFrame(resultados_mstl)
    if porc_eval == 0:
        status_text.success("Pronósticos MSTL Generados")
    else:
        status_text.text("Backtesting MSTL Terminado")
    barra_progreso.empty()
    return  df_mejor_mstl, df_forecast_mstl

# Solo para pronosticos de Novaventa
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

    if porc_eval == 0:
        status_text.success("Pronósticos MSTL Generados")
    else:
        status_text.text("Backtesting MSTL Terminado")
    barra_progreso.empty()
    return df_mejor_mstl, df_forecast_mstl


# ## Modelo milagros

# In[33]:


def pronostico_recursivo(df, n, sku):
    # Pesos del pronóstico
    pesos = [3 / 6, 2 / 6, 1 / 6]

    # Validar si hay suficientes datos históricos
    if len(df) < 3:
        # Retornar un DataFrame vacío si no hay suficientes datos
        return pd.DataFrame(columns=["SKU", "FECHA", "FORECAST", "LAG"])

    # Obtener la serie histórica de demanda
    demanda = df["DEMANDA"].tolist()

    # Listas para almacenar los resultados
    pronosticos = []
    lags = []

    # Calcular los pronósticos recursivamente
    for i in range(1, n + 1):
        nuevo_pronostico = (
            pesos[0] * demanda[-1]
            + pesos[1] * demanda[-2]
            + pesos[2] * demanda[-3]
        )
        pronosticos.append(nuevo_pronostico)
        lags.append(f"Lag_{i}")
        demanda.append(nuevo_pronostico)  # Añadir el pronóstico como entrada futura
       
    # Crear un índice para los pronósticos
    last_date = df.index[-1]
    forecast_index = pd.date_range(start=last_date, periods=n + 1, freq="MS")[1:]

    # Crear el DataFrame resultante
    forecast_df = pd.DataFrame({
        "FECHA": forecast_index,
        "CODIGO": sku,
        "FORECAST": pronosticos,
        "LAG": lags,
    })
    forecast_df.set_index("FECHA", inplace=True)

    return forecast_df

def pronostico_recursivo_nv(df, n, sku):
    # Pesos del pronóstico
    pesos = [3 / 6, 2 / 6, 1 / 6]

    # Validar si hay suficientes datos históricos
    if len(df) < 3:
        # Retornar un DataFrame vacío si no hay suficientes datos
        return pd.DataFrame(columns=["SKU", "CONSECUTIVO", "FORECAST", "LAG"])
        
    # Obtener la serie histórica de demanda
    demanda = df["DEMANDA"].tolist()

    # Listas para almacenar los resultados
    pronosticos = []
    lags = []

    # Calcular los pronósticos recursivamente
    for i in range(1, n + 1):
        nuevo_pronostico = (
            pesos[0] * demanda[-1]
            + pesos[1] * demanda[-2]
            + pesos[2] * demanda[-3]
        )
        pronosticos.append(nuevo_pronostico)
        lags.append(f"Lag_{i}")
        demanda.append(nuevo_pronostico)  # Añadir el pronóstico como entrada futura
        
    # Crear un índice para los pronósticos
    last_date = df['CONSECUTIVO'].iloc[-1]
    forecast_index = range(last_date + 1, last_date + n +1 )

    # Crear el DataFrame resultante
    forecast_df = pd.DataFrame({
        "CONSECUTIVO": forecast_index,
        "CODIGO": sku,
        "FORECAST": pronosticos,
        "LAG": lags,
    })
    #forecast_df.set_index("CONSECUTIVO", inplace=True)
    #display(forecast_df)
    return forecast_df


# In[34]:


# Para pronosticos por meses
def aplicar_milagros(lista_skus, df_mes, df_mes_ceros, 
                                    periodo_max_evaluacion, porc_eval, 
                                    meses_a_pronosticar_evaluacion, 
                                    barra_progreso,
                                    status_text):
    
    forecasts = []
    total_series = len(lista_skus)
   
    for i, sku in enumerate(tqdm(lista_skus, desc="Procesando SKUs")):    

        # Actualizar barra de progreso y mensaje de estado
        barra_progreso.progress((i + 1) / total_series)
        status_text.text(f"Evaluando modelo Milagros para SKU N° {i + 1} de {total_series}...")         
                       
        df_sku_fecha = df_mes[df_mes['CODIGO'] == sku].copy() # Filtrar df_mes por cada SKU
        df_sku_fecha_ceros = df_mes_ceros[df_mes_ceros['CODIGO'] == sku].copy() # Filtrar df_mes_ceros por cada Sku            
        
        # Evaluar el largo de la serie de tiempo y calcular meses a evaluar para cada sku
        meses_evaluar = calcular_meses_a_evaluar(df_sku_fecha, periodo_max_evaluacion, porc_eval)
        #print(sku, 'meses_evaluar:', meses_evaluar)
        # Crear el rango de fechas para corar el set de datos de acuerdo con meses a evaluar
        rango_fechas = crear_rango_fechas(df_sku_fecha, meses_evaluar)
        #print(rango_fechas)
        
        acum_pronosticos = []
        # Iterar por fecha
        for fecha_corte in rango_fechas:
            
            # Filtrar datos hasta la fecha de corte
            df_sku_fecha_temp = df_sku_fecha[df_sku_fecha.index <= fecha_corte].copy()                 
                
            # Generar pronósticos
            pronosticos = pronostico_recursivo(df_sku_fecha_temp, meses_a_pronosticar_evaluacion, sku)
                
                # Si el DataFrame no está vacío, agregarlo a la lista
            if not pronosticos.empty:
                acum_pronosticos.append(pronosticos)
            
        # Combinar todos los pronósticos en un único DataFrame
        acum_pronosticos = [df for df in acum_pronosticos if not df.empty]
        if acum_pronosticos:
            acum_pronosticos = pd.concat(acum_pronosticos)
        else:
            acum_pronosticos = pd.DataFrame(columns=["CODIGO", "FORECAST", "LAG"])
        
        forecasts.append(acum_pronosticos)
    forecasts = [df for df in forecasts if not df.empty]
    if forecasts:
        forecasts = pd.concat(forecasts)
    else:
        forecasts = pd.DataFrame(columns=["CODIGO", "FORECAST", "LAG"])

    if porc_eval == 0:
       forecasts = forecasts.merge(
                df_mes_ceros[['CODIGO', 'DEMANDA']],  # Seleccionamos solo las columnas relevantes
                on=["FECHA", "CODIGO"],              # Claves para el merge
                how="left"                           # Tipo de join (left mantiene todas las filas de forecasts)
            ).drop(columns='DEMANDA')

    else:
        forecasts = forecasts.merge(
                        df_mes_ceros[['CODIGO', 'DEMANDA']],  # Seleccionamos solo las columnas relevantes
                        on=["FECHA", "CODIGO"],              # Claves para el merge
                        how="left"                           # Tipo de join (left mantiene todas las filas de forecasts)
                    ).dropna()
    if porc_eval == 0:
        status_text.success("Pronósticos Modelo Milagros Generados")
    else:
        status_text.text("Backtesting Modelo Milagros Terminado")
    barra_progreso.empty()
    return forecasts


def aplicar_milagros_nv(lista_skus, df_mes, df_mes_ceros, 
                                    periodo_max_evaluacion, porc_eval, 
                                    meses_a_pronosticar_evaluacion, 
                                    barra_progreso,
                                    status_text):
    
    forecasts = []
    total_series = len(lista_skus)
   
    for i, sku in enumerate(tqdm(lista_skus, desc="Procesando SKUs")):    

        # Actualizar barra de progreso y mensaje de estado
        barra_progreso.progress((i + 1) / total_series)
        status_text.text(f"Evaluando modelo Milagros para SKU N° {i + 1} de {total_series}...")         
                       
        df_sku_fecha = df_mes[df_mes['CODIGO'] == sku].copy() # Filtrar df_mes por cada SKU
        df_sku_fecha_ceros = df_mes_ceros[df_mes_ceros['CODIGO'] == sku].copy() # Filtrar df_mes_ceros por cada Sku            
        
        # Evaluar el largo de la serie de tiempo y calcular meses a evaluar para cada sku
        meses_evaluar = calcular_meses_a_evaluar(df_sku_fecha, periodo_max_evaluacion, porc_eval)
        #print(sku, 'meses_evaluar:', meses_evaluar)
        # Crear el rango de fechas para corar el set de datos de acuerdo con meses a evaluar
        rango_fechas = crear_rango_fechas(df_sku_fecha, meses_evaluar)
        #print(rango_fechas)
        
        acum_pronosticos = []
        # Iterar por fecha
        for fecha_corte in rango_fechas:
            
            # Filtrar datos hasta la fecha de corte
            #df_sku_fecha_temp = df_sku_fecha[df_sku_fecha.index <= fecha_corte].copy()                 
            df_sku_fecha_temp = df_sku_fecha[df_sku_fecha['CONSECUTIVO'] <= fecha_corte].copy()    
            # Generar pronósticos
            pronosticos = pronostico_recursivo_nv(df_sku_fecha_temp, meses_a_pronosticar_evaluacion, sku)
                
                # Si el DataFrame no está vacío, agregarlo a la lista
            if not pronosticos.empty:
                acum_pronosticos.append(pronosticos)
            
        # Combinar todos los pronósticos en un único DataFrame
        acum_pronosticos = [df for df in acum_pronosticos if not df.empty]
        if acum_pronosticos:
            acum_pronosticos = pd.concat(acum_pronosticos)
        else:
            acum_pronosticos = pd.DataFrame(columns=["CODIGO","CONSECUTIVO", "FORECAST", "LAG"])
        
        forecasts.append(acum_pronosticos)
    forecasts = [df for df in forecasts if not df.empty]
    if forecasts:
        forecasts = pd.concat(forecasts)
    
    else:
        forecasts = pd.DataFrame(columns=["CODIGO", "CONSECUTIVO", "FORECAST", "LAG"])
    
    if porc_eval == 0:
       forecasts = forecasts.merge(
                df_mes_ceros[['CODIGO', 'DEMANDA', 'CONSECUTIVO']],  # Seleccionamos solo las columnas relevantes
                on=["CONSECUTIVO", "CODIGO"], # Claves para el merge
                how="left" # Tipo de join (left mantiene todas las filas de forecasts)
            ).drop(columns='DEMANDA')

    else:
        forecasts = forecasts.merge(
                        df_mes_ceros[['CODIGO', 'DEMANDA', 'CONSECUTIVO']],  # Seleccionamos solo las columnas relevantes
                        on=["CONSECUTIVO", "CODIGO"],              # Claves para el merge
                        how="left"                           # Tipo de join (left mantiene todas las filas de forecasts)
                    ).dropna()
    if porc_eval == 0:
        status_text.success("Pronósticos Modelo Milagros Generados")
    else:
        status_text.text("Backtesting Modelo Milagros Terminado")
    barra_progreso.empty()
    return forecasts


# # Modelos de Pronosticos con Machine Learning

# ## Funciones de ayuda y preprocesamiento para algoritmos de machine learning

# ### Funciones para extraer atributos de tiempo usando feature engine

# In[35]:


#Atributos de fecha
dtfs = DatetimeFeatures(variables= 'index',  
                    features_to_extract=['month','year','days_in_month'],
                    drop_original=False)

#Atributos Periódicos
cyclicfs = CyclicalFeatures(variables = ['month', 'year'],
                        drop_original=False)


# ### Pipeline de preprocesamiento

# In[36]:


pipe = Pipeline(
    [('Atributos_de_fecha', dtfs),
     ('Atributos_periodicidad', cyclicfs),
     ]
    )


# ### Funciones de Preprocesamiento de dataframe para Machine Learning

# In[37]:


def preprocesar_ml(df_mes):
    df_horiz = pd.pivot_table(df_mes, values=['DEMANDA'], index=['FECHA'], columns=['CODIGO'], observed=True)
    
    # Aplanar el indice multi-columna y anteponer DEMANDA a cada Codigo
    df_horiz.columns = ['{}_{}'.format(col[0], col[1]) for col in df_horiz.columns]
    
    # Asignar Frequencia Mes Inicio 'MS'
    df_horiz = df_horiz.resample('MS').asfreq()
    
    # Reemplazar NaN por 0
    df_horiz = df_horiz.fillna(0)

    return df_horiz


# ### Funciones para particion de datos de entrenamiento y testeo

# In[38]:


def partir_data(data, meses_evaluar, meses_a_pronosticar_evaluacion):

   
    end_train = data.index.max() - pd.offsets.MonthEnd(meses_evaluar)
    start_test = data.index.max() - pd.offsets.MonthEnd(meses_evaluar) 
    end_test = start_test + pd.offsets.MonthEnd(meses_a_pronosticar_evaluacion)
    
    data_train = data.loc[:end_train, :].copy()
    data_test  = data.loc[start_test:end_test, :].copy()
    data_temp = data.loc[:end_test, :].copy()
    
    print(f"Fechas Set Entrenamiento: {data_train.index.min()} -> {data_train.index.max()}  (n={len(data_train)}, {len(data_train)/len(data):.0%})")
    print(f"Fechas Set Testeo       : {data_test.index.min()} -> {data_test.index.max()}  (n={len(data_test)}, {len(data_test)/len(data):.0%})")
    
    return data_train, data_test, data_temp


# ### Creación de Metrica Score% = MAE% + abs(Sesgo%)

# In[39]:


def custom_metric(y_true, y_pred):
    if np.sum(y_true) == 0:
        return 100
    else:
        return 100 * ( np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) + np.abs(np.sum(y_true - y_pred) / np.sum(y_true)) )


# ### Definición de Variables Exogenas de periodos de tiempo

# In[40]:


exog_cols_dtfs = ['month', 'year', 'days_in_month',
       'month_sin', 'month_cos', 'year_sin', 'year_cos']


# ### Preprocesamiento de atributos en set de testeo

# In[41]:


def preparar_vbles_exog(meses_a_pronosticar_evaluacion, data_train, data_test):
    # Frecuencia - Mes Inicio "MS"
    freq = data_train.index.freq.name
    
    fecha_inicio_pronost = data_train.index.max() + pd.DateOffset(months=1)
    fechas_futuras_pronost = pd.date_range(start=fecha_inicio_pronost, freq=freq, periods=meses_a_pronosticar_evaluacion)
    #print('Fechas Futuras Pronóstico:',fechas_futuras_pronost.min(), "->", fechas_futuras_pronost.max())
    
    # Alinear fechas del pronostico con fechas del set de testeo
    fechas_exog_futuras = data_test.query("FECHA <= @fechas_futuras_pronost.max()")
    #print('Fechas Futuras Vbles Exógenas:', fechas_exog_futuras.index.min(), "->", fechas_exog_futuras.index.max())

    return freq, fechas_futuras_pronost, fechas_exog_futuras


# ### Formato pronosticos globales

# In[42]:


def formato_pronosticos_globales(df_global_forecast, df_mes_ceros, pronostico_final):
    
    # Filtrar las columnas que comienzan con "DEMANDA_"
    demanda_columns = [col for col in df_global_forecast.columns if col.startswith("DEMANDA_")]
    
    # Transformar el DataFrame a formato vertical
    df_global_forecast_reset = df_global_forecast.reset_index().rename(columns={'index':'FECHA'})
    
    df_vertical = df_global_forecast_reset.melt(
        id_vars=["FECHA", "LAG"],  # Mantener la columna 'LAG' como identificador
        value_vars=demanda_columns,  # Columnas a transformar
        var_name="CODIGO",  # Nombre para la columna de variables
        value_name="FORECAST"  # Nombre para la columna de valores
    )
    
    # Eliminar el prefijo "DEMANDA_" de la columna 'CODIGO'
    df_vertical["CODIGO"] = df_vertical["CODIGO"].str.replace("DEMANDA_", "", regex=False)
    
    # Reorganizar las columnas
    df_vertical = df_vertical[["FECHA", "LAG", "CODIGO", "FORECAST"]]
    
    # Resetear el índice de df_mes para usar FECHA como una columna normal
    df_mes_ceros_reset = df_mes_ceros.reset_index()
    
    # Realizar el merge entre vertical_df y df_mes usando FECHA y CODIGO
    df_vertical_reset = df_vertical.reset_index()  # Asegurar que FECHA también esté como columna
    
    # Hacer el merge
    df_vertical = df_vertical_reset.merge(
        df_mes_ceros_reset,  # DataFrame con DEMANDA
        how="left",  # Merge para mantener todas las filas de vertical_df
        on=["FECHA", "CODIGO"]  # Claves de unión
    )
    
    # Opcional: reorganizar columnas para claridad
    df_vertical = df_vertical[["FECHA",  "CODIGO", "FORECAST", "DEMANDA", "LAG"]]
    
    # Volver a establecer FECHA como índice
    df_global_forecast_formato = df_vertical.set_index("FECHA")
    
    if pronostico_final == 0:
        df_global_forecast_formato = df_global_forecast_formato.dropna()
    else:
        df_global_forecast_formato = df_global_forecast_formato.drop(columns='DEMANDA')
        
    return df_global_forecast_formato


# ## Modelo XGBoost Local

# ### Definición de espacio de Busqueda de Hyperparametros

# In[43]:


# Espacio de busqueda de hyperparametros
def search_space_xgb(trial):
    search_space_xgb  = {
        'n_estimators'    : trial.suggest_int('n_estimators', 100, 1200, step=100),
        'max_depth'       : trial.suggest_int('max_depth', 1, 8, step=1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10, step=1),
        'max_leaves'      : trial.suggest_int('max_leaves', 1, 15, step=1),
        'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 1.1, step=0.01 ),
        'subsample'       : trial.suggest_float('subsample', 0.1, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
        'gamma'           : trial.suggest_float('gamma', 0, 1),
        'reg_alpha'       : trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda'      : trial.suggest_float('reg_lambda', 0, 1),
        } 
    return search_space_xgb


# ## Pronosticos Globales

# In[44]:


def simular_xgboost_global(periodo_max_evaluacion, 
                           porc_eval, 
                           meses_atras_fin, 
                           meses_a_pronosticar_evaluacion,
                           window_size,
                           df_mes, 
                           barra_progreso,
                           status_text):
    # Preprocesar datos
    df_horiz = preprocesar_ml(df_mes)
    data = pipe.fit_transform(df_horiz)
    meses_evaluar = calcular_meses_a_evaluar(data, periodo_max_evaluacion, porc_eval)
   
    print('meses a evaluar:', meses_evaluar)

    global_forecast = []
    total_periodos = meses_evaluar - meses_atras_fin
    barra_progreso.progress(0)  # Inicializar barra de progreso

    for idx, periodo in enumerate(range(meses_evaluar, meses_atras_fin, -1)):
        # Actualizar barra de progreso y mensaje de estado
        barra_progreso.progress((idx + 1) / total_periodos)
        status_text.text(f"Simulación {idx + 1} de {total_periodos}...")

        print(periodo)
        data_train, data_test, data_temp = partir_data(data, periodo, meses_a_pronosticar_evaluacion)
        freq, fechas_futuras_pronost, fechas_exog_futuras = preparar_vbles_exog(
            meses_a_pronosticar_evaluacion, data_train, data_test
        )
        target_cols = data_train.filter(like="DEMANDA_").columns

        forecaster_global_m1 = ForecasterRecursiveMultiSeries(
            regressor=XGBRegressor(tree_method="hist", enable_categorical=True),
            lags=[1, 2],
            window_features=RollingFeatures(
                stats=["mean", "std", "median", "max", "min"], window_sizes=window_size
            ),
        )

        cv = TimeSeriesFold(
            steps=1,
            initial_train_size=len(data_train),
            window_size=window_size,
            refit=1,
            fixed_train_size=False,
            gap=0,
            allow_incomplete_fold=True,
            skip_folds            = None,
            verbose               = False
        )

        results, best_trial = bayesian_search_forecaster_multiseries(
            forecaster=forecaster_global_m1,
            series=data_temp[target_cols],
            exog=data_temp[exog_cols_dtfs],
            search_space=search_space_xgb,
            cv=cv,
            levels= None,
            metric=custom_metric,
            n_trials=40,
            random_state          = None,
            return_best           = True,
            n_jobs                = 'auto',
            verbose               = False,
            show_progress         = True,
            suppress_warnings     = True,
            kwargs_create_study   = {},
            kwargs_study_optimize = {}
        )

        fecha_max = data_temp.index.max()
        fechas_futuras = pd.date_range(
            start=fecha_max + DateOffset(months=1), periods=meses_a_pronosticar_evaluacion, freq="MS"
        )
        df_futuro = pd.DataFrame(index=fechas_futuras)
        df_futuro["dummy"] = 1
        exog_futuro = pipe.fit_transform(df_futuro)
        exog_futuro = exog_futuro.drop("dummy", axis=1)
        predictions = forecaster_global_m1.predict(
            steps=meses_a_pronosticar_evaluacion, exog=exog_futuro[exog_cols_dtfs]
        )
        predictions["LAG"] = [f"Lag_{i}" for i in range(1, meses_a_pronosticar_evaluacion + 1)]
        global_forecast.append(predictions)

    # Finalizar barra de progreso y limpiar texto de estado
    barra_progreso.empty()
    if meses_atras_fin == 1:
         status_text.success("Pronósticos XGBoost Generados")
    else:
        status_text.text("Backtesting XGBoost Terminado")
    
    df_global_forecast = pd.concat(global_forecast)
    return df_global_forecast


# ## LGBM

# In[45]:


# Espacio de busqueda de hyperparametros
def search_space_lgbm(trial):
    search_space_lgbm = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1200, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 10, step=1),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 10, step=1),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1),
        'subsample': trial.suggest_float('subsample', 0.1, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
    }
    return search_space_lgbm


# In[46]:


def simular_lgbm_global(periodo_max_evaluacion, 
                        porc_eval, 
                        meses_atras_fin, 
                        meses_a_pronosticar_evaluacion,
                        window_size,
                        df_mes,
                        barra_progreso,
                           status_text):

    # Crear barra de progreso y área de estado
    #barra_progreso = st.progress(0)
    #status_test = st.empty()

    # Preprocesamiento inicial
    df_horiz = preprocesar_ml(df_mes)
    data = pipe.fit_transform(df_horiz)
    meses_evaluar = calcular_meses_a_evaluar(data, periodo_max_evaluacion, porc_eval)
    
    print('meses a evaluar:', meses_evaluar)
    global_forecast_lgbm = []
    total_periodos = meses_evaluar - meses_atras_fin
    barra_progreso.progress(0)  # Inicializar barra de progreso
    for i, periodo in enumerate(range(meses_evaluar, meses_atras_fin, -1)):
        # Actualizar barra de progreso y estado        
        barra_progreso.progress((i + 1) / total_periodos)
        status_text.text(f"Simulación {i + 1} de {total_periodos}...")

        # Dividir datos y preparar variables
        data_train, data_test, data_temp = partir_data(data, periodo, meses_a_pronosticar_evaluacion)
        freq, fechas_futuras_pronost, fechas_exog_futuras = preparar_vbles_exog(
            meses_a_pronosticar_evaluacion, data_train, data_test
        )
        target_cols = data_train.filter(like="DEMANDA_").columns

        forecaster_global_m2 = ForecasterRecursiveMultiSeries(
            regressor=LGBMRegressor(verbose=-1),
            lags=[1, 2, 3, 4, 5],
            window_features=RollingFeatures(stats=['mean', 'std', 'median', 'max', 'min'], window_sizes=window_size)
        )

        cv = TimeSeriesFold(
            steps=1,
            initial_train_size=len(data_train),
            window_size=window_size,
            refit=1,
            fixed_train_size=False,
            gap=0,
            allow_incomplete_fold=True,
            verbose=False,
            skip_folds= None,
        )

        results, best_trial = bayesian_search_forecaster_multiseries(
            forecaster=forecaster_global_m2,
            series=data_temp[target_cols],
            exog=data_temp[exog_cols_dtfs],
            search_space=search_space_lgbm,
            cv=cv,
            levels= None,
            metric=custom_metric,
            n_trials              = 30,
            random_state          = None,
            return_best           = True,
            n_jobs                = 'auto',
            verbose               = False,
            show_progress         = True,
            suppress_warnings     = True,
            kwargs_create_study   = {},
            kwargs_study_optimize = {}
        )

        # Generación de predicciones
        fecha_max = data_temp.index.max()
        fechas_futuras = pd.date_range(start=fecha_max + DateOffset(months=1), periods=meses_a_pronosticar_evaluacion, freq='MS')
        df_futuro = pd.DataFrame(index=fechas_futuras)
        df_futuro['dummy'] = 1
        exog_futuro = pipe.fit_transform(df_futuro).drop('dummy', axis=1)
        predictions = forecaster_global_m2.predict(
            steps=meses_a_pronosticar_evaluacion,
            exog=exog_futuro[exog_cols_dtfs]
        )
        predictions['LAG'] = [f'Lag_{i}' for i in range(1, meses_a_pronosticar_evaluacion + 1)]
        global_forecast_lgbm.append(predictions)

    # Concatenar predicciones globales
    df_global_forecast_lgbm = pd.concat(global_forecast_lgbm)

    # Finalizar barra de progreso y estado
    barra_progreso.empty()
    if meses_atras_fin == 1:
        status_text.success("Pronósticos LightGBM Generados")
    else:
        status_text.text("Backtesting LightGBM Terminado")

    return df_global_forecast_lgbm


# # Comparacion de Modelos

# ## Funciones para comparar y seleccionar mejores modelos por SKU

# ### Funcion para generar df con con los errores de la evaluacion

# In[47]:


# def generar_reporte_error_skus(modelos):
#     return {modelo: globals()[f'grupo_sku_error_formato_{modelo}'] for modelo in modelos}

def generar_reporte_error_skus(modelos):
    return {
        modelo: df for modelo in modelos 
        if (df := globals().get(f'grupo_sku_error_formato_{modelo}')) is not None and not df.empty
    }

def generar_reporte_error_skus_nv(modelos_nv):
    return {modelo_nv: globals()[f'grupo_sku_error_formato_{modelo_nv}_nv'] for modelo_nv in modelos_nv}


# ### Funcion para crear df con mejores modelos

# In[48]:


# Para pronosticos por meses
def comparar_y_graficar_modelos(reporte_error_skus, df_mes_ceros):
    # Crear el DataFrame base con la columna 'Codigo'
    #df_final = reporte_error_skus['pms'][['CODIGO']].copy()
    #df_final = pd.DataFrame({'CODIGO': df_mes_ceros['CODIGO'].unique(),'DESCRIPCION': df_mes_ceros['DESCRIPCION'].unique()})
    
    # Crear una columna temporal con el código alfanumérico antes del guion bajo
    df_mes_ceros['CODIGO_ALFA'] = df_mes_ceros['CODIGO'].str.split('_').str[0]
    
    # Crear un mapeo entre CODIGO_ALFA y DESCRIPCION
    descripcion_mapping = df_mes_ceros.drop_duplicates(subset='CODIGO_ALFA').set_index('CODIGO_ALFA')['DESCRIPCION']
    
    # Crear el DataFrame final con valores únicos de CODIGO
    df_final = df_mes_ceros[['CODIGO']].drop_duplicates().copy()
    
    # Asignar la descripción correspondiente a cada CODIGO usando el mapeo
    df_final['DESCRIPCION'] = df_final['CODIGO'].str.split('_').str[0].map(descripcion_mapping)
    
        
    
    # Iterar sobre los modelos para combinarlos en df_final
    # for nombre_modelo, df in reporte_error_skus.items():
    #     df_final = df_final.merge(
    #         #df[['CODIGO', 'MAE%']].rename(columns={'MAE%': nombre_modelo}), 
    #         df[['CODIGO','SCORE%']].rename(columns={'SCORE%': nombre_modelo}),
    #         on='CODIGO', 
    #         how='left'
    #     )
    #     df['MODELO'] = nombre_modelo

    for nombre_modelo, df in reporte_error_skus.items():
        if df is not None:  # Evitar errores por valores None
            df_final = df_final.merge(
                df[['CODIGO','SCORE%']].rename(columns={'SCORE%': nombre_modelo}),
                on='CODIGO', 
                how='left'
            )
            df['MODELO'] = nombre_modelo
        else:
            print(f"Advertencia: El modelo {nombre_modelo} está vacío (None).")
        
    # Remover simbolos de porcentaje y convertir columnas a valores numericos
    modelos_cols = list(reporte_error_skus.keys())
    df_final[modelos_cols] = df_final[modelos_cols].apply(lambda col: abs(col.str.rstrip('%').astype(float)))
    
    # Identificar la columna con el valor minimo para cada fila
    df_final['MEJOR_MODELO'] = df_final[modelos_cols].idxmin(axis=1)
    #dejar una copia sin formato porcentaje
    df_minimos = df_final.copy()
    # Dar formato a las columnas con un decimal y agregar el simbolo %
    df_final[modelos_cols] = df_final[modelos_cols].apply(lambda x: x.map('{:.1f}%'.format))
    
    # Contar cuantas veces el modelo es el mejor
    report = df_final['MEJOR_MODELO'].value_counts()
    
    # Preparar y crear la grafica de dona
    fig1 = go.Figure(data=[go.Pie(
        labels=report.index, 
        values=report.values, 
        hole=0.4,  
        textinfo='percent+label',  
        marker=dict(colors=px.colors.qualitative.Plotly)  
    )])
    
    # Actualizar Layout de la grafica
    fig1.update_layout(
        title='Distribucion de Mejor Modelo por SKUs',
        title_x=0.5,  
        template='plotly_white'  
    )
   


    # Concatenar todos los DataFrames en uno solo
    df_errores_totales = pd.concat(reporte_error_skus.values(), ignore_index=True) 
    
    return df_minimos, df_final, reporte_error_skus, fig1, df_errores_totales

# Solo para novaventa
def comparar_y_graficar_modelos_nv(reporte_error_skus, df_mes_ceros):
    # Crear el DataFrame base con la columna 'CODIGO'
    df_final = None

    # Filtrar los DataFrames válidos (no None y no vacíos)
    reporte_error_skus_validos = {nombre: df for nombre, df in reporte_error_skus.items() if df is not None and not df.empty}

    if not reporte_error_skus_validos:
        print("No hay modelos válidos para procesar.")
        return None, None, None, None, None

    # # Usar el primer DataFrame válido como base para 'CODIGO'
    # for nombre_modelo, df in reporte_error_skus_validos.items():
    #     if df_final is None:
    #         df_final = df[['CODIGO']].copy()
    #     break
        # Inicializar variables para el DataFrame base y su longitud
    df_final = None
    max_len = 0
    
    # Iterar sobre los DataFrames en el diccionario
    for nombre_modelo, df in reporte_error_skus_validos.items():
        # Verificar si el DataFrame actual tiene mayor longitud
        if len(df) > max_len:
            max_len = len(df)  # Actualizar la longitud máxima
            df_final = df[['CODIGO']].copy()  # Usar este DataFrame como base
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

    df_minimos = df_minimos.merge(
            df_mes_ceros[['CODIGO', 'DESCRIPCION']], 
            on='CODIGO', 
            how='left'
            )
    
    # Reordena las columnas para que DESCRIPCION esté justo después de CODIGO
    cols = list(df_minimos.columns)
    index_codigo = cols.index('CODIGO')  # Encuentra el índice de CODIGO
    # Inserta DESCRIPCION justo después de CODIGO
    cols.insert(index_codigo + 1, cols.pop(cols.index('DESCRIPCION')))
    df_minimos = df_minimos[cols].drop_duplicates()
    

    
    return df_minimos, df_final, reporte_error_skus_validos, fig1, df_errores_totales


# ### Funcion para acumular todos los pronosticos generados, no solo los de menor error

# In[49]:


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

# Solo para novaventa
def concatenar_forecasts_pronosticos_nv(modelos_nv):
    # Filtrar los DataFrames válidos (no None y no vacíos)
    dfs_validos_nv = [
        globals()[f'df_forecast_final_{modelo_nv}_nv']
        for modelo_nv in modelos_nv
        if globals()[f'df_forecast_final_{modelo_nv}_nv'] is not None and not globals()[f'df_forecast_final_{modelo_nv}_nv'].empty
    ]
    
    # Verificar si hay DataFrames válidos
    if not dfs_validos_nv:
        print("No hay pronósticos válidos para concatenar.")
        return None

    # Concatenar todos los DataFrames válidos en uno solo
    df_todos_pronosticos_nv = pd.concat(dfs_validos_nv)

    # Asegurar que la columna 'CODIGO' sea de tipo string
    df_todos_pronosticos_nv['CODIGO'] = df_todos_pronosticos_nv['CODIGO'].astype(str)

    return df_todos_pronosticos_nv


# ### Funcion para generar periodos futuros novaventa

# In[50]:


def generar_periodos_futuros(df_periodo, meses_a_pronosticar_produccion):
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
    
    for _ in range(meses_a_pronosticar_produccion):
        campaña_actual += 1
        if campaña_actual > 19:  # Reiniciar campañas después de la 19
            campaña_actual = 1
            año_actual += 1
        futuros.append(f"{año_actual:04d}-{campaña_actual:02d}")
    
    return periodo_max, futuros


# In[51]:


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


# ### Funcion para consolidar los RMSE de los modelos

# In[52]:


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

# Solo para novaventa
def concatenar_rmse_nv(modelos_nv):
    # Obtener los DataFrames dinámicamente usando la lista de modelos
    dfs_error_nv = []
    
    for modelo_nv in modelos_nv:
        # Obtener el DataFrame para cada modelo
        df = globals().get(f'rmse_sku_mes_{modelo_nv}_nv')
        
        # Verificar si el DataFrame es None o está vacío
        if df is None or df.empty:
            print(f"El modelo {modelo_nv} fue ignorado porque no tiene datos.")
            continue
        
        # Añadir una columna 'MODELO' con el nombre del modelo
        df['MODELO'] = modelo_nv
        df['RMSE'] = np.ceil(df['RMSE'])#.astype(int)
        
        # Añadir el DataFrame a la lista
        dfs_error_nv.append(df)
    
    # Verificar si hay DataFrames para concatenar
    if not dfs_error_nv:
        print("No hay datos para concatenar.")
        return pd.DataFrame()  # Devuelve un DataFrame vacío
    
    # Concatenar todos los DataFrames en uno solo
    df_todos_rmse_nv = pd.concat(dfs_error_nv, ignore_index=True)
    
    # Asegurar que la columna 'CODIGO' sea de tipo string
    df_todos_rmse_nv['CODIGO'] = df_todos_rmse_nv['CODIGO'].astype(str)

    return df_todos_rmse_nv


# ### Funcion para seleccionar el mejor modelo para cada sku

# In[53]:


def obtener_mejor_pronostico(df_minimos, df_todos_pronosticos, df_errores_totales, df_todos_rmse):
    # Crear una lista para almacenar los DataFrames filtrados
    lista_filtrados = [
        df_todos_pronosticos[
            (df_todos_pronosticos['CODIGO'] == row['CODIGO']) & 
            (df_todos_pronosticos['MODELO'] == row['MEJOR_MODELO'])
        ]
        for _, row in df_minimos.iterrows()
    ]
    
    # Concatenar todos los DataFrames filtrados
    df_pronosticos_mejor_modelo = pd.concat(lista_filtrados)
  
    # Pivotear el resultado para mostrar el forecast por Código, Modelo y Fecha
    #df_pronosticos_12_meses = df_pronosticos_mejor_modelo.pivot_table(index=["CODIGO", "MODELO"], columns="FECHA", values="FORECAST")#.reset_index()
    df_pronosticos_finales = df_pronosticos_mejor_modelo.pivot_table(index=["CODIGO", "MODELO"], columns="FECHA", values="FORECAST").reset_index()
    # Realizamos un merge para agregar las columnas coincidiendo por CODIGO y MODELO
    # Realiza el merge entre ambos DataFrames en las claves 'CODIGO' y 'MODELO'
    df_merged = pd.merge(
        df_pronosticos_finales, 
        df_errores_totales[['CODIGO', 'MODELO', 'MAE%', 'SESGO%', 'SCORE%']], 
        on=['CODIGO', 'MODELO'], 
        how='left'
    )
    df_merged_rmse = pd.merge(
        df_merged, 
        df_todos_rmse[['CODIGO', 'MODELO', 'RMSE']], 
        on=['CODIGO', 'MODELO'], 
        how='left'
    )
 
    # Inserta las columnas en las posiciones deseadas
    df_merged_rmse.insert(0, 'MAE%', df_merged_rmse.pop('MAE%'))
    df_merged_rmse.insert(1, 'SESGO%', df_merged_rmse.pop('SESGO%'))
    df_merged_rmse.insert(2, 'SCORE%', df_merged_rmse.pop('SCORE%'))
    df_merged_rmse.insert(3, 'RMSE', df_merged_rmse.pop('RMSE'))
    # Si deseas restaurar el índice anterior
    df_pronosticos_12_meses = df_merged_rmse.set_index(['CODIGO', 'MODELO'])
    
    # Filtrar columnas que parezcan fechas en formato yyyy-mm-dd
    columnas_tiempo = [col for col in df_pronosticos_12_meses.columns if '-' in str(col)]
    
    # Renombrar las columnas seleccionadas al formato yyyy-mm-dd
    df_pronosticos_12_meses = df_pronosticos_12_meses.rename(
        columns={col: pd.to_datetime(col).strftime('%Y-%m-%d') for col in columnas_tiempo})
    
    # Identificar las columnas de fechas
    columns_to_round = [col for col in df_pronosticos_12_meses.columns if col.startswith('20')]
    
    # Redondear los valores de estas columnas al entero más cercano
    df_pronosticos_12_meses[columns_to_round] = df_pronosticos_12_meses[columns_to_round].round().fillna(0).astype(int)

    df_pronosticos_12_meses = df_pronosticos_12_meses.reset_index().merge(
            df_minimos[['CODIGO', 'DESCRIPCION']], 
            on='CODIGO', 
            how='left'
            )
    
    # Reordena las columnas para que DESCRIPCION esté justo después de CODIGO
    cols = list(df_pronosticos_12_meses.columns)
    index_codigo = cols.index('CODIGO')  # Encuentra el índice de CODIGO
    # Inserta DESCRIPCION justo después de CODIGO
    cols.insert(index_codigo + 1, cols.pop(cols.index('DESCRIPCION')))
    df_pronosticos_12_meses = df_pronosticos_12_meses[cols]
    df_pronosticos_12_meses = df_pronosticos_12_meses.set_index(['CODIGO', 'DESCRIPCION', 'MODELO'])

    
    # Combina los DataFrames para obtener la columna DESCRIPCION correspondiente al CODIGO
    df_pronosticos_mejor_modelo = df_pronosticos_mejor_modelo.merge(
        df_minimos[['CODIGO', 'DESCRIPCION']], 
        on='CODIGO', 
        how='left'
    )
    
    # Reordena las columnas para que DESCRIPCION esté justo después de CODIGO
    cols = list(df_pronosticos_mejor_modelo.columns)
    index_codigo = cols.index('CODIGO')  # Encuentra el índice de CODIGO
    # Inserta DESCRIPCION justo después de CODIGO
    cols.insert(index_codigo + 1, cols.pop(cols.index('DESCRIPCION')))
    df_pronosticos_mejor_modelo = df_pronosticos_mejor_modelo[cols]

    
    return df_pronosticos_mejor_modelo, df_pronosticos_12_meses

# Solo para novaventa
def obtener_mejor_pronostico_nv(df_minimos, df_todos_pronosticos_fecha, 
                                df_errores_totales,
                               df_todos_rmse):
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
    df_pronosticos_finales = df_pronosticos_mejor_modelo.pivot_table(index=["CODIGO", "MODELO"], columns="FECHA", values="FORECAST")
        
    df_merged = pd.merge(
        df_pronosticos_finales, 
        df_errores_totales[['CODIGO', 'MODELO', 'MAE%', 'SESGO%', 'SCORE%']], 
        on=['CODIGO', 'MODELO'], 
        how='left'
    )
    df_merged_rmse = pd.merge(
        df_merged, 
        df_todos_rmse[['CODIGO', 'MODELO', 'RMSE']], 
        on=['CODIGO', 'MODELO'], 
        how='left'
    )
    
    # Inserta las columnas en las posiciones deseadas
    df_merged_rmse.insert(0, 'MAE%', df_merged_rmse.pop('MAE%'))
    df_merged_rmse.insert(1, 'SESGO%', df_merged_rmse.pop('SESGO%'))
    df_merged_rmse.insert(2, 'SCORE%', df_merged_rmse.pop('SCORE%'))
    df_merged_rmse.insert(3, 'RMSE', df_merged_rmse.pop('RMSE'))
    # Si deseas restaurar el índice anterior
    df_pronosticos_12_meses = df_merged_rmse.set_index(['CODIGO', 'MODELO'])

    # Identificar las columnas de fechas
    columns_to_round = [col for col in df_pronosticos_12_meses.columns if col.startswith('20')]
    
    # Redondear los valores de estas columnas al entero más cercano
    df_pronosticos_12_meses[columns_to_round] = df_pronosticos_12_meses[columns_to_round].round().fillna(0).astype(int)
    
    df_pronosticos_12_meses = df_pronosticos_12_meses.reset_index().merge(
            df_minimos[['CODIGO', 'DESCRIPCION']], 
            on='CODIGO', 
            how='left'
            )
    
    # Reordena las columnas para que DESCRIPCION esté justo después de CODIGO
    cols = list(df_pronosticos_12_meses.columns)
    index_codigo = cols.index('CODIGO')  # Encuentra el índice de CODIGO
    # Inserta DESCRIPCION justo después de CODIGO
    cols.insert(index_codigo + 1, cols.pop(cols.index('DESCRIPCION')))
    df_pronosticos_12_meses = df_pronosticos_12_meses[cols]
    df_pronosticos_12_meses = df_pronosticos_12_meses.set_index(['CODIGO', 'DESCRIPCION', 'MODELO'])

    
    return df_pronosticos_mejor_modelo, df_pronosticos_12_meses


# ### Funcion para crear la grafica de demanda + pronostico

# In[54]:


def crear_grafica_pronostico(df_mes_ceros, df_todos_pronosticos, df_pronosticos_mejor_modelo):
    # Obtener modelos únicos
    modelos_unicos = df_todos_pronosticos['MODELO'].unique()
    
    # Generar una paleta de colores en seaborn
    dark_colors = sns.color_palette("muted", n_colors=len(modelos_unicos)).as_hex()
    
    # Crear un diccionario para asignar colores a cada modelo
    color_mapping = {modelo: dark_colors[i] for i, modelo in enumerate(modelos_unicos)}

    # Lista de códigos únicos
    codigos_unicos = df_pronosticos_12_meses.index.unique(0)
    #codigos_unicos = df_mes_ceros["CODIGO"].unique()

    # Crear una figura
    fig = go.Figure()

    # Crear todas las trazas (una por cada Código y Modelo) y agregar al gráfico
    for codigo in codigos_unicos:
        # Filtrar df_mes por Codigo (para graficar la demanda)
        df_mes_filtrado = df_mes_ceros[df_mes_ceros["CODIGO"] == codigo]

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
            #name=f'{codigo}',
            name=df_mes_filtrado['DESCRIPCION'].iloc[0],
            line=dict(color='navy'),
            hovertemplate=(
                "<b>Fecha:</b> %{x}<br>"
                "<b>Demanda:</b> %{y}<br>"
                #f"<b>Descripción:</b> {df_mes_filtrado['DESCRIPCION'].iloc[0]}<br>"
                #"<extra></extra>"
                ),
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
                x=df_modelo_filtrado.index, 
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
                x=-0.05, y=1.0,  # Posición del dropdown
                xanchor="right",
                yanchor="top"
            )
        ],
        title="Demanda vs Pronóstico por Código",
        xaxis_title="Fecha",
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
    if 'CONSECUTIVO' in df_mes_ceros:
        fig.update_xaxes(        
            type='category',  # Especificar que el eje X es categórico
            tickmode='array',  # Asegurar que las etiquetas del eje X no sean interpretadas como fechas
            tickangle=-45,  # Rotar las etiquetas del eje X a 45 grados
            tickfont=dict(size=9)  # Reducir el tamaño de la fuente en un 25%
        ) 
    return fig


# Solo para novaventa
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
        
        #mejor_modelo = df_pronosticos_filtrado["MODELO"].values[0]  # Extraer el mejor modelo para ese código
        if df_pronosticos_filtrado.empty:
          
            continue  
    
        # Obtener el mejor modelo si el DataFrame no está vacío
        mejor_modelo = df_pronosticos_filtrado["MODELO"].values[0]
        # Agregar la traza de DEMANDA para este código (inicialmente invisible)
        fig.add_trace(go.Scatter(
            x=df_mes_filtrado.index, 
            y=df_mes_filtrado["DEMANDA"], 
            mode='lines',
            name=df_mes_filtrado['DESCRIPCION'].iloc[0],
            line=dict(color='navy'),
            hovertemplate=(
                "<b>Fecha:</b> %{x}<br>"
                "<b>Demanda:</b> %{y}<br>"
                #f"<b>Descripción:</b> {df_mes_filtrado['DESCRIPCION'].iloc[0]}<br>"
                #"<extra></extra>"
                ),
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


# ### Funcion para seleccionar un pronostico diferente al minimo estadistico

# In[55]:


def validar_pronosticos(sku, modelo, df_todos_pronosticos):
    
    # Filtrar DataFrame basado en selección
    df_filtrado = df_todos_pronosticos[
                (df_todos_pronosticos['CODIGO'] == sku) & 
                (df_todos_pronosticos['MODELO'] == modelo)
            ]     
    df_filtrado['FORECAST'] = np.ceil(df_filtrado['FORECAST']).astype(int)  
    return df_filtrado


# ### Funcion para agrupar df_test y obtener metricas combinadas

# In[56]:


def concatenar_df_test(modelos):
    dfs = []
    for modelo in modelos:
        df = globals().get(f'df_test_{modelo}')  # Usar .get() para evitar errores si no existe
        if df is not None:  # Solo añadir si no es None
            df = df.copy()  # Evitar SettingWithCopyWarning
            df['modelo'] = modelo  # Añadir la columna 'modelo'
            dfs.append(df)

    if not dfs:  # Si la lista está vacía, retornar un DataFrame vacío con las columnas esperadas
        return pd.DataFrame(columns=['CODIGO', 'modelo'])

    df_todos_df_test = pd.concat(dfs, ignore_index=True)  # Concatenar e ignorar los índices originales
    df_todos_df_test['CODIGO'] = df_todos_df_test['CODIGO'].astype(str)  # Asegurar que 'CODIGO' sea string

    return df_todos_df_test

# Solo para Novaventa
def concatenar_df_test_nv(modelos_nv):
    
    # Obtener los DataFrames dinámicamente usando la lista de modelos
    dfs_nv = []
    for modelo_nv in modelos_nv:
        df = globals()[f'df_test_{modelo_nv}']
        # Añadir la columna 'modelo' con el valor del modelo
        if df is not None:  # Solo añadir si no es None
            df = df.copy()  # Evitar SettingWithCopyWarning
            df['modelo'] = modelo_nv  # Añadir la columna 'modelo'
            dfs_nv.append(df)
            
    if not dfs_nv:  # Si la lista está vacía, retornar un DataFrame vacío con las columnas esperadas
        return pd.DataFrame(columns=['CODIGO', 'modelo'])
    # Concatenar todos los DataFrames en uno solo
    df_todos_df_test_nv = pd.concat(dfs_nv, ignore_index=True)
    
    # Asegurar que la columna 'CODIGO' sea de tipo string
    df_todos_df_test_nv['CODIGO'] = df_todos_df_test_nv['CODIGO'].astype(str)

    return df_todos_df_test_nv


# In[57]:


def filtrar_y_concatenatar_df_test(df_minimos, df_todos_df_test):
    # Lista para almacenar los DataFrames filtrados
    dfs_filtrados = []
    
    # Iterar sobre cada fila de df_minimos
    for _, row in df_minimos.iterrows():
        # Obtener CODIGO y MEJOR_MODELO de cada fila
        codigo = row['CODIGO']
        mejor_modelo = row['MEJOR_MODELO']
        
        # Filtrar df_todos_df_test por la combinación de CODIGO y MEJOR_MODELO
        df_filtrado = df_todos_df_test[(df_todos_df_test['CODIGO'] == codigo) & (df_todos_df_test['modelo'] == mejor_modelo)]
        
        # Añadir el DataFrame filtrado a la lista
        dfs_filtrados.append(df_filtrado)
    
    # Concatenar todos los DataFrames filtrados en uno solo, manteniendo los índices originales
    df_resultado_test = pd.concat(dfs_filtrados, ignore_index=False)
    
    return df_resultado_test


# # Script de prueba parte 1

# Esta celda es para probar el algoritmo y realizar cambios o ajsutes posteriores a la entrega, debe permancer inactivada y no debe hacer parte del codigo ejecutable

# inicio_tiempo = time.time()
# ## Cargar data
# #ruta_demanda = r'dataset/historico_venta 2022_2024.xlsx'
# ruta_demanda = r"dataset\Histórico venta 2023 - Corte Enero 2025.xlsx"
# df = cargar_data(ruta_demanda)
# #df = df.iloc[:,:-1]
# ## Preprocesar data
# df_vertical = convertir_a_df_vertical(df)
# df_vertical_fecha = convertir_texto_a_fecha(df_vertical, meses)
# df_resultado = eliminar_ceros_iniciales(df_vertical_fecha)
# df_mes_cliente_sin_filtro = preprocesar_tabla_2(df_resultado)
# 
# # Seleccionar Clientes
# clientes = df_mes_cliente_sin_filtro['CLIENTE'].unique()
# clientes_seleccionados = ['NOVAVENTA', 'DISTRIBUIDORES', 'FARMATODO', 'E-COMMERCE']
# df_mes_cliente = df_mes_cliente_sin_filtro[df_mes_cliente_sin_filtro['CLIENTE'].isin(clientes_seleccionados)]
# 
# ## Grafica 1: Demanda por Codigo - Cliente
# graficar_demanda_codigo_cliente(df_mes_cliente)
# 
# ## Grafica 2: Demanda por Codigo Agregado
# #df_mes_orig, reporte_codigos = agrupar_demanda(df_mes_cliente)
# #graficar_demanda_codigo(df_mes_orig)
# 
# ## Seleccionar tipo de Pronostico
# opciones = ['POR CODIGO CLIENTE','POR CODIGO AGREGADO']
# opcion = opciones[1]
# df_mes_orig, reporte_codigos = seleccionar_tipo_pronostico(opcion, df_mes_cliente)
# graficar_demanda_codigo(df_mes_orig)
# ## Reemplazar los Ceros por la mediana
# df_mes_ceros = reemplazar_ceros(df_mes_orig)
# 
# ## Grafica 3: Demanda con los ceros reemplazados
# graficar_demanda_codigo(df_mes_ceros)
# 
# ## Definir limites sup e inf
# sup = 0.98
# inf = 0.02
# 
# ## Definir n para el pronostico ingenuo
# n = 6
# 
# ## Imputar Outliers
# df_mes, df_outliers, reporte_outliers = eliminar_outliers(df_mes_ceros, sup, inf, n)
# 
# ## Grafica 4: Demanda sin Outliers
# graficar_demanda_codigo(df_mes)
# 
# ## Grafica 5: Visualizacion de outliers imputados
# graficar_outliers_subplots(df_mes_ceros, df_outliers, sup=sup, inf=inf, n=n)
# 
# ## Evaluar Modelos de Pronosticos de Serie de Tiempo
# lista_skus = crear_lista_skus(df_mes) # Crear lista de skus
# 
# ## Parametros
# meses_a_pronosticar_evaluacion = 6 # Numero de meses a pronosticar para evaluar y seleccionar el modelo
# periodo_max_evaluacion = 12 # Numero de periodos maximos de evaluacion de cada serie de tiempo
# porc_eval = 0.35 # Porcentaje de meses para evaluar el modelo
# barra_progreso = st.progress(0)
# status_text = st.text("Iniciando Evaluación...")
# 
# ## PMS
# df_mejor_n, df_forecast_pms = evaluar_y_generar_pms(df_mes, df_mes_ceros, lista_skus, 
#                                                     periodo_max_evaluacion, 
#                                                     porc_eval, 
#                                                     meses_a_pronosticar_evaluacion,
#                                                    barra_progreso,
#                                                    status_text)
# 
# ## Reportes de error PMS
# grupo_mes_error_formato_pms, df_test_pms = kpi_error_lag(df_forecast_pms) # Reporte global
# grupo_sku_error_formato_pms, rmse_sku_lag_pms, rmse_sku_mes_pms = kpi_error_sku(df_forecast_pms) # Reporte por sku
# 
# ## Generar Pronosticos finales con PMS
# meses_a_pronosticar_produccion = 12 # Numero de meses finales a pronosticar
# df_forecast_final_pms = construir_pronostico_pms(df_mejor_n, df_mes, meses_a_pronosticar_produccion, 'pms', barra_progreso, status_text)
# 
# ## Suavizacion Exponencial
# df_mejor_se,  df_forecast_se = encontrar_mejor_se(df_mes, 
#                                                   df_mes_ceros, 
#                                                   lista_skus, 
#                                                   periodo_max_evaluacion, 
#                                                   porc_eval, 
#                                                   meses_a_pronosticar_evaluacion,
#                                                   barra_progreso,
#                                                     status_text)
# 
# ## Reportes de error SE
# grupo_mes_error_formato_se, df_test_se = kpi_error_lag(df_forecast_se) # Reporte global
# grupo_sku_error_formato_se, rmse_sku_lag_se, rmse_sku_mes_se = kpi_error_sku(df_forecast_se) # Reporte por sku
# 
# ## Generar Pronosticos finales con SE
# porc_eval_pronost = 0 # Porcentaje de Evaluacion se lleva a 0 para pronosticar
# df_mejor_se_final,  df_forecast_final_se = encontrar_mejor_se(df_mes, df_mes_ceros, 
#                                                               lista_skus, 
#                                                               periodo_max_evaluacion, 
#                                                               porc_eval_pronost, 
#                                                               meses_a_pronosticar_produccion,
#                                                               barra_progreso,
#                                                                status_text)
# 
# ## Adicionar nombre a los pronosticos de SE
# df_forecast_final_se = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_se, 'se')
# 
# ## Regresion lineal simple y "estacional"
# df_mejor_rl_lineal, df_mejor_rl_estacional, df_forecast_rl_lineal, df_forecast_rl_estacional = aplicar_regresion_lineal_simple(lista_skus, df_mes, df_mes_ceros, 
#                                     periodo_max_evaluacion, porc_eval, 
#                                     meses_a_pronosticar_evaluacion,
#                                     barra_progreso,
#                                     status_text)
#                                                                                                                               
# 
# ## Reportes error RL
# grupo_mes_error_formato_rl_lineal, df_test_rl_lineal= kpi_error_lag(df_forecast_rl_lineal) # Reporte global RL simple
# grupo_sku_error_formato_rl_lineal, rmse_sku_lag_rl_lineal, rmse_sku_mes_rl_lineal = kpi_error_sku(df_forecast_rl_lineal) # Reporte por sku RL simple
# grupo_mes_error_formato_rl_estacional, df_test_rl_estacional= kpi_error_lag(df_forecast_rl_estacional) # Reporte global RL estacional
# grupo_sku_error_formato_rl_estacional, rmse_sku_lag_rl_estacional, rmse_sku_mes_rl_estacional = kpi_error_sku(df_forecast_rl_estacional) # Reporte por sku RL estacional
# 
# ## Generar Pronosticos finales con RL 
# df_final_mejor_rl_lineal, df_final_mejor_rl_estacional, df_forecast_final_rl_lineal, df_forecast_final_rl_estacional = aplicar_regresion_lineal_simple(lista_skus, df_mes, df_mes_ceros, 
#                                     periodo_max_evaluacion, porc_eval_pronost, 
#                                     meses_a_pronosticar_produccion,
#                                     barra_progreso,
#                                     status_text)
# 
# ## Adicionar nombre a los pronosticos de RL
# df_forecast_final_rl_lineal = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_rl_lineal, 'rl_lineal')
# df_forecast_final_rl_estacional = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_rl_estacional, 'rl_estacional')
# 
# ## Modelo de descomposicion MSTL
# porc_eval = 0.35
# peso_ult_data = 0.08
# df_mejor_mstl, df_forecast_mstl = aplicar_mstl(lista_skus, df_mes, df_mes_ceros, 
#                                     periodo_max_evaluacion, porc_eval, 
#                                     meses_a_pronosticar_evaluacion, peso_ult_data,
#                                     barra_progreso,
#                                     status_text)
# ## Reportes de error MSTL
# grupo_mes_error_formato_mstl, df_test_mstl = kpi_error_lag(df_forecast_mstl) # Reporte golbal
# grupo_sku_error_formato_mstl, rmse_sku_lag_mstl, rmse_sku_mes_mstl = kpi_error_sku(df_forecast_mstl) # Reporte por sku
# 
# ## Generar Pronosticos finales con MSTL
# tabla_final_pronost, df_forecast_final_mstl = aplicar_mstl(lista_skus, df_mes, df_mes_ceros, 
#                                     periodo_max_evaluacion, porc_eval_pronost, 
#                                     meses_a_pronosticar_produccion, 
#                                     peso_ult_data,
#                                     barra_progreso,
#                                     status_text)                      
# 
# ## Adicionar nombre a los pronosticos de MSTL                                                         
# df_forecast_final_mstl = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_mstl, 'mstl')
# 
# # Modelo Milagros
# df_forecast_milagros = aplicar_milagros(lista_skus, df_mes, df_mes_ceros, 
#                                     periodo_max_evaluacion, porc_eval, 
#                                     meses_a_pronosticar_evaluacion, 
#                                     barra_progreso,
#                                     status_text)
# 
# df_forecast_milagros = crear_columnas_error(df_forecast_milagros)
# df_mejor_milagros = agrupar_por_codigo(df_forecast_milagros)
# 
# grupo_mes_error_formato_milagros, df_test_milagros = kpi_error_lag(df_forecast_milagros)
# grupo_sku_error_formato_milagros, rmse_sku_lag_milagros, rmse_sku_mes_milagros = kpi_error_sku(df_forecast_milagros)
# 
# ## Generar pronostico Modelo Milagros
# 
# porc_eval = 0
# df_forecast_final_milagros = aplicar_milagros(lista_skus, df_mes, df_mes_ceros, 
#                                     periodo_max_evaluacion, porc_eval, 
#                                     meses_a_pronosticar_produccion, 
#                                     barra_progreso,
#                                     status_text)
# 
# df_forecast_final_milagros = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_milagros, 'milagros')
# #"""
# ## XGBoost
# 
# meses_atras_fin = 5
# window_size = 5
# porc_eval = 0.35
# df_forecast_xgboost_global_raw = simular_xgboost_global(periodo_max_evaluacion, 
#                            porc_eval, 
#                            meses_atras_fin, 
#                            meses_a_pronosticar_evaluacion,
#                            window_size,                        
#                            df_mes, barra_progreso,
#                            status_text)
# 
# df_forecast_xgboost_global = formato_pronosticos_globales(df_forecast_xgboost_global_raw, df_mes_ceros, pronostico_final=0)
# df_forecast_xgboost_global = crear_columnas_error(df_forecast_xgboost_global)
# df_mejor_xgboost_global = agrupar_por_codigo(df_forecast_xgboost_global)
# 
# ## Reportes de error XGBoost Global
# grupo_mes_error_formato_xgboost_global, df_test_xgboost_global = kpi_error_lag(df_forecast_xgboost_global)
# grupo_sku_error_formato_xgboost_global, rmse_sku_lag_xgboost_global, rmse_sku_mes_xgboost_global = kpi_error_sku(df_forecast_xgboost_global)
# 
# ## Generar Pronosticos finales con XGBoost
# periodo_max_evaluacion_xgboost = 2
# meses_atras_fin = 1
# window_size = 5
# 
# df_forecast_final_xgboost_global = simular_xgboost_global(periodo_max_evaluacion_xgboost, 
#                            porc_eval, 
#                            meses_atras_fin, 
#                            meses_a_pronosticar_produccion,
#                            window_size,                        
#                            df_mes, barra_progreso,
#                            status_text)
# 
# df_forecast_final_xgboost_global = formato_pronosticos_globales(df_forecast_final_xgboost_global, df_mes_ceros, pronostico_final=1)
# df_forecast_final_xgboost_global = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_xgboost_global, 'xgboost_global')
# 
# 
# ## LightGBM
# meses_atras_fin = 5
# window_size = 5
# porc_eval = 0.35
# df_forecast_lgbm_global_raw = simular_lgbm_global(periodo_max_evaluacion, 
#                            porc_eval, 
#                            meses_atras_fin, 
#                            meses_a_pronosticar_evaluacion,
#                            window_size,                        
#                            df_mes,
#                            barra_progreso,
#                            status_text)
# 
# df_forecast_lgbm_global = formato_pronosticos_globales(df_forecast_lgbm_global_raw, df_mes_ceros, pronostico_final=0)
# df_forecast_lgbm_global = crear_columnas_error(df_forecast_lgbm_global)
# df_mejor_lgbm_global = agrupar_por_codigo(df_forecast_lgbm_global)
# 
# ## Reportes de error lgbm Global
# grupo_mes_error_formato_lgbm_global, df_test_lgbm_global = kpi_error_lag(df_forecast_lgbm_global)
# grupo_sku_error_formato_lgbm_global, rmse_sku_lag_lgbm_global, rmse_sku_mes_lgbm_global = kpi_error_sku(df_forecast_lgbm_global)
# 
# ## Generar Pronosticos finales con lgbm
# periodo_max_evaluacion = 2
# meses_atras_fin = 1
# window_size = 5
# 
# df_forecast_final_lgbm_global = simular_lgbm_global(periodo_max_evaluacion, 
#                            porc_eval, 
#                            meses_atras_fin, 
#                            meses_a_pronosticar_produccion,
#                            window_size,                        
#                            df_mes,barra_progreso,
#                            status_text)
# 
# df_forecast_final_lgbm_global = formato_pronosticos_globales(df_forecast_final_lgbm_global, df_mes_ceros, pronostico_final=1)
# df_forecast_final_lgbm_global = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_lgbm_global, 'lgbm_global')
# 
# #"""
# 
# 
# ## Crear reporte acumulado de errores de todos los modelos
# modelos = ['pms', 'se', 'rl_lineal', 'rl_estacional', 'mstl', 'milagros',
#            #'xgboost_global',
#           #'lgbm_global'
#           ]
# reporte_error_skus = generar_reporte_error_skus(modelos)
# df_todos_rmse = concatenar_rmse(modelos)
# 
# ## Grafica 6: Distribucion de merjor modelos por sku
# df_minimos, df_final, reporte_error_skus, fig1, df_errores_totales = comparar_y_graficar_modelos(reporte_error_skus, df_mes_ceros)
# fig1.show()
# 
# ## Concatener todos los pronosticos finales generados
# df_todos_pronosticos = concatenar_forecasts_pronosticos(modelos)
# 
# ## Crear tabla final con pronosticos a 12 meses
# df_pronosticos_mejor_modelo, df_pronosticos_12_meses = obtener_mejor_pronostico(df_minimos, df_todos_pronosticos, df_errores_totales, df_todos_rmse)
# 
# ## Grafica final demanda vs mejor pronostico
# fig = crear_grafica_pronostico(df_mes_ceros, df_todos_pronosticos, df_pronosticos_mejor_modelo)
# fig.show()
# 
# df_todos_df_test = concatenar_df_test(modelos)
# 
# df_resultado_test = filtrar_y_concatenatar_df_test(df_minimos, df_todos_df_test)
# sesgo_porc, mae_porc, rmse, score = metricas_error(df_resultado_test, imprimir=1)
# df_lags = evaluar_lags(df_resultado_test)[['MAE%','SESGO%','SCORE%']]
# display(df_lags[['MAE%','SESGO%','SCORE%']])
# 
# # Medicion de tiempo de ejeucion
# fin_tiempo = time.time()
# tiempo_ejecucion = round(fin_tiempo - inicio_tiempo, 2)
# tiempo_ejecucion_min = tiempo_ejecucion / 60
# minutos, segundos = divmod(tiempo_ejecucion_min, 1)
# segundos *= 60
# 
# print("Tiempo de Ejecución:", "%02d:%02d" % (minutos, segundos))

# # Script de prueba parte 2 - Novaventa

# ## Ruta ubicacion de archivo fuente
# ruta_demanda_nv = r'dataset/Historico_Campañas_Novaventa.xlsx'
# df = cargar_data(ruta_demanda_nv)
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
# # Modelo Milagros
# df_forecasts_milagros_nv = aplicar_milagros_nv(lista_skus, df_periodo, df_ceros, 
#                                     periodo_max_evaluacion, porc_eval, 
#                                     meses_a_pronosticar_evaluacion, 
#                                     barra_progreso,
#                                     status_text)
# df_forecasts_milagros_nv = crear_columnas_error(df_forecasts_milagros_nv)
# df_mejor_milagros_nv = agrupar_por_codigo(df_forecasts_milagros_nv)
# print('reporte errores milagros:')
# grupo_mes_error_formato_milagros_nv, df_test_milagros_nv = kpi_error_lag(df_forecasts_milagros_nv)
# grupo_sku_error_formato_milagros_nv, rmse_sku_lag_milagros_nv, rmse_sku_mes_milagros_nv = kpi_error_sku(df_forecasts_milagros_nv)
# 
# 
# # Generar pronostcos finales con Modelo Milagros
# porc_eval_pronost = 0
# meses_a_pronosticar_produccion = 12
# df_forecast_final_milagros_nv = aplicar_milagros_nv(lista_skus, df_periodo, df_ceros, 
#                                     periodo_max_evaluacion, porc_eval_pronost, 
#                                     meses_a_pronosticar_produccion, 
#                                     barra_progreso,
#                                     status_text)
# 
# df_forecast_final_milagros_nv = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_milagros_nv, 'milagros')
# 
# # PMS
# 
# df_mejor_n_nv, df_forecast_pms_nv = evaluar_y_generar_pms_nv(df_periodo, df_ceros, lista_skus, 
#                                                     periodo_max_evaluacion, 
#                                                     porc_eval, 
#                                                     meses_a_pronosticar_evaluacion,
#                                                    barra_progreso,
#                                                    status_text)
# print('reporte errores pms:')
# grupo_mes_error_formato_pms_nv, df_test_pms_nv = kpi_error_lag(df_forecast_pms_nv) # Reporte global
# grupo_sku_error_formato_pms_nv, rmse_sku_lag_pms_nv, rmse_sku_mes_pms_nv = kpi_error_sku(df_forecast_pms_nv) # Reporte por sku
# 
# ## Generar Pronosticos finales con PMS
# meses_a_pronosticar_produccion = 12 # Numero de meses finales a pronosticar
# df_forecast_final_pms_nv = construir_pronostico_pms_nv(df_mejor_n_nv, df_periodo, meses_a_pronosticar_produccion, 'pms',barra_progreso,  status_text)
# 
# ## Suavizacion Exponencial
# df_mejor_se_nv,  df_forecast_se_nv = encontrar_mejor_se_nv(df_periodo, df_ceros, lista_skus, periodo_max_evaluacion, porc_eval, 
#                         meses_a_pronosticar_evaluacion,
#                         barra_progreso,
#                         status_text)
# print('reporte errores se:')
# grupo_mes_error_formato_se_nv, df_test_se_nv = kpi_error_lag(df_forecast_se_nv) # Reporte global
# grupo_sku_error_formato_se_nv, rmse_sku_lag_se_nv, rmse_sku_mes_se_nv = kpi_error_sku(df_forecast_se_nv)
# 
# 
# meses_a_pronosticar_produccion = 12
# df_mejor_se_final_nv,  df_forecast_final_se_nv = encontrar_mejor_se_nv(df_periodo, df_ceros, lista_skus, periodo_max_evaluacion, porc_eval_pronost, 
#                         meses_a_pronosticar_produccion,
#                         barra_progreso,
#                         status_text)
# ## Adicionar nombre a los pronosticos de SE
# df_forecast_final_se_nv = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_se_nv, 'se')
# 
# # Regresion lineal
# df_mejor_rl_lineal_nv, df_mejor_rl_estacional_nv, df_forecast_rl_lineal_nv, df_forecast_rl_estacional_nv = aplicar_regresion_lineal_simple_nv(lista_skus, df_periodo, df_ceros, 
#                                     periodo_max_evaluacion, porc_eval, 
#                                     meses_a_pronosticar_evaluacion,
#                                     barra_progreso,
#                                     status_text)
# print('reporte errores rl_lineal:')
# grupo_mes_error_formato_rl_lineal_nv, df_test_rl_lineal_nv = kpi_error_lag(df_forecast_rl_lineal_nv) # Reporte global
# grupo_sku_error_formato_rl_lineal_nv, rmse_sku_lag_rl_lineal_nv, rmse_sku_mes_rl_lineal_nv = kpi_error_sku(df_forecast_rl_lineal_nv)
# print('reporte errores rl_lineal_estacional:')
# grupo_mes_error_formato_rl_estacional_nv, df_test_rl_estacional_nv = kpi_error_lag(df_forecast_rl_estacional_nv) # Reporte global
# grupo_sku_error_formato_rl_estacional_nv, rmse_sku_lag_rl_estacional_nv, rmse_sku_mes_rl_estacional_nv = kpi_error_sku(df_forecast_rl_estacional_nv)
# 
# 
# df_final_mejor_rl_lineal_nv, df_final_mejor_rl_estacional_nv, df_forecast_final_rl_lineal_nv, df_forecast_final_rl_estacional_nv = aplicar_regresion_lineal_simple_nv(lista_skus, df_periodo, df_ceros, 
#                                     periodo_max_evaluacion, porc_eval_pronost, 
#                                     meses_a_pronosticar_produccion,
#                                     barra_progreso,
#                                     status_text)
# 
# ## Adicionar nombre a los pronosticos de RL
# df_forecast_final_rl_lineal_nv = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_rl_lineal_nv, 'rl_lineal')
# df_forecast_final_rl_estacional_nv = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_rl_estacional_nv, 'rl_estacional')
# 
# ## Modelo de descomposicion MSTL
# peso_ult_data = 0.08
# df_mejor_mstl_nv, df_forecast_mstl_nv = aplicar_mstl_nv(lista_skus, df_periodo, df_ceros, 
#                                     periodo_max_evaluacion, porc_eval, 
#                                     meses_a_pronosticar_evaluacion, peso_ult_data,
#                                     barra_progreso,
#                                     status_text)
# ## Reportes de error MSTL
# print('reporte errores mstl:')
# grupo_mes_error_formato_mstl_nv, df_test_mstl_nv = kpi_error_lag(df_forecast_mstl_nv) # Reporte golbal
# grupo_sku_error_formato_mstl_nv, rmse_sku_lag_mstl_nv, rmse_sku_mes_mstl_nv = kpi_error_sku(df_forecast_mstl_nv) # Reporte por sku
# 
# ## Generar Pronosticos finales con MSTL
# tabla_final_pronost_nv, df_forecast_final_mstl_nv = aplicar_mstl_nv(lista_skus, df_periodo, df_ceros, 
#                                     periodo_max_evaluacion, porc_eval_pronost, 
#                                     meses_a_pronosticar_produccion, 
#                                     peso_ult_data,
#                                     barra_progreso,
#                                     status_text)                      
# 
# ## Adicionar nombre a los pronosticos de MSTL                                                         
# df_forecast_final_mstl_nv = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_mstl_nv, 'mstl')
# modelos_nv = ['milagros','pms', 'se', 'rl_lineal', 'rl_estacional', 'mstl']
# reporte_error_skus_nv = generar_reporte_error_skus_nv(modelos_nv)
# df_todos_rmse_nv = concatenar_rmse_nv(modelos_nv)
# df_minimos_nv, df_final_nv, reporte_error_skus_nv, fig1_nv, df_errores_totales_nv = comparar_y_graficar_modelos_nv(reporte_error_skus_nv, df_ceros)
# fig1_nv.show()
# n = 12
# periodo_max, futuros = generar_periodos_futuros(df_periodo, n)
# df_todos_pronosticos_nv = concatenar_forecasts_pronosticos_nv(modelos_nv)
# df_todos_pronosticos_fecha = agregar_fecha_a_grupo(df_todos_pronosticos_nv, futuros)
# df_pronosticos_mejor_modelo_nv, df_pronosticos_12_meses_nv = obtener_mejor_pronostico_nv(df_minimos_nv, 
#                                                                                          df_todos_pronosticos_fecha,
#                                                                                         df_errores_totales_nv,
#                                                                                         df_todos_rmse_nv)
# fig_nv = crear_grafica_pronostico_nv(df_periodo, df_todos_pronosticos_fecha, df_pronosticos_mejor_modelo_nv)
# fig_nv.show()

# # Front end Streamlit

# ## Seccion 1

# In[ ]:


# Configurar el layout de Streamlit
st.set_page_config(layout="wide")

# Título de la aplicación
st.title("Pronósticos de Series de Tiempo MILAGROS")

# Barra lateral
st.sidebar.title('Flujo de Datos')

# Secciones
seccion = st.sidebar.radio('⬇️ Ir a:', ('📂 Carga de datos',
                                        '📊 Demanda a pronosticar y outliers',
                                        '🔮 Evaluar y Generar Pronosticos', 
                                        '🛠️ Herramientas de Análisis',
                                        '📦 Pronosticos Novaventa por Campaña')
                          )

# Variables globales
session_vars = ['df','df_vertical','df_mes_cliente','df_mes_orig', 'df_mes_ceros', 'df_mes', 'df_outliers',                            
                'sup', 'inf', 'n', 'df_vertical', 'df_mes_cliente_sin_filtro',
                'meses_a_pronosticar_evaluacion', 'meses_a_pronosticar_produccion', 'periodo_max_evaluacion',
                'porc_eval', 'porc_eval_pronost','df_todos_pronosticos',                 
                'codigo_seleccionado', 'modelo_seleccionado', 'mostrar_grafica_cliente', 
                'mostrar_grafica_codigo', 'df_pronosticos_12_meses', 'reporte_outliers', 'fig',
                'sesgo_porc_formato', 'mae_porc_formato', 'score_formato',               
                'df_periodo','mostrar_grafica_outliers','df_pronosticos_12_meses_nv','df_nv','df_ceros', 'df_orig_nv', 'df_vertical_nv', 'df_resultado_nv', 'mostrar_grafica', 'sup_nv', 'inf_nv', 'n_nv', 'reporte_outliers_nv',
                'fig_nv','df_todos_pronosticos_nv', 'meses_a_pronosticar_evaluacion_nv', 'meses_a_pronosticar_produccion_nv', 'periodo_max_evaluacion_nv', 'porc_eval_nv','porc_eval_pronost_nv','pronosticos_generados_nv',
                'codigo_seleccionado_nv','modelo_seleccionado_nv', 'df_outliers_nv', 'excel_data_nv', 'clientes_seleccionados'
               ]

# Inicializar session_state si no existe
for var in session_vars:
    if var not in st.session_state:
        st.session_state[var] = None

# Sección: Carga de Datos
if seccion == '📂 Carga de datos':
    # Resetear estados de gráficas para evitar mostrar gráficas automáticamente
    st.session_state.mostrar_grafica_cliente = False
    st.session_state.mostrar_grafica_codigo = False
    
    st.header("Cargar Datos")

    # Comprobar si los datos ya están cargados
    if st.session_state.df is not None:
        st.success('Datos ya cargados previamente.')
        st.write("Muestra Aleatoria de los Datos cargados:")
        st.write(st.session_state.df.sample(3))

    else:
        # Subida de datos
        ruta_demanda = st.file_uploader("Sube el archivo de demanda en formato Excel", type=['xlsx'])
        if ruta_demanda is not None:
            # Cargar y procesar datos
            df = cargar_data(ruta_demanda)
            st.session_state.df = df
            st.success("Archivo histórico cargado correctamente.")
            st.write("Muestra Aleatoria de los Datos cargados:")
            st.write(st.session_state.df.sample(5))

    # Selección de clientes
    if st.session_state.df is not None:
        # Procesar datos si no se ha hecho previamente
        if st.session_state.df_vertical is None:
            df_vertical = convertir_a_df_vertical(st.session_state.df)
            df_vertical_fecha = convertir_texto_a_fecha(df_vertical, meses)
            df_resultado = eliminar_ceros_iniciales(df_vertical_fecha)
            df_mes_cliente_sin_filtro = preprocesar_tabla_2(df_resultado)
            st.session_state.df_vertical = df_vertical
            st.session_state.df_mes_cliente_sin_filtro = df_mes_cliente_sin_filtro

        # Mostrar clientes para seleccionar
        clientes = st.session_state.df_mes_cliente_sin_filtro['CLIENTE'].unique()
        clientes_seleccionados = st.multiselect(
            "Seleccione los clientes a trabajar:",
            options=clientes,
            default=st.session_state.clientes_seleccionados or clientes
        )
        st.session_state.clientes_seleccionados = clientes_seleccionados

        if clientes_seleccionados:
            df_mes_cliente = st.session_state.df_mes_cliente_sin_filtro[
                st.session_state.df_mes_cliente_sin_filtro['CLIENTE'].isin(clientes_seleccionados)
            ]
            st.session_state.df_mes_cliente = df_mes_cliente
            st.success(f"{len(clientes_seleccionados)} clientes seleccionados:")
            #st.markdown("### Clientes seleccionados:")
            st.markdown("\n".join([f"✓  **{cliente}**" for cliente in clientes_seleccionados]))
            #st.text(clientes_seleccionados)
            

        else:
            st.warning("No se han seleccionado clientes. Por favor, elija al menos uno.")

    # Mostrar gráficas solo si hay clientes seleccionados
    st.write("")
    if st.session_state.df_mes_cliente is not None and len(st.session_state.clientes_seleccionados) > 0:
        st.header("Ver Gráficas de Demanda")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Graficar Demanda por Código - Cliente"):
                st.text("Se muestran gráficas para los clientes:")
                st.markdown("\n".join([f"▪️ *{cliente}*" for cliente in st.session_state.clientes_seleccionados]))
                st.session_state.mostrar_grafica_cliente = True
                st.session_state.mostrar_grafica_codigo = False
        with col2:
            if st.button("Graficar Demanda por Código Agregado"):
                st.text("Se muestran gráficas para la demanda agrupada de los clientes:")
                st.markdown("\n".join([f"▪️ *{cliente}*" for cliente in st.session_state.clientes_seleccionados]))
                st.session_state.mostrar_grafica_cliente = False
                st.session_state.mostrar_grafica_codigo = True

        # Mostrar las gráficas según la selección
        if st.session_state.mostrar_grafica_cliente:
            graficar_demanda_codigo_cliente(st.session_state.df_mes_cliente)
        elif st.session_state.mostrar_grafica_codigo:
            df_mes_orig, reporte_codigos = agrupar_demanda(st.session_state.df_mes_cliente)
            graficar_demanda_codigo(df_mes_orig)


# ## Seccion 2

# In[59]:


if seccion == '📊 Demanda a pronosticar y outliers':
      
    # Seleccionar tipo de pronóstico
    if "df" in st.session_state and "df_mes_cliente" in st.session_state:
        st.header("Seleccionar Tipo de Pronóstico")
        opciones = ['POR CODIGO AGREGADO','POR CODIGO CLIENTE',] 
    
        if 'opcion_seleccionada' not in st.session_state:
            st.session_state['opcion_seleccionada'] = opciones[0]  # Selección por defecto

        # Mostrar el menu desplegable y guardar la decision en session_state
        st.session_state['opcion_seleccionada'] = st.selectbox(
                "Selecciona cuales demandas deseas pronosticar", 
                opciones, 
                index=opciones.index(st.session_state['opcion_seleccionada'])
                )
        opcion = st.session_state['opcion_seleccionada']
            
            
        df_mes_orig, reporte_codigos = seleccionar_tipo_pronostico(opcion, st.session_state.df_mes_cliente)
        st.success(f"Modelo seleccionado: {opcion}")
        st.write("")
        st.text("Se pronosticarán los siguientes clientes:")
        st.markdown("\n".join([f"▪️ {cliente}" for cliente in st.session_state.clientes_seleccionados]))   
        df_mes_ceros = reemplazar_ceros(df_mes_orig)    
        st.session_state.df_mes_orig = df_mes_orig
        st.session_state.df_mes_ceros = df_mes_ceros
        st.write("Referencias con un solo dato, no seran pronosticadas:")
        st.write(reporte_codigos)

    
    # Parámetros de configuración
    if "df_mes_orig" in st.session_state and "df_mes_ceros" in st.session_state:
        st.header("Manejo de Outliers")
        st.write("Si no desea imputar los outliers, seleccione [1] en Limite Superior y [0] en Limite Inferior")
        sup = st.number_input("Límite Superior", min_value=0.0, max_value=1.0, value=0.98, step=0.01)
        inf = st.number_input("Límite Inferior", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
        n = st.number_input("Número de Periodos para Pronóstico Ingenuo", min_value=1, max_value=12, value=6, step=1)
        st.write("Un valor pequeño puede ser menos sensible a los outliers, un valor alto (6 o más) puede aumentar la sensiblidad")
        df_mes, df_outliers, reporte_outliers = eliminar_outliers(st.session_state.df_mes_ceros, sup, inf, n)
        st.session_state.df_mes = df_mes
        st.session_state.df_outliers = df_outliers
        st.session_state.reporte_outliers = reporte_outliers
        st.session_state.sup = sup
        st.session_state.inf = inf
        st.session_state.n = n  
        st.success("Outliers Imputados correctamente.")
        
    if "df_mes" in st.session_state and "df_outliers" in st.session_state:

        # Exportar a excel outliers
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.reporte_outliers.to_excel(writer, index=True, sheet_name='Outliers')
            
        excel_data = output.getvalue()
        
        # Boton de descarga a excel
        st.download_button(
            label="📥 Descargar Outliers (Excel)",
            data=excel_data,
            file_name="df_outliers.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.header("Generar Gráfica Manejo Outliers")
        if st.button("Grafica Imputacion de Outliers"):
            graficar_outliers_subplots(st.session_state.df_mes_ceros, 
                                       st.session_state.df_outliers, 
                                       sup=st.session_state.sup, 
                                       inf=st.session_state.inf, 
                                       n=st.session_state.n)


# ## Seccion 3

# In[60]:


if seccion == '🔮 Evaluar y Generar Pronosticos':
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
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Sesgo% Combinado", value=st.session_state.sesgo_porc_formato)
            with col2:
                st.metric(label="Mae% Combinado", value=st.session_state.mae_porc_formato)
            with col3:
                st.metric(label="Score% Combinado", value=st.session_state.score_formato)
                    
            st.dataframe(st.session_state.df_pronosticos_12_meses)
            st.plotly_chart(st.session_state.fig)

    else:    
        if "df_mes" in st.session_state and "df_mes_ceros" in st.session_state:
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
        if "df_mes" in st.session_state and "df_mes_ceros" in st.session_state:
            st.header("Evaluar y Generar Pronósticos")
            st.text("Los pronósticos se calcularán:")
            st.markdown(st.session_state['opcion_seleccionada'])
            st.text("Y se generarán para los siguientes clientes:")
            st.markdown("\n".join([f"▪️ *{cliente}*" for cliente in st.session_state.clientes_seleccionados]))
            st.write("")
            # Diccionario de equivalencias entre nombres amigables y los modelos internos
            equivalencias_modelos = {
                'Modelo Milagros': ['milagros'],
                'Promedio Móvil Simple': ['pms'],
                'Suavización Exponencial': ['se'],
                'Regresión Lineal': ['rl_lineal', 'rl_estacional'],
                'MSTL': ['mstl'],
                'XGBoost': ['xgboost_global'],
                'LightGBM': ['lgbm_global']
            }
            # Selección de modelos a ejecutar
            modelos_disponibles = list(equivalencias_modelos.keys())
            modelos_seleccionados = st.multiselect(
                "Seleccione los modelos a ejecutar:",
                options=modelos_disponibles,
                default=modelos_disponibles  # Seleccionados por defecto
            )
            modelos = []
            for modelo in modelos_seleccionados:
                modelos.extend(equivalencias_modelos[modelo])
                
            if st.button("Evaluar y Generar"):
                      
                st.session_state.pronosticos_generados = True
                lista_skus = crear_lista_skus(st.session_state.df_mes)


                if 'Modelo Milagros' in modelos_seleccionados:
                    # Modelo Milagros
                    barra_progreso = st.progress(0)
                    status_text = st.text("Iniciando Evaluación Modelo Milagros...")
                    with st.spinner('Evaluando Modelo Milagros:'):
                        df_forecast_milagros = aplicar_milagros(lista_skus, st.session_state.df_mes, 
                                                                st.session_state.df_mes_ceros, 
                                        st.session_state.periodo_max_evaluacion, 
                                        st.session_state.porc_eval, 
                                        st.session_state.meses_a_pronosticar_evaluacion, 
                                        barra_progreso,
                                        status_text)
    
                        df_forecast_milagros = crear_columnas_error(df_forecast_milagros)
                        df_mejor_milagros = agrupar_por_codigo(df_forecast_milagros)
    
                        grupo_mes_error_formato_milagros, df_test_milagros = kpi_error_lag(df_forecast_milagros)
                        grupo_sku_error_formato_milagros, rmse_sku_lag_milagros, rmse_sku_mes_milagros = kpi_error_sku(df_forecast_milagros)
                    
                    barra_progreso = st.empty()
                    status_text = st.empty()
    
                    # Generar pronósticos finales Modelo Milagros
                    barra_progreso = st.progress(0)
                    status_text = st.text("Pronosticando con Modelo Milagros...") 
    
                    with st.spinner('Generando Pronóstico Modelo Milagros:'):
                        
                        df_forecast_final_milagros = aplicar_milagros(lista_skus, 
                                                                      st.session_state.df_mes, 
                                                                      st.session_state.df_mes_ceros, 
                                                                      st.session_state.periodo_max_evaluacion, 
                                                                      st.session_state.porc_eval_pronost, 
                                                                      st.session_state.meses_a_pronosticar_produccion, 
                                                                      barra_progreso,
                                                                      status_text)
                        
                        df_forecast_final_milagros = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_milagros, 'milagros')
    
                    
                    barra_progreso = st.empty()
                    status_text = st.empty()
                    
                if 'Promedio Móvil Simple' in modelos_seleccionados:
                    # PMS
                    barra_progreso = st.progress(0)
                    status_text = st.text("Iniciando Evaluación PMS...")
                    with st.spinner('Evaluando Promedio Móvil Simple:'):
                        df_mejor_n, df_forecast_pms = evaluar_y_generar_pms(st.session_state.df_mes,
                                                                            st.session_state.df_mes_ceros, 
                                                                            lista_skus, 
                                                                            st.session_state.periodo_max_evaluacion, 
                                                                            st.session_state.porc_eval, 
                                                                            st.session_state.meses_a_pronosticar_evaluacion,
                                                                           barra_progreso,
                                                                           status_text)
                        
                        grupo_mes_error_formato_pms, df_test_pms = kpi_error_lag(df_forecast_pms) # Reporte global
                        grupo_sku_error_formato_pms, rmse_sku_lag_pms, rmse_sku_mes_pms = kpi_error_sku(df_forecast_pms) 
                    
                    barra_progreso = st.empty()
                    status_text = st.empty() 
                    
                    # Generar pronósticos finales PMS
                    barra_progreso = st.progress(0)
                    status_text = st.text("Iniciando Generacion Pronóstico con PMS...")     
                    
                    with st.spinner('Generando Promedio Móvil Simple:'):
                        df_forecast_final_pms = construir_pronostico_pms(df_mejor_n, 
                                                                         st.session_state.df_mes,
                                                                         st.session_state.meses_a_pronosticar_produccion, 
                                                                         'pms',barra_progreso,  status_text)
                    barra_progreso = st.empty()
                    status_text = st.empty()
    
                # Sueavizacion Exponencial
                if 'Suavización Exponencial' in modelos_seleccionados:
                    barra_progreso = st.progress(0)
                    status_text = st.text("Iniciando Evaluación SE...") 
                    with st.spinner('Evaluando Suavizacion Exponencial Simple:'):
                        df_mejor_se,  df_forecast_se = encontrar_mejor_se(st.session_state.df_mes,
                                                                          st.session_state.df_mes_ceros, 
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
                                    
                    # Generar pronósticos finales Suavización Exponencial
                    barra_progreso = st.progress(0)
                    status_text = st.text("Pronosticando con SE...")     
                    with st.spinner('Generando Suavización Exponencial:'):
                        
                        df_mejor_se_final,  df_forecast_final_se = encontrar_mejor_se(st.session_state.df_mes, 
                                                                                      st.session_state.df_mes_ceros, 
                                                                                      lista_skus, 
                                                                                      st.session_state.periodo_max_evaluacion,
                                                                                      st.session_state.porc_eval_pronost,                                                                           
                                                                                      st.session_state.meses_a_pronosticar_produccion,
                                                                                      barra_progreso,
                                                                                      status_text)
                        
                        df_forecast_final_se = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_se, 'se')
                    
                    barra_progreso = st.empty()
                    status_text = st.empty()
                    
                if 'Regresión Lineal' in modelos_seleccionados:
                    # Regresiones lineales
                    barra_progreso = st.progress(0)
                    status_text = st.text("Iniciando Evaluación RL...") 
                    with st.spinner('Evaluando Regresion Lineal Simple y "Estacional":'):
                        df_mejor_rl_lineal, df_mejor_rl_estacional, df_forecast_rl_lineal, df_forecast_rl_estacional = aplicar_regresion_lineal_simple(lista_skus, 
                                                                                                                                                       st.session_state.df_mes, 
                                                                                                                                                       st.session_state.df_mes_ceros,
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
    
                    # Generar pronósticos finales regresiones lineales
                    barra_progreso = st.progress(0)
                    status_text = st.text("Pronosticando con RL...") 
                    with st.spinner('Generando Regresion Lineal Simple y "Estacional":'):
                        df_final_mejor_rl_lineal, df_final_mejor_rl_estacional, df_forecast_final_rl_lineal, df_forecast_final_rl_estacional = aplicar_regresion_lineal_simple(lista_skus, 
                                                                                                                                                                                st.session_state.df_mes, 
                                                                                                                                                                                st.session_state.df_mes_ceros,
                                                                                                                                                                                st.session_state.periodo_max_evaluacion, 
                                                                                                                                                                                st.session_state.porc_eval_pronost, 
                                                                                                                                                                                st.session_state.meses_a_pronosticar_produccion,
                                                                                                                                                                                barra_progreso,
                                                                                                                                                                                status_text)
                        
                        df_forecast_final_rl_lineal = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_rl_lineal, 'rl_lineal')
                        df_forecast_final_rl_estacional = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_rl_estacional, 'rl_estacional')
                    barra_progreso = st.empty()
                    status_text = st.empty() 
                    
                if 'MSTL' in modelos_seleccionados:
                    # MSTL
                    barra_progreso = st.progress(0)
                    status_text = st.text("Iniciando Evaluación MSTL...") 
                    with st.spinner('Evaluando MSTL:'):
                        peso_ult_data = 0.08 
                        df_mejor_mstl, df_forecast_mstl = aplicar_mstl(lista_skus, st.session_state.df_mes, 
                                                                       st.session_state.df_mes_ceros, 
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
    
                    # Generar pronósticos finales MSTL
                    barra_progreso = st.progress(0)
                    status_text = st.text("Pronosticando con MSTL...") 
                    with st.spinner('Generando MSTL:'):
        
                        tabla_final_pronost, df_forecast_final_mstl = aplicar_mstl(lista_skus, 
                                                                                   st.session_state.df_mes, 
                                                                                   st.session_state.df_mes_ceros, 
                                                                                   st.session_state.periodo_max_evaluacion, 
                                                                                   st.session_state.porc_eval_pronost, 
                                                                                   st.session_state.meses_a_pronosticar_produccion, 
                                                                                   peso_ult_data, 
                                                                                   barra_progreso, 
                                                                                   status_text)
                        
                        df_forecast_final_mstl = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_mstl, 'mstl')
                    
                    barra_progreso = st.empty()
                    status_text = st.empty()
                    
                if 'XGBoost' in modelos_seleccionados:
                    # XGboost 
                    meses_atras_fin = 5
                    window_size = 5             
                    barra_progreso = st.progress(0)                
                    status_text = st.text("Iniciando Evaluación Xgboost...") 
                    with st.spinner('Evaluando Xgboost:'):
                        df_forecast_xgboost_global_raw = simular_xgboost_global(st.session_state.periodo_max_evaluacion,
                               st.session_state.porc_eval, 
                               meses_atras_fin, 
                               st.session_state.meses_a_pronosticar_evaluacion,
                               window_size,                        
                               st.session_state.df_mes,
                                barra_progreso,                           
                                status_text)
    
                        df_forecast_xgboost_global = formato_pronosticos_globales(df_forecast_xgboost_global_raw, 
                                                                st.session_state.df_mes_ceros,  
                                                                pronostico_final=0)
                        
                        df_forecast_xgboost_global = crear_columnas_error(df_forecast_xgboost_global)
                        df_mejor_xgboost_global = agrupar_por_codigo(df_forecast_xgboost_global)
    
                        ## Reportes de error XGBoost Global
                        grupo_mes_error_formato_xgboost_global, df_test_xgboost_global = kpi_error_lag(df_forecast_xgboost_global)
                        grupo_sku_error_formato_xgboost_global, rmse_sku_lag_xgboost_global, rmse_sku_mes_xgboost_global = kpi_error_sku(df_forecast_xgboost_global)
                    
                    barra_progreso = st.empty()
                    status_text = st.empty()
    
                    # Generar pronósticos finales XGBoost
                    barra_progreso = st.progress(0)
                    status_text = st.text("Pronosticando con XGBoost...") 
                    periodo_max_evaluacion_xgboost = 2
                    meses_atras_fin = 1
                    window_size = 5
                    with st.spinner('Generando XGboost:'):                   
                        df_forecast_final_xgboost_global = simular_xgboost_global(periodo_max_evaluacion_xgboost, 
                               st.session_state.porc_eval, 
                               meses_atras_fin, 
                               st.session_state.meses_a_pronosticar_produccion,
                               window_size,                        
                               st.session_state.df_mes, barra_progreso, 
                               status_text)
    
                        df_forecast_final_xgboost_global = formato_pronosticos_globales(df_forecast_final_xgboost_global, st.session_state.df_mes, pronostico_final=1)
                        df_forecast_final_xgboost_global = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_xgboost_global, 'xgboost_global')
                                    
                    barra_progreso = st.empty()
                    status_text = st.empty()

                if 'LightGBM' in modelos_seleccionados:
                    # LightGBM
                    meses_atras_fin = 5
                    window_size = 5              
                    barra_progreso = st.progress(0)                
                    status_text = st.text("Iniciando Evaluación LightGBM...") 
                    with st.spinner('Evaluando LightGBM:'):
                        df_forecast_lgbm_global_raw  = simular_lgbm_global(st.session_state.periodo_max_evaluacion,
                               st.session_state.porc_eval, 
                               meses_atras_fin, 
                               st.session_state.meses_a_pronosticar_evaluacion,
                               window_size,                        
                               st.session_state.df_mes,
                                barra_progreso, 
                                status_text)
    
                        df_forecast_lgbm_global = formato_pronosticos_globales(df_forecast_lgbm_global_raw, 
                                                                st.session_state.df_mes_ceros,  
                                                                pronostico_final=0)
                        
                        df_forecast_lgbm_global = crear_columnas_error(df_forecast_lgbm_global)
                        df_mejor_xgboost_global = agrupar_por_codigo(df_forecast_lgbm_global)
    
                        ## Reportes de error XGBoost Global
                        grupo_mes_error_formato_lgbm_global, df_test_lgbm_global = kpi_error_lag(df_forecast_lgbm_global)
                        grupo_sku_error_formato_lgbm_global, rmse_sku_lag_lgbm_global, rmse_sku_mes_lgbm_global = kpi_error_sku(df_forecast_lgbm_global)
                    
                    barra_progreso = st.empty()
                    status_text = st.empty()
    
                    # Generar pronósticos LightGBM
                    barra_progreso = st.progress(0)
                    status_text = st.text("Pronosticando con LightGBM...") 
                    periodo_max_evaluacion_lgbm = 2
                    meses_atras_fin = 1
                    window_size = 5
                    with st.spinner('Generando pronóstico LightGBM:'):                   
                        df_forecast_final_lgbm_global = simular_lgbm_global(periodo_max_evaluacion_lgbm, 
                               st.session_state.porc_eval, 
                               meses_atras_fin, 
                               st.session_state.meses_a_pronosticar_produccion,
                               window_size,                        
                               st.session_state.df_mes, barra_progreso,
                               status_text)
    
                        df_forecast_final_lgbm_global = formato_pronosticos_globales(df_forecast_final_lgbm_global, st.session_state.df_mes, pronostico_final=1)
                        df_forecast_final_lgbm_global = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_lgbm_global, 'lgbm_global')
                    
                    barra_progreso = st.empty()
                    status_text = st.empty()
                

                st.success("✅✨ ¡Modelos Calculados Correctamente! 🎯🚀")
                st.balloons()
                
                #modelos = ['milagros', 'pms', 'se', 'rl_lineal', 'rl_estacional', 'mstl', 'xgboost_global', 'lgbm_global' ]
                
                reporte_error_skus = generar_reporte_error_skus(modelos)
                df_todos_rmse = concatenar_rmse(modelos)
                df_todos_df_test = concatenar_df_test(modelos)
                df_minimos, df_final, reporte_error_skus, fig1, df_errores_totales = comparar_y_graficar_modelos(reporte_error_skus,
                                                                                                                st.session_state.df_mes_ceros)
                with st.expander("Mostrar Estadisticas de Modelos de Pronosticos"):
                    st.plotly_chart(fig1)
                
                
                df_todos_pronosticos = concatenar_forecasts_pronosticos(modelos)
                st.session_state.df_todos_pronosticos = df_todos_pronosticos
                
                df_resultado_test = filtrar_y_concatenatar_df_test(df_minimos, df_todos_df_test)
                st.subheader("Metricas de Error Combinadas")
                sesgo_porc, mae_porc, rmse, score = metricas_error(df_resultado_test, imprimir=0)
                
                # Convertir a formato porcentaje
                sesgo_porc_formato = f"{sesgo_porc * 100:.2f}%"
                mae_porc_formato = f"{mae_porc * 100:.2f}%"
                score_formato = f"{score * 100:.2f}%"
                st.session_state.sesgo_porc_formato = sesgo_porc_formato
                st.session_state.mae_porc_formato = mae_porc_formato
                st.session_state.score_formato = score_formato
                
                # Mostrar Metricas con streamlit
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Sesgo% Combinado", value=st.session_state.sesgo_porc_formato)
                with col2:
                    st.metric(label="Mae% Combinado", value=st.session_state.mae_porc_formato)
                with col3:
                    st.metric(label="Score% Combinado", value=st.session_state.score_formato)

                df_lags_metricas = evaluar_lags(df_resultado_test)[['MAE%','SESGO%','SCORE%']]
                df_lags_metricas = df_lags_metricas.applymap(lambda x: f"{x:.2%}")                
                st.dataframe(df_lags_metricas, 
                             #use_container_width=True
                            )
                df_pronosticos_mejor_modelo, df_pronosticos_12_meses = obtener_mejor_pronostico(df_minimos, 
                                                                                                st.session_state.df_todos_pronosticos, 
                                                                                                df_errores_totales, df_todos_rmse)
                                                                                               
                st.session_state.df_pronosticos_12_meses = df_pronosticos_12_meses
                # Mostrar resultados
                st.subheader("Pronósticos a 12 meses")
                st.write(df_pronosticos_12_meses)

                # Exportar pronosticos a excel 
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    st.session_state.df_pronosticos_12_meses.to_excel(writer, index=True, sheet_name='Pronosticos')
                    
                excel_data = output.getvalue()
                
                # Boton de descarga a excel
                st.download_button(
                    label="📥 Descargar Pronosticos xlxs",
                    data=excel_data,
                    file_name="pronosticos.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
                # Mostrar gráfica final
                fig = crear_grafica_pronostico(st.session_state.df_mes, df_todos_pronosticos, df_pronosticos_mejor_modelo)
                st.session_state.fig = fig
                st.plotly_chart(fig)


# ## Seccion 4

# In[61]:


if seccion == '🛠️ Herramientas de Análisis':
                
    # Sección de filtrado (solo se muestra si df_todos_pronosticos existe)
    if 'df_todos_pronosticos' in st.session_state and st.session_state.df_todos_pronosticos is not None:
        st.header("Filtrar Pronósticos")
        st.text('Desea usar otro pronóstico diferente al sugerido estadísticamente?')
        
        # Menú desplegable para seleccionar el código
        codigo_seleccionado = st.selectbox(
            "Seleccione el Código:",
            options=st.session_state.df_todos_pronosticos['CODIGO'].unique()
        )
        
        # Menú desplegable para seleccionar el modelo
        modelo_seleccionado = st.selectbox(
            "Seleccione el Modelo:",
            options=st.session_state.df_todos_pronosticos['MODELO'].unique()
        )
        if st.button('Validar Series de Tiempo'):
            df_filtrado = validar_pronosticos(codigo_seleccionado, 
                                          modelo_seleccionado, 
                                          st.session_state.df_todos_pronosticos)
            
            st.write('Datos de pronostico para codigo y modelo seleccionado:')
            st.dataframe(df_filtrado)


# ## Seccion 5

# In[62]:


if seccion == '📦 Pronosticos Novaventa por Campaña': 
    st.markdown("<h1 style='color:navy;'>Novaventa</h1>", unsafe_allow_html=True)
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
        
        if st.session_state.df_nv is not None:
            st.success('Datos ya cargados previamente')
            st.write("Datos cargados:")
            st.write(st.session_state.df_nv.head())
            if st.button('Cargar Nuevos Datos'):
                for var in session_vars:
                    st.session_state[var] = None
                st.experimental_rerun()
        else:
            ruta_demanda_nv = st.file_uploader("Sube el archivo de demanda en formato Excel", type=['xlsx'])
            if ruta_demanda_nv is not None:        
                df_nv = cargar_data(ruta_demanda_nv)
                st.success("Archivo histórico cargado correctamente.")
                st.session_state.df_nv = df_nv
                st.write("Datos cargados:")
                st.write(st.session_state.df_nv.head())
                df_orig_nv = preprocesar_datos_1_nv(df_nv)
                df_vertical_nv = llenar_nan(df_orig_nv)
                st.session_state.df_resultado_nv = eliminar_ceros_iniciales_nv(df_vertical_nv)
                st.session_state.df_ceros = reemplazar_ceros_nv(st.session_state.df_resultado_nv)
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
                graficar_demanda_codigo_nv(st.session_state.df_resultado_nv)
            elif st.session_state.mostrar_grafica == "sin_ceros":
                st.subheader("Gráfica: Demanda Sin Ceros")
                graficar_demanda_codigo_nv(st.session_state.df_ceros)

    with tabs[1]:
        # Parámetros de configuración
        if "df_ceros" in st.session_state and st.session_state.df_ceros is not None:
            st.header("Manejo de Outliers")
            sup_nv = st.number_input("Límite Superior", min_value=0.0, max_value=1.0, value=0.98, step=0.01)
            inf_nv = st.number_input("Límite Inferior", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
            n_nv = st.number_input("Número de Periodos para Pronóstico Ingenuo", min_value=1, max_value=12, value=6, step=1)
            df_periodo, df_outliers_nv, reporte_outliers_nv = eliminar_outliers(st.session_state.df_ceros, sup_nv, inf_nv, n_nv)
            st.session_state.df_periodo = df_periodo
            st.session_state.df_outliers_nv = df_outliers_nv
            st.session_state.reporte_outliers_nv = reporte_outliers_nv
            st.session_state.sup_nv = sup_nv
            st.session_state.inf_nv = inf_nv
            st.session_state.n_nv = n_nv  
            st.success("Outliers Imputados correctamente.")
                
        #if "df_periodo" in st.session_state and "df_outliers_nv" in st.session_state:
        if "df_outliers_nv" in st.session_state and st.session_state.df_outliers_nv is not None:
            excel_data_env = None
            # Exportar a excel outliers            
            #if st.session_state.df_outliers_nv is not None:
            output_nv = io.BytesIO()
            with pd.ExcelWriter(output_nv, engine='openpyxl') as writer:
                st.session_state.reporte_outliers_nv.to_excel(writer, index=False, sheet_name='Outliers')
                    
            excel_data_nv = output_nv.getvalue()
                
            #else:
                #st.warning("No hay datos procesados aun para exportar.")
            # Boton de descarga a excel
            if excel_data_nv is not None:
                st.download_button(
                        label="📥 Descargar Outliers (Excel)",
                        data=excel_data_nv,
                        file_name="df_outliers_nv.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            st.header("Generar Gráfica Manejo Outliers")
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
                                           st.session_state.df_outliers_nv, 
                                           sup=st.session_state.sup_nv, 
                                           inf=st.session_state.inf_nv, 
                                           n=st.session_state.n_nv)


    with tabs[2]:
         # Chequear si los pronosticos ya han sido generados
        if st.session_state.df_pronosticos_12_meses_nv is not None and st.session_state.fig_nv is not None and st.session_state.df_todos_pronosticos_nv is not None:
            st.success("Pronósticos ya generados previamente.")
            # Boton para regenerar pronosticos
            if st.button('Regenerar Pronósticos'):
                # Limpiar los pronosticos previos y volver a correr script de la seccion
                st.session_state.df_pronosticos_12_meses_nv = None
                st.session_state.fig_nv = None
                st.session_state.df_todos_pronosticos_nv = None
                st.experimental_rerun()  # Volver a correr el script
            else: 
                st.dataframe(st.session_state.df_pronosticos_12_meses_nv)
                st.plotly_chart(st.session_state.fig_nv)
        
        else:    
            if "df_periodo" in st.session_state and "df_ceros" in st.session_state:
                st.header("Parámetros de evaluación de los modelos")
                meses_a_pronosticar_evaluacion_nv = st.number_input("Meses a Pronosticar para Evaluación", 
                                                                 min_value=1, max_value=24, value=6, step=1)
                meses_a_pronosticar_produccion_nv = st.number_input("Meses a Pronosticar para Produccion", 
                                                                 min_value=1, max_value=24, value=12, step=1)
                periodo_max_evaluacion_nv = 12
                porc_eval_nv = st.number_input("Porcentaje de Evaluación", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
                porc_eval_pronost_nv = 0
            
                st.session_state.meses_a_pronosticar_evaluacion_nv = meses_a_pronosticar_evaluacion_nv
                st.session_state.meses_a_pronosticar_produccion_nv = meses_a_pronosticar_produccion_nv
                st.session_state.periodo_max_evaluacion_nv = periodo_max_evaluacion_nv
                st.session_state.porc_eval_nv = porc_eval_nv
                st.session_state.porc_eval_pronost_nv = porc_eval_pronost_nv
        
            # Evaluar y generar pronósticos
            if "df_periodo" in st.session_state and "df_ceros" in st.session_state:
                st.header("Evaluar y Generar Pronósticos")
                
                if st.button("Evaluar y Generar"):
                          
                    st.session_state.pronosticos_generados_nv = True
                    lista_skus_nv = crear_lista_skus(st.session_state.df_periodo)

                    # Modelo Milagros
                    barra_progreso = st.progress(0)
                    status_text = st.text("Iniciando Evaluación Modelo Milagros...")
                    with st.spinner('Evaluando Modelo Milagros:'):
                        
                        df_forecasts_milagros_nv = aplicar_milagros_nv(lista_skus_nv, 
                                                                       st.session_state.df_periodo,
                                                                       st.session_state.df_ceros, 
                                                                       st.session_state.periodo_max_evaluacion_nv, 
                                                                       st.session_state.porc_eval_nv, 
                                                                       st.session_state.periodo_max_evaluacion_nv, 
                                                                       barra_progreso,
                                                                       status_text)
                        
                        df_forecasts_milagros_nv = crear_columnas_error(df_forecasts_milagros_nv)
                        df_mejor_milagros_nv = agrupar_por_codigo(df_forecasts_milagros_nv)
                        
                        grupo_mes_error_formato_milagros_nv, df_test_milagros_nv = kpi_error_lag(df_forecasts_milagros_nv)
                        grupo_sku_error_formato_milagros_nv, rmse_sku_lag_milagros_nv, rmse_sku_mes_milagros_nv = kpi_error_sku(df_forecasts_milagros_nv)
                    
                    barra_progreso = st.empty()
                    status_text = st.empty()    
                    # Generar pronostcos finales con Modelo Milagros
                    with st.spinner('Generando Pronosticos Modelo Milagros:'):   
                        
                        df_forecast_final_milagros_nv = aplicar_milagros_nv(lista_skus_nv, 
                                                                            st.session_state.df_periodo,
                                                                            st.session_state.df_ceros,  
                                                                            st.session_state.periodo_max_evaluacion_nv, 
                                                                            st.session_state.porc_eval_pronost_nv, 
                                                                            st.session_state.meses_a_pronosticar_produccion_nv, 
                                                                            barra_progreso,
                                                                            status_text)
                        
                        df_forecast_final_milagros_nv = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_milagros_nv, 'milagros')
                    
                    barra_progreso = st.empty()
                    status_text = st.empty()
                    
                    # Evaluacion PMS
                    barra_progreso = st.progress(0)
                    status_text = st.text("Iniciando Evaluación PMS...")
                    with st.spinner('Evaluando Promedio Móvil Simple:'):
                        df_mejor_n_nv, df_forecast_pms_nv = evaluar_y_generar_pms_nv(st.session_state.df_periodo,
                                                                            st.session_state.df_ceros, 
                                                                            lista_skus_nv, 
                                                                            st.session_state.periodo_max_evaluacion_nv, 
                                                                            st.session_state.porc_eval_nv, 
                                                                            st.session_state.meses_a_pronosticar_evaluacion_nv,
                                                                           barra_progreso,
                                                                           status_text)
                        
                        grupo_mes_error_formato_pms_nv, df_test_pms_nv = kpi_error_lag(df_forecast_pms_nv) # Reporte global
                        grupo_sku_error_formato_pms_nv, rmse_sku_lag_pms_nv, rmse_sku_mes_pms_nv = kpi_error_sku(df_forecast_pms_nv) 
                    
                    barra_progreso = st.empty()
                    status_text = st.empty()     
                    # Generar pronósticos finales
                    with st.spinner('Generando Promedio Móvil Simple:'):
                        df_forecast_final_pms_nv = construir_pronostico_pms_nv(df_mejor_n_nv, 
                                                                         st.session_state.df_periodo,
                                                                         st.session_state.meses_a_pronosticar_produccion_nv, 
                                                                         'pms',barra_progreso,  status_text)
                    barra_progreso = st.empty()
                    status_text = st.empty()
                    
                    barra_progreso = st.progress(0)
                    status_text = st.text("Iniciando Evaluación SE...") 
                    with st.spinner('Evaluando Suavizacion Exponencial Simple:'):
                        df_mejor_se_nv,  df_forecast_se_nv = encontrar_mejor_se_nv(st.session_state.df_periodo,
                                                                          st.session_state.df_ceros, 
                                                                          lista_skus_nv, 
                                                                          st.session_state.periodo_max_evaluacion_nv, 
                                                                          st.session_state.porc_eval_nv, 
                                                                          st.session_state.meses_a_pronosticar_evaluacion_nv,
                                                                           barra_progreso,
                                                                           status_text)
                        
                        grupo_mes_error_formato_se_nv, df_test_se_nv = kpi_error_lag(df_forecast_se_nv) 
                        grupo_sku_error_formato_se_nv, rmse_sku_lag_se_nv, rmse_sku_mes_se_nv = kpi_error_sku(df_forecast_se_nv)
                    
                    barra_progreso = st.empty()
                    status_text = st.empty()                
                    barra_progreso = st.progress(0)
                    status_text = st.text("Pronosticando con SE...")     
                    with st.spinner('Generando Suavización Exponencial:'):
                        
                        df_mejor_se_final_nv,  df_forecast_final_se_nv = encontrar_mejor_se_nv(st.session_state.df_periodo, 
                                                                                      st.session_state.df_ceros, 
                                                                                      lista_skus_nv, 
                                                                                      st.session_state.periodo_max_evaluacion_nv,
                                                                                      st.session_state.porc_eval_pronost_nv,                                                                           
                                                                                      st.session_state.meses_a_pronosticar_produccion_nv,
                                                                                      barra_progreso,
                                                                                      status_text)
                        
                        df_forecast_final_se_nv = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_se_nv, 'se')
                    
                    barra_progreso = st.empty()
                    status_text = st.empty()   
                    barra_progreso = st.progress(0)
                    status_text = st.text("Iniciando Evaluación RL...") 
                    with st.spinner('Evaluando Regresion Lineal Simple y "Estacional":'):
                        df_mejor_rl_lineal_nv, df_mejor_rl_estacional_nv, df_forecast_rl_lineal_nv, df_forecast_rl_estacional_nv = aplicar_regresion_lineal_simple_nv(lista_skus_nv, 
                                                                                                                                                       st.session_state.df_periodo, 
                                                                                                                                                       st.session_state.df_ceros,
                                                                                                                                                       st.session_state.periodo_max_evaluacion_nv,
                                                                                                                                                       st.session_state.porc_eval_nv, 
                                                                                                                                                       st.session_state.meses_a_pronosticar_evaluacion_nv,
                                                                                                                                                        barra_progreso,
                                                                                                                                                       status_text)
        
                        grupo_mes_error_formato_rl_lineal_nv, df_test_rl_lineal_nv= kpi_error_lag(df_forecast_rl_lineal_nv) # Reporte global RL simple
                        grupo_sku_error_formato_rl_lineal_nv, rmse_sku_lag_rl_lineal_nv, rmse_sku_mes_rl_lineal_nv = kpi_error_sku(df_forecast_rl_lineal_nv) # Reporte por sku RL simple
                        grupo_mes_error_formato_rl_estacional_nv, df_test_rl_estacional_nv= kpi_error_lag(df_forecast_rl_estacional_nv) # Reporte global RL estacional
                        grupo_sku_error_formato_rl_estacional_nv, rmse_sku_lag_rl_estacional_nv, rmse_sku_mes_rl_estacional_nv = kpi_error_sku(df_forecast_rl_estacional_nv) # Reporte por sku RL estacional
        
                    barra_progreso = st.empty()
                    status_text = st.empty() 
                    barra_progreso = st.progress(0)
                    status_text = st.text("Pronosticando con RL...") 
                    with st.spinner('Generando Regresion Lineal Simple y "Estacional":'):
                        df_final_mejor_rl_lineal_nv, df_final_mejor_rl_estacional_nv, df_forecast_final_rl_lineal_nv, df_forecast_final_rl_estacional_nv = aplicar_regresion_lineal_simple_nv(lista_skus_nv, 
                                                                                                                                                                                st.session_state.df_periodo, 
                                                                                                                                                                                st.session_state.df_ceros,
                                                                                                                                                                                st.session_state.periodo_max_evaluacion_nv, 
                                                                                                                                                                                st.session_state.porc_eval_pronost_nv, 
                                                                                                                                                                                st.session_state.meses_a_pronosticar_produccion_nv,
                                                                                                                                                                                barra_progreso,
                                                                                                                                                                                status_text)
                        
                        df_forecast_final_rl_lineal_nv = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_rl_lineal_nv, 'rl_lineal')
                        df_forecast_final_rl_estacional_nv = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_rl_estacional_nv, 'rl_estacional')
                    barra_progreso = st.empty()
                    status_text = st.empty() 
                    barra_progreso = st.progress(0)
                    status_text = st.text("Iniciando Evaluación MSTL...") 
                    with st.spinner('Evaluando MSTL:'):
                        peso_ult_data_nv = 0.08 
                        df_mejor_mstl_nv, df_forecast_mstl_nv = aplicar_mstl_nv(lista_skus_nv, st.session_state.df_periodo, 
                                                                       st.session_state.df_ceros, 
                                                                       st.session_state.periodo_max_evaluacion_nv, 
                                                                       st.session_state.porc_eval_nv, 
                                                                       st.session_state.meses_a_pronosticar_evaluacion_nv, 
                                                                       peso_ult_data_nv, 
                                                                       barra_progreso, 
                                                                       status_text)
                        
                        grupo_mes_error_formato_mstl_nv, df_test_mstl_nv = kpi_error_lag(df_forecast_mstl_nv) # Reporte golbal
                        grupo_sku_error_formato_mstl_nv, rmse_sku_lag_mstl_nv, rmse_sku_mes_mstl_nv = kpi_error_sku(df_forecast_mstl_nv) # Reporte por sku
                    barra_progreso = st.empty()
                    status_text = st.empty()
                    
                    barra_progreso = st.progress(0)
                    status_text = st.text("Pronosticando con MSTL...") 
                    with st.spinner('Generando MSTL:'):
        
                        tabla_final_pronost_nv, df_forecast_final_mstl_nv = aplicar_mstl_nv(lista_skus_nv, 
                                                                                   st.session_state.df_periodo, 
                                                                                   st.session_state.df_ceros, 
                                                                                   st.session_state.periodo_max_evaluacion_nv, 
                                                                                   st.session_state.porc_eval_pronost_nv, 
                                                                                   st.session_state.meses_a_pronosticar_produccion_nv, 
                                                                                   peso_ult_data_nv, 
                                                                                   barra_progreso, 
                                                                                   status_text)
                        
                        df_forecast_final_mstl_nv = adicionar_nombre_modelo_serie_tiempo(df_forecast_final_mstl_nv, 'mstl')
                    
                    barra_progreso = st.empty()
                    status_text = st.empty()  
        
                    st.success("✅✨ ¡Modelos Calculados Correctamente! 🎯🚀")
                    #st.balloons()
                    
                    modelos_nv = ['milagros','pms', 'se', 'rl_lineal', 'rl_estacional', 'mstl']
                    
                    reporte_error_skus_nv = generar_reporte_error_skus_nv(modelos_nv)
                    df_todos_rmse_nv = concatenar_rmse_nv(modelos_nv)
                    df_minimos_nv, df_final_nv, reporte_error_skus_nv, fig1_nv, df_errores_totales_nv = comparar_y_graficar_modelos_nv(reporte_error_skus_nv, st.session_state.df_ceros)
                    with st.expander("Mostrar Estadisticas de Modelos de Pronosticos"):
                        st.plotly_chart(fig1_nv)
                    periodo_max, futuros = generar_periodos_futuros(df_periodo, st.session_state.meses_a_pronosticar_produccion_nv)            
                    df_todos_pronosticos_nv = concatenar_forecasts_pronosticos_nv(modelos_nv)
                    df_todos_pronosticos_fecha = agregar_fecha_a_grupo(df_todos_pronosticos_nv, futuros)
                    st.session_state.df_todos_pronosticos_nv = df_todos_pronosticos_nv
                    st.session_state.df_todos_pronosticos_fecha  = df_todos_pronosticos_fecha 
                    
                    df_pronosticos_mejor_modelo_nv, df_pronosticos_12_meses_nv = obtener_mejor_pronostico_nv(df_minimos_nv, 
                                                                                                    st.session_state.df_todos_pronosticos_fecha, 
                                                                                                    df_errores_totales_nv, 
                                                                                                    df_todos_rmse_nv
                                                                                                   )
                                                                                                   
                    st.session_state.df_pronosticos_12_meses_nv = df_pronosticos_12_meses_nv
                    # Mostrar resultados
                    st.subheader("Pronósticos proximas campañas")
                    st.write(df_pronosticos_12_meses_nv)
            
                    # Mostrar gráfica final
                    fig_nv = crear_grafica_pronostico_nv(st.session_state.df_periodo, st.session_state.df_todos_pronosticos_fecha, df_pronosticos_mejor_modelo_nv)
                    st.session_state.fig = fig_nv
                    st.plotly_chart(fig_nv)

    with tabs[3]:
        # Sección de filtrado (solo se muestra si df_todos_pronosticos existe)
        if 'df_todos_pronosticos_fecha' in st.session_state and st.session_state.df_todos_pronosticos_fecha is not None:
            st.header("Filtrar Pronósticos")
            st.text('Desea usar otro pronóstico diferente al sugerido estadísticamente?')
            
            # Menú desplegable para seleccionar el código
            codigo_seleccionado_nv = st.selectbox(
                "Seleccione el Código:",
                options=st.session_state.df_todos_pronosticos_fecha['CODIGO'].unique()
            )
            
            # Menú desplegable para seleccionar el modelo
            modelo_seleccionado_nv = st.selectbox(
                "Seleccione el Modelo:",
                options=st.session_state.df_todos_pronosticos_fecha['MODELO'].unique()
            )
            if st.button('Validar Series de Tiempo'):
                df_filtrado_nv = validar_pronosticos(codigo_seleccionado_nv, 
                                              modelo_seleccionado_nv, 
                                              st.session_state.df_todos_pronosticos_fecha)
                
                st.write('Datos de pronostico para codigo y modelo seleccionado:')
                st.dataframe(df_filtrado_nv)

