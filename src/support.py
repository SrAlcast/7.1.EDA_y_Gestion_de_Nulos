# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Otros objetivos
# -----------------------------------------------------------------------
import math
import numpy as np

# Gráficos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

# Imputación de nulos usando métodos avanzados estadísticos
# -----------------------------------------------------------------------
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor

def separar_dataframe(dataframe):
    return dataframe.select_dtypes(include=np.number),dataframe.select_dtypes(include="O")

def plot_numericas(dataframe):
    df_num=dataframe.select_dtypes(include=np.number)
    num_filas=math.ceil(len(df_num.columns)/2)
    fig, axes=plt.subplots(nrows=num_filas, ncols=2,figsize=(15,10))
    axes=axes.flat

    for indice, columna in enumerate(df_num.columns):
        sns.histplot(x=columna, data=df_num, ax=axes[indice])
        axes[indice].set_title(columna)
        axes[indice].set_xlabel("")

    if len(df_num.columns)%2!=0:
        fig.delaxes(axes[-1])  
    else:
        pass

    plt.tight_layout()

def plot_categoricas(dataframe,paleta="mako"):
    df_cat=dataframe.select_dtypes(include="O")
    num_filas=math.ceil(len(df_cat.columns)/2)
    fig, axes=plt.subplots(nrows=num_filas, ncols=2,figsize=(15,10))
    axes=axes.flat

    for indice, columna in enumerate(df_cat.columns):
        sns.countplot(x=columna, data=df_cat, ax=axes[indice],palette=paleta,order=df_cat[columna].value_counts().index)
        axes[indice].set_title(columna)
        axes[indice].set_xlabel("")  
    if len(df_cat.columns)%2!=0:
        fig.delaxes(axes[-1])  
    else:
        pass
    
    plt.tight_layout()

def relacion_vr_categoricas(dataframe, variable_respuesta, paleta="mako", tamaño_grafica=(15,10)):
    df_cat=dataframe.select_dtypes(include="O")
    num_filas=math.ceil(len(df_cat.columns)/2)
    fig, axes=plt.subplots(nrows=num_filas, ncols=2,figsize=tamaño_grafica)
    axes=axes.flat

    for indice, columna in enumerate(df_cat.columns):
        datos_agrupados=dataframe.groupby(columna)[variable_respuesta].mean().reset_index().sort_values(variable_respuesta,ascending=False)
        sns.barplot(x=columna,
                    y=variable_respuesta,
                    data=datos_agrupados,
                    ax=axes[indice],
                    palette=paleta)
        axes[indice].set_title(columna)
        axes[indice].set_xlabel("")  
        axes[indice].tick_params(axis='x', rotation=90)
    
    if len(dataframe.columns)%2!=0:
        fig.delaxes(axes[-1])  
    else:
        pass
    
    plt.tight_layout()

def relacion_vr_numericas(dataframe, variable_respuesta, paleta="mako", tamaño_grafica=(15,10)):
    df_num=dataframe.select_dtypes(include=np.number)
    num_filas=math.ceil(len(df_num.columns)/2)
    fig, axes=plt.subplots(nrows=num_filas, ncols=2,figsize=tamaño_grafica)
    axes=axes.flat

    for indice, columna in enumerate(df_num.columns):
        if columna==variable_respuesta:
            fig.delaxes(axes[indice])
            pass
        else:
            sns.scatterplot(x=columna,
                        y=variable_respuesta,
                        data=df_num,
                        ax=axes[indice],
                        palette=paleta)
            axes[indice].set_title(columna)
            axes[indice].set_xlabel("")  
            axes[indice].tick_params(axis='x', rotation=90)
    
    if len(dataframe.columns)%2!=0:
        fig.delaxes(axes[-1])  
    else:
        pass
    
    plt.tight_layout()

def matriz_correlacion(dataframe):
    matriz_corr=dataframe.corr(numeric_only=True)
    mascara=np.triu(np.ones_like(matriz_corr,dtype=np.bool_))
    sns.heatmap(matriz_corr,
                annot=True,
                vmin=1,
                vmax=-1,
                mask=mascara,
                cmap="hsv")
    plt.figure(figsize=(10,15))
    plt.tight_layout()

def detectar_metricas(dataframe, color="orange", tamaño_grafica=(15,10)):
    df_num = dataframe.select_dtypes(include=np.number)
    num_filas = math.ceil(len(df_num.columns) / 2)
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=tamaño_grafica)
    axes = axes.flat

    # Configuración de los outliers en color naranja
    flierprops = dict(marker='o', markerfacecolor='orange', markersize=5, linestyle='none')

    for indice, columna in enumerate(df_num.columns):
        sns.boxplot(x=columna,
                    data=df_num,
                    ax=axes[indice],
                    color=color,
                    flierprops=flierprops)  # Aplica color naranja a los outliers
        axes[indice].set_title(f"Outliers de {columna}")
        axes[indice].set_xlabel("")

    # Eliminar el último subplot si el número de columnas es impar
    if len(df_num.columns) % 2 != 0:
        fig.delaxes(axes[-1])
    
    plt.tight_layout()

def exploracion_basica_dataframe(dataframe):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ------------------------------- \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ------------------------------- \n")
    
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
    
    print("\n ------------------------------- \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    
    
    print("\n ------------------------------- \n")
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = pd.DataFrame(dataframe.select_dtypes(include = "O"))
    print(dataframe_categoricas.columns)
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene las siguientes valore únicos:")
        display(pd.DataFrame(dataframe[col].value_counts()))
        if dataframe[col].isnull().sum()>0:
            print(f"La columna {col.upper()} tiene {dataframe[col].isnull().sum()} valores nulos")
        print("\n ------------------------------- \n")

    print("Los valores que tenemos para las columnas numericas son: ")
    dataframe_numericas = pd.DataFrame(dataframe.select_dtypes(include =np.number))
    
    for col in dataframe_numericas.columns:
        print(f"La columna {col.upper()} tiene las siguientes valore únicos:")
        display(pd.DataFrame(dataframe[col].value_counts())) 
        if dataframe[col].isnull().sum()>0:
            print(f"La columna {col.upper()} tiene {dataframe[col].isnull().sum()} valores nulos")   