import os.path
import numpy as np
from os import listdir
from os.path import join
from plotly import graph_objs as go
import time
import yfinance as yf
import pandas as pd



# Devuelve información importante sobre la acción cuyo ticker se pase como
# parametro
def get_info_accion(ticker):
    accion=yf.Ticker(ticker)
    #hist = accion.history(period="max")["Close"]
    print(accion.info)

# Crea un archivo .csv con la información diaria sobre el activo
# cuyo ticker se pase como parametro
def guardar_accion_csv(carpeta,ticker,start_date,end_date):
    accion = yf.Ticker(ticker)
    try:
        print("Guardando información de:",ticker)
        df = accion.history(start=start_date,end=end_date)
        time.sleep(2)
        if df.empty:
            acciones_sin_descargar.append(ticker)
        archivo = carpeta + ticker.replace(".","_")+".csv"
        print(archivo)
        df.to_csv(archivo)
    except Exception as ex:
        acciones_sin_descargar.append(ticker)
        print("No se ha podido descargar",ticker)


# Descarga los valores y crea csv para todos los tickers
# dentro de la carpeta pasada como parametro
def descargar_info_csv_carpeta(carpeta,path):
    archivos = [x for x in listdir(path) if os.path.isfile(join(path,x))]
    tickers = [os.path.splitext(x)[0] for x in archivos]
    numero_tickers = len(tickers)

    for i in range(numero_tickers):
        try:
            print("Descargando "+tickers[i]+"...")
            guardar_accion_csv(carpeta,tickers[i].replace("_","."))
        except Exception as ex:
            print("No se ha podido descargar:",tickers[i])

# Devuelve un DF de la accion correspondiente al ticker
def get_df_desde_csv(carpeta,ticker):
    return pd.read_csv(carpeta+ticker+".csv", index_col=0)

def set_retornos_diarios_al_df(df):
    df['retorno diario']= (df['Close']/df['Close'].shift(1)) - 1
    return df

def set_retornos_acumulados_al_df(df):
    df["retorno acumulado"]= (1 + df["retorno diario"]).cumprod()
    return df

def set_bandas_bollinger_al_df(df):
    df["banda_medio"] = df['Close'].rolling(window=20).mean()
    df["banda_superior"] = df['banda_medio'] + 1.96 * df['Close'].rolling(window=20).std()
    df["banda_inferior"] = df['banda_medio'] - 1.96 * df['Close'].rolling(window=20).std()
    return df

def set_ichimoku_al_df(df):
    #Linea de Conversión
    alto = df["High"].rolling(window=9).max()
    bajo = df["Low"].rolling(window=9).min()
    df["Conversion"] = (alto+bajo)/2
    #Linea Base
    alto = df["High"].rolling(window=26).max()
    bajo = df["Low"].rolling(window=26).min()
    df["Base"] = (alto+bajo)/2
    #Span A
    df["SpanA"]=((df["Conversion"] + df["Base"])/2)
    #Span B
    alto = df["High"].rolling(window=52).max()
    bajo = df["Low"].rolling(window=52).min()
    df["SpanB"]=((alto + bajo)/2).shift(26)
    #Lagging Span
    df["Lagging"] = df["Close"].shift(-26)

    return df

def set_info_df(df):
    set_retornos_diarios_al_df(df)
    set_retornos_acumulados_al_df(df)
    set_bandas_bollinger_al_df(df)
    set_ichimoku_al_df(df)

    return df


def aniadir_info_extra_stocks_carpeta(path):
    archivos = [x for x in listdir(path) if os.path.isfile(join(path,x))]
    tickers = [os.path.splitext(x)[0] for x in archivos]
    numero_tickers = len(tickers)

    for i in range(numero_tickers):
        try:
            print("Editando "+tickers[i]+"...")
            df = get_df_desde_csv(path,tickers[i])
            set_info_df(df)
            df.to_csv(path+tickers[i]+".csv")
        except Exception as ex:
            print("No se ha podido editar:",tickers[i])


# Actualiza un csv con los datos más nuevos, es decir, solo descarga los datos que faltan desde la ultima fecha guardada (CUIDADO CUANDO LOS MERCADOS ESTAN ABIERTOS)
def actualizar_csv(path,ticker):
    df_a_actualizar = pd.read_csv(path,index_col=0)
    df_a_actualizar.index = pd.to_datetime(df_a_actualizar.index)
    most_recent_date = df_a_actualizar.index.max()

    accion = yf.Ticker(ticker)
    df_nueva_info = accion.history(start=most_recent_date)

    while most_recent_date >= df_nueva_info.index.min():
        df_nueva_info.drop(axis=0,index=df_nueva_info.index[0],inplace=True)

    df_copia = df_a_actualizar.copy()
    df_copia = pd.concat([df_copia,df_nueva_info])

    ## Ahora tenemos que rellenar los datos que faltan (retornos acumulados etc)
    df_copia = set_info_df(df_copia)

    return df_copia


def actualizar_csv_carpeta(path):
    archivos = [x for x in listdir(path) if os.path.isfile(join(path,x))]
    tickers = [os.path.splitext(x)[0] for x in archivos]
    numero_tickers = len(tickers)

    for i in range(numero_tickers):
        try:
            print("Actualizando "+tickers[i]+"...")
            df_actualizar = actualizar_csv(path+tickers[i]+'.csv',tickers[i])
            time.sleep(0.1)
            df_actualizar.to_csv(path+tickers[i]+".csv")
        except Exception as ex:
            print("No se ha podido editar:",tickers[i])


def plot_candlesticks(df):
    fig = go.Figure()
    candle = go.Candlestick(x=df.index, open = df['Open'],high=df['High'],low=df['Low'],close=df["Close"],name="Candlestick")
    fig.add_trace(candle)
    fig.update_xaxes(title="Fecha",rangeslider_visible=True)
    fig.update_xaxes(title="Precio")
    fig.show()

def plot_bandas_bollinger(df,ticker):
    fig = go.Figure()

    candle = go.Candlestick(x=df.index, open=df['Open'],high=df['High'], low=df['Low'],close=df['Close'], name="Candlestick")
    upper_line = go.Scatter(x=df.index, y=df['banda_superior'],line=dict(color='rgba(250, 0, 0, 0.75)',width=1), name="Upper Band")
    mid_line = go.Scatter(x=df.index, y=df['banda_medio'],line=dict(color='rgba(0, 0, 250, 0.75)',width=0.7), name="Middle Band")
    lower_line = go.Scatter(x=df.index, y=df['banda_inferior'],line=dict(color='rgba(0, 250, 0, 0.75)',width=1), name="Lower Band")

    fig.add_trace(candle)
    fig.add_trace(upper_line)
    fig.add_trace(mid_line)
    fig.add_trace(lower_line)

    fig.update_xaxes(title="Fecha", rangeslider_visible=True)
    fig.update_yaxes(title="Precio")

    fig.update_layout(title=ticker + " Bandas Bollinger",height=1200, width=1800, showlegend=True)
    fig.show()

def color_accion(marca):
    if marca >=1:
        return 'rgba(0,250,0,0.3)'
    else:
        return 'rgba(250,0,0,0.3)'

def plot_ichimoku(df,ticker):
    fig = go.Figure()

    df_copia=df.copy()

    df['marca'] = np.where(df['SpanA']>df['SpanB'],1,0)
    df['group'] = df['marca'].ne(df['marca'].shift()).cumsum()
    df = df.groupby('group')

    dfs =[]
    for name,data in df:
        dfs.append(data)
    for df in dfs:
        fig.add_traces(go.Scatter(x=df.index,y=df['SpanA'],line=dict(color='rgba(0,0,0,0)')))
        fig.add_traces(go.Scatter(x=df.index,y=df['SpanB'],line=dict(color='rgba(0,0,0,0)'),
                                 fill='tonexty',fillcolor=color_accion(df['marca'].iloc[0])))

    candle = go.Candlestick(x=df_copia.index, open=df_copia['Open'],high=df_copia['High'], low=df_copia['Low'],close=df_copia['Close'], name="Candlestick")
    base = go.Scatter(x=df_copia.index, y=df_copia['Base'],
                      line=dict(color='pink',width=2), name="Base")
    conversion = go.Scatter(x=df_copia.index, y=df_copia['Conversion'],
                            line=dict(color='black',width=1), name="Conversión")
    span_a = go.Scatter(x=df_copia.index, y=df_copia['SpanA'],
                        line=dict(color='green',width=1), name="SpanA")
    span_b = go.Scatter(x=df_copia.index, y=df_copia['SpanB'],
                        line=dict(color='red',width=1), name="SpanB")
    lagging = go.Scatter(x=df_copia.index, y=df_copia['Lagging'],
                         line=dict(color='purple',width=1), name="Lagging")

    fig.add_trace(candle)
    fig.add_trace(conversion)
    fig.add_trace(base)
    fig.add_trace(span_a)
    fig.add_trace(span_b)
    fig.add_trace(lagging)

    fig.update_xaxes(title="Fecha", rangeslider_visible=True)
    fig.update_yaxes(title="Precio")

    fig.update_layout(title=ticker + " Ichimoku",height=1000, width=1500, showlegend=True)
    fig.show()

# Actualiza un csv con los datos más nuevos, es decir, solo descarga los datos que faltan desde la ultima fecha guardada
def update_csv(path,ticker):
    df_a_actualizar = pd.read_csv(path,index_col=0)
    df_a_actualizar.index = pd.to_datetime(df_a_actualizar.index)
    most_recent_date = df_a_actualizar.index.max()

    accion = yf.Ticker(ticker)
    df_nueva_info = accion.history(start=most_recent_date)

    while most_recent_date >= df_nueva_info.index.min():
        df_nueva_info.drop(axis=0,index=df_nueva_info.index[0],inplace=True)

    df_copia = df_a_actualizar.copy()
    df_copia = pd.concat([df_copia,df_nueva_info])

    ## Ahora tenemos que rellenar los datos que faltan (retornos acumulados etc)
    df_copia = set_info_df(df_copia)

    return df_copia

#Funcion para eliminar variables correlacionadas con más de un umbral,
# se eliminara aquella de las dos que tenga menos correlación con la variables
# objetivo, que suponemos esta en la primera columna del DF
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_index = set()
    corr_matrix = dataset.corr()
    # Empezamos en 1 para saltarnos la variable target
    for i in range(1,len(corr_matrix.columns)):
        if(i not in corr_index):
            for j in range(1,i):
                if(j not in corr_index):
                    if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                        if corr_matrix.iloc[i, 0]>corr_matrix.iloc[j, 0]:
                            colname = corr_matrix.columns[j]  # getting the name of column
                            col_corr.add(colname)
                            corr_index.add(j)
                        else:
                            colname = corr_matrix.columns[i]  # getting the name of column
                            col_corr.add(colname)
                            corr_index.add(i)

    return list(col_corr),corr_index