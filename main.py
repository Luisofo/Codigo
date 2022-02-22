#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import time
import yfinance as yf
#%%
acciones_sin_descargar = []
#%%
def get_info_accion(ticker):
    accion=yf.Ticker(ticker)
    #hist = accion.history(period="max")["Close"]
    print(accion.info)

def guardar_accion_csv(carpeta,ticker):
    accion = yf.Ticker(ticker)
    try:
        print("Guardando información de:",ticker)
        df = accion.history(period="max")["Close"]
        time.sleep(2)
        if df.empty:
            acciones_sin_descargar.append(ticker)
        archivo = carpeta + ticker.replace(".","_")+".csv"
        print(archivo)
        df.to_csv(archivo)
    except Exception as ex:
        acciones_sin_descargar.append(ticker)
        print("No se ha podido descargar",ticker)
#%%
get_info_accion("ANA.MC")
#%%
tickers = pd.read_csv("IBEX35.csv")
tickers = tickers["TICKER"]
#%%
carpeta = "./datos/"
for x in range(20):
    guardar_accion_csv(carpeta,tickers[x])