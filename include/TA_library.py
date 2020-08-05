import pandas as pd

def MA(df_col, n):
    return df_col.rolling(n).mean()

def EMA(df_col, n=10):
    return df_col.ewm(span=n).mean()

def dMA(df_col, n=1):
    return df_col - df_col.shift(n)

def multiply(df, colA, colB):
    colname = colA + '_times_' + colB
    df[colname] = df[colA]*df[colB]
    return df

def MA_dMA(df, col_name,n=5):
    df['MA_' + col_name] = MA(df[col_name], n)
    df['dMA_' + col_name] = dMA(df[col_name])
    return df

def EMA_dEMA(df, col_name,n=5):
    df['EMA_' + col_name] = EMA(df[col_name], n)
    df['dEMA_' + col_name] = dMA(df[col_name])
    return df

def RSI(df_col, n=14):
    delta = df_col.diff()
    
    dUP, dDN = delta.copy(), delta.copy()
    dUP[dUP < 0] = 0
    dDN[dDN > 0] = 0
    
    avgUPs = dUP.rolling(n).mean()
    avgDNs = dDN.rolling(n).mean().abs()
    
    RS = avgUPs / avgDNs
    RSI = 100.0 - (100.0 / (1.0 + RS))
    
    #RSI[0:n] = RSI[n]
    
    return RSI

def LL(df_col, n):
    return df_col.rolling(n).min()

def HH(df_col, n):
    return df_col.rolling(n).max()

def STOCH(df_col_high, df_col_low, df_col_close, n=14, fastk_period=3, fastd_period=3):
    LL_n = LL(df_col_low, n)
    HH_n = HH(df_col_high, n)
    
    K = ((df_col_close - LL_n)/(HH_n - LL_n))*100
    K = K.rolling(fastk_period).mean()
    D = K.rolling(fastd_period).mean()
    return K, D

def BBANDS(df_col, n, num_std=2):
    _mean = df_col.rolling(n).mean()
    _std = df_col.rolling(n).std()
    
    upper = _mean + num_std*_std
    lower = _mean - num_std*_std
    
    return upper, _mean, lower

def ATR(high, low, close, n=14):
    
    H_L = high - low
    H_Cp = abs(high - close.shift(1))
    L_Cp = abs(low - close.shift(1))
    
    _df = pd.DataFrame()
    _df[0] = H_L
    _df[1] = H_Cp
    _df[2] = L_Cp
    
    TR = _df.max(axis=1)
    
    ATR = TR.rolling(n).mean()
    
    return ATR

def CHAIKIN(high, low, close, volume, n1=3, n2=10):
    n1 = 3
    n2 = 10
    MFM = ((close-low) - (high-close))/(high-low)
    MFV = MFM * volume
    ADL = MFV.shift(1) + MFV
    ChaikinO = ADL.ewm(span=n1).mean() - ADL.ewm(span=n2).mean()
    return ChaikinO

def MACD(close, n_slow=26, n_fast=12):
    ema_slow = close.ewm(span=n_slow).mean()
    ema_fast = close.ewm(span=n_fast).mean()
    macd = ema_fast-ema_slow
    return macd, ema_slow, ema_fast

def CCI(high, low, close, n=10):
    TP = (high + low + close)/3
    CCI=(TP - TP.rolling(n).mean()) / (0.015 * TP.rolling(n).std())
    return CCI