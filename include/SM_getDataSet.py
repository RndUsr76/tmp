import TA_library as TA

import numpy as np
import pandas as pd
import copy

def getDataSet(df, genes, params):

    periods_away_to_predict = params['periods_away_to_predict']
    get_target = params['get_target']
    
    #### TARGET
    if get_target == True:
        target = df['close'].shift(-periods_away_to_predict) > df['close']
        #df['target'] = [1 if t==True else 0 for t in target]

    cols_to_drop = []
    cols_to_drop.append('open')
    cols_to_drop.append('high')
    cols_to_drop.append('low')
    cols_to_drop.append('close')

    df['return'] = np.log(df['close']/df['close'].shift(1))

    df['range'] = (df['high']-df['low'])/(df['low']+(df['high']-df['low'])/2)
    df['high_to_close'] = df['close']/df['high']
    df['low_to_close'] = df['close']/df['low']

    df['rsi'] = TA.RSI(df['close'], genes['i_rsi_period'])
    df['stoch_k'], df['stoch_d'] = TA.STOCH(df['high'], df['low'], df['close'], genes['i_stoch_n_period'], genes['i_stoch_fastk_period'], genes['i_stoch_fastd_period'])
    df['d_stoch_d'] = TA.dMA(df['stoch_d'])
    df['stock_rel'] = df['stoch_k'] / df['stoch_d']
    df['atr'] = TA.ATR(df['high'], df['low'], df['close'], genes['i_atr_period'])
    
    upper, _mean, lower = TA.BBANDS(df['close'], genes['i_bbands_n'])
    df['bbands_upper_rel'] = _mean/upper
    df['bbands_lower_rel'] = _mean/lower
    
    df['chaikin'] = TA.CHAIKIN(df['high'], df['low'], df['close'], genes['i_chaikin_n1'], genes['i_chaikin_n2'])
    
    df['cci'] = TA.CCI(df['high'], df['low'], df['close'], genes['i_cci_n'])
    
    df['macd'], df['macd_ema_slow'], df['macd_ema_fast'] = TA.MACD(df['close'], genes['i_macd_slow'], genes['i_macd_fast'])
    df['macd_cross'] = df['macd_ema_fast'] / df['macd_ema_slow']
        
    df['hh'] = TA.HH(df['high'], genes['i_hh_period'])
    df['close_to_hh'] = df['close'] / df['hh']

    df['ll'] = TA.HH(df['low'], genes['i_ll_period'])
    df['close_to_ll'] = df['close'] / df['ll']

    cols_to_drop.append('hh')
    cols_to_drop.append('ll')

    df['high_to_close'] = df['close']/df['high']
    df['low_to_close'] = df['close']/df['low']

    df['volatility'] = df['return'].rolling(genes['i_volatility_period']).std(ddof=0)

    ma_dma = []
    ma_dma.append(['return', genes['i_return_ma_period']])
    ma_dma.append(['range', genes['i_range_ma_period']])
    ma_dma.append(['volume', genes['i_volume_ma_period']])
    ma_dma.append(['volatility', genes['i_volatility_ma_period']])
    ma_dma.append(['high_to_close', genes['i_high_to_close_ma_period']])
    ma_dma.append(['low_to_close', genes['i_low_to_close_ma_period']])
    ma_dma.append(['hh', genes['i_hh_ma_period']])
    ma_dma.append(['close_to_hh', genes['i_close_to_hh_ma_period']])
    ma_dma.append(['ll', genes['i_ll_ma_period']])
    ma_dma.append(['close_to_ll', genes['i_close_to_ll_ma_period']])
    ma_dma.append(['atr', genes['i_atr_ma_period']])
    ma_dma.append(['rsi', genes['i_rsi_ma_period']])
    ma_dma.append(['macd', 5])
    ma_dma.append(['macd_cross', 5]) 

    for feat, period in ma_dma:
        df = TA.MA_dMA(df, feat, period)
        df = TA.EMA_dEMA(df, feat, period)
        df['MA_EMA_ratio_' + feat] = df['EMA_' + feat] / (df['MA_' + feat] + 1e-7)
        df['dMA_dEMA_ratio_' + feat] = df['dEMA_' + feat] / (df['dMA_' + feat] + 1e-7)

    df = TA.multiply(df,'return','volume')
    df = TA.multiply(df,'dMA_return','dMA_volume')
    
    
    cols_to_drop.append('datetime')

    df.drop(cols_to_drop, axis=1, inplace=True)
   
    if get_target:
        df['target'] = target
    
    df.replace(-np.inf, np.nan, inplace=True)
    df.replace(np.inf, np.nan, inplace=True)

    df.dropna(axis=0, inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df
