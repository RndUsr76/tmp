def gene_list():
    genes=[]
    genes.append(['i_rsi_period',[3, 5,10,15, 20]])
    genes.append(['i_stoch_n_period',[7, 10, 12]])
    genes.append(['i_stoch_fastk_period',[2,3,4]])
    genes.append(['i_stoch_fastd_period',[3,5,7, 9]])
    genes.append(['i_atr_period',[10, 15,20, 25, 30]])
    genes.append(['i_hh_period',[45,60,90,120]])
    genes.append(['i_ll_period',[15,30,45,60,90]])
    genes.append(['i_volatility_period',[60,90,120, 150,180]])
    genes.append(['i_bbands_n',[3,5,10,15,20]])
    genes.append(['i_chaikin_n1',[1,2,3]])
    genes.append(['i_chaikin_n2',[2,3,4,5, 10, 15, 20]])
    genes.append(['i_macd_slow',[30,40,50,60]])
    genes.append(['i_macd_fast',[2,4,12,15,18]])
    genes.append(['i_cci_n',[5,8,10,12,15]])

    genes.append(['i_return_ma_period',[20,30,45,60,90]])
    genes.append(['i_range_ma_period',[45,60,90,120,150,180]])
    genes.append(['i_volume_ma_period',[10,15,30,45,60,90]])
    genes.append(['i_volatility_ma_period',[30,45,60,90]])
    genes.append(['i_high_to_close_ma_period',[90,120,150, 180]])
    genes.append(['i_low_to_close_ma_period',[5,10,15,30,45,60,90,120]])
    genes.append(['i_hh_ma_period',[5,10,15,90,120, 120]])
    genes.append(['i_close_to_hh_ma_period',[5,10,15,30,45,60,90]])
    genes.append(['i_ll_ma_period',[5,10,15,30,45,60,90]])
    genes.append(['i_close_to_ll_ma_period',[45,60,90,120,150]])
    genes.append(['i_atr_ma_period',[30,45,60]])
    genes.append(['i_rsi_ma_period',[3,5,10,15]])

    genes.append(['b_use_model_without_parameter',[True, False]])

    return genes