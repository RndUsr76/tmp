def gene_list():
    genes=[]
    genes.append(['i_nthread',[1,3,6,8]])
    genes.append(['f_eta',[0.01, 0.1, 0.3, 0.5, 0.7]])
    genes.append(['f_gamma',[0, 0.2, 0.4, 0.6, 0.8, 0.99, 10, 100]])
    genes.append(['i_max_depth',[1,3,5,10,20,50]])
    genes.append(['i_min_child_weight',[1,3,5,10,20,50]])
    genes.append(['i_max_delta_step',[1,3,5,7,9]])
    genes.append(['f_subsample',[0.1, 0.2, 0.4, 0.6, 0.8, 0.99]])
    genes.append(['f_colsample_bytree',[0.1, 0.2, 0.4, 0.6, 0.8, 1]])
    genes.append(['f_colsample_bylevel',[0.1, 0.2, 0.4, 0.6, 0.8, 1]])
    genes.append(['f_colsample_bynode',[0.1, 0.2, 0.4, 0.6, 0.8, 1]])
    genes.append(['f_learning_rate',[0.0001, 0.001, 0.01, 0.1, 0.2]])
    genes.append(['i_n_estimators',[125, 250, 500,1000,1500]])
    
    
    
    

    
    
    

    #genes.append([,[]])



    return genes