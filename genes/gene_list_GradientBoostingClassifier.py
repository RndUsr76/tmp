def gene_list():
    genes=[]
    genes.append(['s_loss',['deviance','exponential']])
    genes.append(['i_n_estimators',[10,50,100, 200]])
    genes.append(['f_learning_rate',[0.001, 0.01, 0.1, 0.5, 1]])
    genes.append(['f_subsample',[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    genes.append(['s_criterion',['friedman_mse', 'mse', 'mae']])
    genes.append(['f_min_samples_split',[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    genes.append(['f_min_samples_leaf',[0.1, 0.2, 0.3, 0.4, 0.5]])
    genes.append(['f_min_weight_fraction_leaf',[0, 0.1, 0.2, 0.3, 0.4, 0.5]])
    genes.append(['i_max_depth',[1,2,3,5,7,9]])

    return genes