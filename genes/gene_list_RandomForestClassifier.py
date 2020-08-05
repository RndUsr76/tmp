def gene_list():
    genes=[]
    genes.append(['i_n_estimators',[10,50,100, 200]])
    genes.append(['s_criterion',['gini','entropy']])
    genes.append(['i_max_depth',[1,2,3,5,7,9]])
    genes.append(['f_min_samples_split',[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    genes.append(['f_min_samples_leaf',[0.1, 0.2, 0.3, 0.4, 0.5]])
    genes.append(['f_min_weight_fraction_leaf',[0, 0.1, 0.2, 0.3, 0.4, 0.5]])
    genes.append(['f_max_features',[0.3, 0.5, 0.7, 0.8, 0.9, 1.0]])
    genes.append(['b_oob_score',[True, False]])




    return genes