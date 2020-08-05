def gene_list():
    genes=[]
    genes.append(['s_loss',['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']])
    genes.append(['s_penalty',['l1', 'l2', 'elasticnet']])
    genes.append(['f_alpha',[0.00001, 0.0001, 0.001]])
    
    genes.append(['i_max_iter',[500, 1000, 1500]])
    genes.append(['s_learning_rate',['constant','optimal','invscaling','adaptive']])
    genes.append(['b_early_stopping',[True, False]])

    #genes.append([,[]])



    return genes