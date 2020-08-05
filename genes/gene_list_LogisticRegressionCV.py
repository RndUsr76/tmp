def gene_list():
    genes=[]
    genes.append(['s_penalty',['l1', 'l2', 'elasticnet']])
    genes.append(['s_solver',['newton-cg', 'lbfgs', 'sag', 'saga']])
    genes.append(['i_max_iter',[50, 100, 150]])



    return genes