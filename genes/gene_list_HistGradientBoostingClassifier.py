def gene_list():
    genes=[]
    genes.append(['f_learning_rate',[0.001, 0.01, 0.1, 0.5, 1]])
    genes.append(['i_max_iter',[50, 100, 200]])
    genes.append(['i_max_leaf_nodes',[10, 20, 31]])
    genes.append(['i_max_depth',[1,2,3,5,7,9]])
    genes.append(['f_l2_regularization',[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])

    return genes