def gene_list():
    genes=[]
    genes.append(['i_n_neighbors',[3,5,7,9]])
    genes.append(['s_weights',['uniform', 'distance']])
    genes.append(['s_algorithm',['auto', 'ball_tree', 'kd_tree', 'brute']])
    genes.append(['i_leaf_size',[10, 20, 30, 40, 50]])
    genes.append(['i_p',[1,2]])



    return genes