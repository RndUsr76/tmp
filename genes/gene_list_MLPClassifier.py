def gene_list():
    genes=[]
    genes.append(['i_hidden_layer_sizes',[64, 128, 256]])
    genes.append(['s_activation',['logistic', 'tanh', 'relu']])
    genes.append(['s_solver',['lbfgs', 'sgd', 'adam']])
    genes.append(['f_alpha',[0.000001, 0.00001, 0.0001]])
    genes.append(['i_max_iter',[50, 100, 150]])

    return genes