def getParams(gene_dict, classifier_name):
    para = []
    p = [ k for k,v in gene_dict.items() if k.startswith(classifier_name)]
    for i in range(len(p)):
        key = p[i]
        key = key.replace(classifier_name + '_','')
        value = gene_dict[p[i]]
        #print(f'Key: {key} value: {value}')
        para.append([key,value])
    
    param_dict = {}
    for p in para:
        param_dict[p[0]] = p[1]
    para = param_dict
    
    return param_dict