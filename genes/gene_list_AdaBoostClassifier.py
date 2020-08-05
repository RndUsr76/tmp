def gene_list():
    genes=[]
    genes.append(['i_n_estimators',[10,50,100, 200]])
    genes.append(['f_learning_rate',[0.001, 0.01, 0.1, 0.5, 1]])
    genes.append(['m_base_estimator',getClassifiers()])


    #genes.append([,[]])



    return genes


def getClassifiers():
    from sklearn.utils import all_estimators
    import sklearn
    import xgboost as xgb
    estimators = all_estimators()

    classifiers = []
    
    
    
    classifiers.append(sklearn.ensemble._bagging.BaggingClassifier())                   # OK
    classifiers.append(sklearn.tree._classes.DecisionTreeClassifier())                  # OK
    classifiers.append(sklearn.ensemble._forest.ExtraTreesClassifier())                 # OK 
    classifiers.append(sklearn.naive_bayes.BernoulliNB())                               # OK
    classifiers.append(sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier())              # OK
    classifiers.append(xgb.XGBClassifier())
    classifiers.append(sklearn.linear_model._logistic.LogisticRegressionCV())               # OK
    classifiers.append(sklearn.tree._classes.ExtraTreeClassifier())                       # OK
    classifiers.append(sklearn.ensemble._forest.RandomForestClassifier())                  # OK
    classifiers.append(sklearn.linear_model._logistic.LogisticRegression())                 # OK
    
    
    #classifiers.append(sklearn.ensemble._gb.GradientBoostingClassifier())                 # OK tar lång tid

    
    c = []
    for classifier in classifiers:
        c.append(type(classifier).__name__)



    return c
