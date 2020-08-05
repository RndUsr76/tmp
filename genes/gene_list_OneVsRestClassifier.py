def gene_list():
    genes=[]
    genes.append(['m_estimator',getClassifiers()])

    #genes.append([,[]])



    return genes

def getClassifiers():
    from sklearn.utils import all_estimators
    import sklearn
    import xgboost as xgb
    estimators = all_estimators()

    classifiers = []
    classifiers.append(sklearn.ensemble._gb.GradientBoostingClassifier())                 # OK
    classifiers.append(sklearn.neighbors._classification.KNeighborsClassifier())          # OK
    classifiers.append(sklearn.linear_model._logistic.LogisticRegressionCV())             # OK
    classifiers.append(sklearn.svm._classes.NuSVC())                                       # OK
    classifiers.append(sklearn.gaussian_process._gpc.GaussianProcessClassifier())         # OK
    classifiers.append(sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis())       # OK
    classifiers.append(sklearn.linear_model._logistic.LogisticRegression())                 # OK
    classifiers.append(xgb.XGBClassifier())
    classifiers.append(sklearn.naive_bayes.BernoulliNB())                               # OK
    classifiers.append(sklearn.svm._classes.SVC())                                         # OK Men kolla output! Verkar vara ensidig...
    classifiers.append(sklearn.tree._classes.DecisionTreeClassifier())                   # OK
    classifiers.append(sklearn.calibration.CalibratedClassifierCV(base_estimator=sklearn.ensemble._weight_boosting.AdaBoostClassifier()))          # OK
    classifiers.append(sklearn.linear_model._stochastic_gradient.SGDClassifier())          # OK
    classifiers.append(sklearn.naive_bayes.GaussianNB())                                  # OK
    classifiers.append(sklearn.neural_network._multilayer_perceptron.MLPClassifier())       # OK
    classifiers.append(sklearn.ensemble._forest.RandomForestClassifier())                  # OK
    classifiers.append(sklearn.tree._classes.ExtraTreeClassifier())                       # OK
    classifiers.append(sklearn.ensemble._forest.ExtraTreesClassifier())                   # OK
    classifiers.append(sklearn.discriminant_analysis.LinearDiscriminantAnalysis())          # OK
    classifiers.append(sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier())              # OK
    
    #c = []
    #for classifier in classifiers:
    #    c.append(type(classifier).__name__)

    return classifiers