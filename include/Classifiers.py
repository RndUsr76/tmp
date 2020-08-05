
import xgboost as xgb
import sklearn
from sklearn.utils import all_estimators
estimators = all_estimators()

def getClassifiers():
    from sklearn.utils import all_estimators
    estimators = all_estimators()

    classifiers = []
    classifiers.append(sklearn.ensemble._weight_boosting.AdaBoostClassifier())          # OK
    classifiers.append(sklearn.ensemble._bagging.BaggingClassifier())                   # OK
    #classifiers.append(sklearn.mixture._bayesian_mixture.BayesianGaussianMixture())     # Kolla denna, ger 0 som svar
    classifiers.append(sklearn.naive_bayes.BernoulliNB())                               # OK
    classifiers.append(sklearn.calibration.CalibratedClassifierCV(base_estimator=sklearn.ensemble._weight_boosting.AdaBoostClassifier()))          # OK
    classifiers.append(sklearn.tree._classes.DecisionTreeClassifier())                   # OK
    classifiers.append(sklearn.tree._classes.ExtraTreeClassifier())                       # OK
    classifiers.append(sklearn.ensemble._forest.ExtraTreesClassifier())                   # OK
    #classifiers.append(sklearn.mixture._gaussian_mixture.GaussianMixture())               # OK
    #classifiers.append(sklearn.naive_bayes.GaussianNB())                                  # OK
    #classifiers.append(sklearn.gaussian_process._gpc.GaussianProcessClassifier())         # OK
    classifiers.append(sklearn.ensemble._gb.GradientBoostingClassifier())                 # OK
    classifiers.append(sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier())              # OK
    classifiers.append(sklearn.neighbors._classification.KNeighborsClassifier())                                    # OK
    classifiers.append(sklearn.discriminant_analysis.LinearDiscriminantAnalysis())          # OK
    classifiers.append(sklearn.linear_model._logistic.LogisticRegression())                 # OK
    classifiers.append(sklearn.linear_model._logistic.LogisticRegressionCV())               # OK
    classifiers.append(sklearn.neural_network._multilayer_perceptron.MLPClassifier())       # OK
    #classifiers.append(sklearn.svm._classes.NuSVC())                                       # OK
    classifiers.append(sklearn.multiclass.OneVsRestClassifier(sklearn.ensemble._weight_boosting.AdaBoostClassifier()))    # OK
    classifiers.append(sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis())       # OK
    classifiers.append(sklearn.ensemble._forest.RandomForestClassifier())                  # OK
    classifiers.append(sklearn.linear_model._stochastic_gradient.SGDClassifier())          # OK
    #classifiers.append(sklearn.svm._classes.SVC())                                         # OK Men kolla output! Verkar vara ensidig...
    classifiers.append(xgb.XGBClassifier())

    return classifiers


def getClassifier(classifier_name, genes, standard=False):
    if classifier_name == 'AdaBoostClassifier':
        if standard == True: return sklearn.ensemble._weight_boosting.AdaBoostClassifier()

        return sklearn.ensemble._weight_boosting.AdaBoostClassifier(
            base_estimator = getClassifier(genes['m_base_estimator'],genes, standard=True), 
            n_estimators = genes['i_n_estimators'], 
            learning_rate = genes['f_learning_rate'], 
            algorithm='SAMME.R')
        #return sklearn.ensemble._weight_boosting.AdaBoostClassifier()
    elif classifier_name == 'BaggingClassifier':
        if standard == True: return sklearn.ensemble._bagging.BaggingClassifier()

        return sklearn.ensemble._bagging.BaggingClassifier(
            base_estimator = getClassifier(genes['m_base_estimator'],genes, standard=True), 
            n_estimators = genes['i_n_estimators'])
    elif classifier_name == 'BernoulliNB':
        if standard == True: return sklearn.naive_bayes.BernoulliNB()

        return sklearn.naive_bayes.BernoulliNB(
            alpha = genes['f_alpha'], 
            binarize = genes['f_binarize'])
    elif classifier_name == 'CalibratedClassifierCV':
        if standard == True: return sklearn.calibration.CalibratedClassifierCV()

        return sklearn.calibration.CalibratedClassifierCV(
            base_estimator = getClassifier(genes['m_base_estimator'],genes, standard=True)
            )
    elif classifier_name == 'DecisionTreeClassifier':
        if standard == True: return sklearn.tree._classes.DecisionTreeClassifier()

        return sklearn.tree._classes.DecisionTreeClassifier(
            criterion = genes['s_criterion'],
            splitter = genes['s_splitter'],
            max_depth = genes['i_max_depth'],
            min_samples_split = genes['f_min_samples_split'],
            min_samples_leaf = genes['f_min_samples_leaf'],
            min_weight_fraction_leaf = genes['f_min_weight_fraction_leaf'],
            max_features = genes['f_max_features']
        )       
    elif classifier_name == 'ExtraTreeClassifier':
        if standard == True: return sklearn.ensemble._forest.ExtraTreeClassifier()

        return sklearn.tree._classes.ExtraTreeClassifier(
            criterion = genes['s_criterion'],
            splitter = genes['s_splitter'],
            max_depth = genes['i_max_depth'],
            min_samples_split = genes['f_min_samples_split'],
            min_samples_leaf = genes['f_min_samples_leaf'],
            min_weight_fraction_leaf = genes['f_min_weight_fraction_leaf'],
            max_features = genes['f_max_features']
        )
    elif classifier_name == 'ExtraTreesClassifier':
        if standard == True: return sklearn.ensemble._forest.ExtraTreesClassifier()

        return sklearn.ensemble._forest.ExtraTreesClassifier(
            n_estimators = genes['i_n_estimators'], 
            criterion = genes['s_criterion'],
            max_depth = genes['i_max_depth'],
            min_samples_split = genes['f_min_samples_split'],
            min_samples_leaf = genes['f_min_samples_leaf'],
            min_weight_fraction_leaf = genes['f_min_weight_fraction_leaf'],
            max_features = genes['f_max_features'],
            oob_score = genes['b_oob_score'],
            bootstrap=True
        )
    elif classifier_name == 'GradientBoostingClassifier':
        if standard == True: return sklearn.ensemble._gb.GradientBoostingClassifier()
        
        return sklearn.ensemble._gb.GradientBoostingClassifier(
            loss = genes['s_loss'],
            n_estimators = genes['i_n_estimators'], 
            learning_rate = genes['f_learning_rate'], 
            subsample = genes['f_subsample'],
            criterion = genes['s_criterion'],
            max_depth = genes['i_max_depth'],
            min_samples_split = genes['f_min_samples_split'],
            min_samples_leaf = genes['f_min_samples_leaf'],
            min_weight_fraction_leaf = genes['f_min_weight_fraction_leaf']
        )
        #return sklearn.ensemble._gb.GradientBoostingClassifier()
    elif classifier_name == 'HistGradientBoostingClassifier':
        if standard == True: return sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier()

        return sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier(
            learning_rate = genes['f_learning_rate'],
            max_iter = genes['i_max_iter'],
            max_leaf_nodes = genes['i_max_leaf_nodes'],
            max_depth = genes['i_max_depth'],
            l2_regularization = genes['f_l2_regularization']
        )
        #return sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier()
    elif classifier_name == 'KNeighborsClassifier':
        if standard == True: return sklearn.neighbors._classification.KNeighborsClassifier()

        return sklearn.neighbors._classification.KNeighborsClassifier(
            n_neighbors = genes['i_n_neighbors'],
            weights = genes['s_weights'],
            algorithm = genes['s_algorithm'],
            leaf_size = genes['i_leaf_size'],
            p = genes['i_p']
        )
        #return sklearn.neighbors._classification.KNeighborsClassifier()
    elif classifier_name == 'LinearDiscriminantAnalysis':
        if standard == True: return sklearn.discriminant_analysis.LinearDiscriminantAnalysis()

        return sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
            solver = genes['s_solver']
        )
        #return sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    elif classifier_name == 'LogisticRegression':
        if standard == True: return sklearn.linear_model._logistic.LogisticRegression()

        return sklearn.linear_model._logistic.LogisticRegression(
            penalty = genes['s_penalty'],
            solver = genes['s_solver'],
            max_iter = genes['i_max_iter'],
        )
        #return sklearn.linear_model._logistic.LogisticRegression()
    elif classifier_name == 'LogisticRegressionCV':
        if standard == True: return sklearn.linear_model._logistic.LogisticRegressionCV()

        return sklearn.linear_model._logistic.LogisticRegressionCV(
            #penalty = genes['s_penalty'],
            #solver = genes['s_solver'],
            max_iter = genes['i_max_iter'],
        )
        #return sklearn.linear_model._logistic.LogisticRegressionCV()
    elif classifier_name == 'MLPClassifier':
        if standard == True: return sklearn.neural_network._multilayer_perceptron.MLPClassifier()

        return sklearn.neural_network._multilayer_perceptron.MLPClassifier(
            hidden_layer_sizes = (genes['i_hidden_layer_sizes']),
            activation = genes['s_activation'],
            solver = genes['s_solver'],
            alpha = genes['f_alpha'],
            max_iter = genes['i_max_iter']
        )
        #return sklearn.neural_network._multilayer_perceptron.MLPClassifier()
    elif classifier_name == 'OneVsRestClassifier':
        if standard == True: return sklearn.multiclass.OneVsRestClassifier()

        return sklearn.multiclass.OneVsRestClassifier(
            estimator = genes['m_estimator']
        )
        #return sklearn.multiclass.OneVsRestClassifier(sklearn.ensemble._weight_boosting.AdaBoostClassifier())
    elif classifier_name == 'QuadraticDiscriminantAnalysis':
        return sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()    
    elif classifier_name == 'RandomForestClassifier':
        if standard == True: return sklearn.ensemble._forest.RandomForestClassifier()

        return sklearn.ensemble._forest.RandomForestClassifier(
            n_estimators = genes['i_n_estimators'], 
            criterion = genes['s_criterion'],
            max_depth = genes['i_max_depth'],
            min_samples_split = genes['f_min_samples_split'],
            min_samples_leaf = genes['f_min_samples_leaf'],
            min_weight_fraction_leaf = genes['f_min_weight_fraction_leaf'],
            max_features = genes['f_max_features'],
            oob_score = genes['b_oob_score'],
            bootstrap=True

        )
        #return sklearn.ensemble._forest.RandomForestClassifier()
    elif classifier_name == 'SGDClassifier':
        if standard == True: return sklearn.linear_model._stochastic_gradient.SGDClassifier()

        return sklearn.linear_model._stochastic_gradient.SGDClassifier(
            loss = genes['s_loss'],
            penalty = genes['s_penalty'],
            alpha = genes['f_alpha'],
            max_iter = genes['i_max_iter'],
            learning_rate = genes['s_learning_rate'],
            early_stopping = genes['b_early_stopping'],
            eta0 = 0.001
        )
        #return sklearn.linear_model._stochastic_gradient.SGDClassifier() 
    elif classifier_name == 'XGBClassifier':
        if standard == True: return xgb.XGBClassifier()

        return xgb.XGBClassifier(
            nthread = genes['i_nthread'],
            eta = genes['f_eta'],
            gamma = genes['f_gamma'],
            max_depth = genes['i_max_depth'],
            min_child_weight = genes['i_min_child_weight'],
            max_delta_step = genes['i_max_delta_step'],
            subsample = genes['f_subsample'],
            colsample_bytree = genes['f_colsample_bytree'],
            colsample_bylevel = genes['f_colsample_bylevel'],
            colsample_bynode = genes['f_colsample_bynode'],
            learning_rate = genes['f_learning_rate'],
            n_estimators = genes['i_n_estimators'],

        )
    
