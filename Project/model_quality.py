import pandas as pd
import glob
import sklearn as sk
import sklearn.ensemble as en
import sklearn.tree as tree
import sklearn.linear_model as linear
import sklearn.naive_bayes as nb
import sklearn.neural_network as nn
import sklearn.feature_selection as fs
import sklearn.model_selection as ms
import quality_test as feature


def random_forest_classifier(train_data,features,n_trees=50,cv_folds=5):
    model = en.RandomForestClassifier(n_estimators=n_trees)
    print "Testing Random Forest model..."
    scores = ms.cross_val_score(model, train_data[features], train_data["quality"], cv=cv_folds)
    print "Random Forest. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
    model.fit(train_data[features],train_data["quality"])
    return {"model":model,"score":scores.mean()}


def decision_tree_classifier(train_data,features,cv_folds=5):
    model = tree.DecisionTreeClassifier()
    print "Testing Decision Tree model..."
    scores = ms.cross_val_score(model, train_data[features], train_data["quality"], cv=cv_folds)
    print "Decision Tree. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
    model.fit(train_data[features],train_data["quality"])
    return {"model":model,"score":scores.mean()}


def adaboost_classifier(train_data,features,n_estimators=50,cv_folds=5):
    model = en.AdaBoostClassifier(n_estimators=n_estimators)
    print "Testing Adaboost model..."
    scores = ms.cross_val_score(model, train_data[features], train_data["quality"], cv=cv_folds)
    print "Adaboost. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
    model.fit(train_data[features],train_data["quality"])
    return {"model":model,"score":scores.mean()}


def gradient_boosting_classifier(train_data,features,cv_folds=5):
    model = en.GradientBoostingClassifier(loss="exponential")
    print "Testing Gradient Boosting model..."
    scores = ms.cross_val_score(model, train_data[features], train_data["quality"], cv=cv_folds)
    print "Gradient Boosting. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
    model.fit(train_data[features],train_data["quality"])
    return {"model":model,"score":scores.mean()}


def logistic_classifier(train_data,features,cv_folds=5):
    model = linear.LogisticRegression()
    print "Testing Logistic Regression model..."
    scores = ms.cross_val_score(model, train_data[features], train_data["quality"], cv=cv_folds)
    print "Logistic. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
    model.fit(train_data[features],train_data["quality"])
    return {"model":model,"score":scores.mean()}


def bayes_classifier(train_data,features,cv_folds=5):
    model = nb.BernoulliNB()
    print "Testing Naive Bayes model..."
    scores = ms.cross_val_score(model, train_data[features], train_data["quality"], cv=cv_folds)
    print "Naive Bayes. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
    model.fit(train_data[features],train_data["qaulity"])
    return {"model":model,"score":scores.mean()}


def neural_network_classifier(train_data,features,cv_folds=5):
    model = sk.neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25, 15, 3), random_state=1)
    scores = ms.cross_val_score(model, train_data[features], train_data["quality"], cv=cv_folds)
    print "Neural Network. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
    model.fit(train_data[features],train_data["quality"])
    return {"model":model,"score":scores.mean()}


def get_best_model(train_data,features):
    best_score = 0
    best_model = None
    best_model_name = ""

    results = random_forest_classifier(train_data,features)
    if results["score"] > best_score:
        best_model_name = "RandomForest"
        best_model = results["model"]
        best_score = results["score"]

    results = decision_tree_classifier(train_data,features)
    if results["score"] > best_score:
        best_model_name = "DecisionTree"
        best_model = results["model"]
        best_score = results["score"]

    results = adaboost_classifier(train_data,features)
    if results["score"] > best_score:
        best_model_name = "Adaboost"
        best_model = results["model"]
        best_score = results["score"]

##    results = gradient_boosting_classifier(train_data,features)
##    if results["score"] > best_score:
##        best_model_name = "GradientBoosting"
##        best_model = results["model"]
##        best_score = results["score"]

    results = logistic_classifier(train_data,features)
    if results["score"] > best_score:
        best_model_name = "Logistic"
        best_model = results["model"]
        best_score = results["score"]

##    results = bayes_classifier(train_data,features)
##    if results["score"] > best_score:
##        best_model_name = "Bayes"
##        best_model = results["model"]
##        best_score = results["score"]

    results = neural_network_classifier(train_data,features)
    if results["score"] > best_score:
        best_model_name = "Neural Network"
        best_model = results["model"]
        best_score = results["score"]

    # results = grid_search(train_data,features)
    # if results["score"] > best_score:
    #     best_model_name = "Grid Search"
    #     best_model = results["model"]
    #     best_score = results["score"]

    print "Best Model:{0}, score:{1}".format(best_model_name, best_score)

    return best_model


def main():
    # Generate train data
    #all_files = glob.glob("./sleepqualityfeature.csv")
   
    train_data = pd.DataFrame()
    list_ = []
    df = pd.read_csv("./LabeledData/sleepqualityfeature1.csv", index_col=None, header=0)
    list_.append(df)
    train_data = pd.concat(list_)
    #train_data = list_
    #train_data = df['times of movements']
    # Generate feature sets
    #features['times_of_movements'] 
    features = []
    numbers = ['1','2','3','4','5','6','7','8']
    for number in numbers:
        features.append('times_of_movements'+number)

 

    # Get best model
    model = get_best_model(train_data, features)
    feature.plt.show()


if __name__ == '__main__':
    main()


