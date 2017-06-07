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
import feature_eng as feature
import matplotlib.pyplot as plt
import numpy as np


def random_forest_classifier(train_data,features,n_trees=50,cv_folds=5):
    model = en.RandomForestClassifier(n_estimators=n_trees)
    print "Testing Random Forest model..."
    scores = ms.cross_val_score(model, train_data[features], train_data["sleep"], cv=cv_folds)
    print "Random Forest. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
    model.fit(train_data[features],train_data["sleep"])
    return {"model":model,"score":scores.mean()}


def decision_tree_classifier(train_data,features,cv_folds=5):
    model = tree.DecisionTreeClassifier()
    print "Testing Decision Tree model..."
    scores = ms.cross_val_score(model, train_data[features], train_data["sleep"], cv=cv_folds)
    print "Decision Tree. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
    model.fit(train_data[features],train_data["sleep"])
    return {"model":model,"score":scores.mean()}


def adaboost_classifier(train_data,features,n_estimators=50,cv_folds=5):
    model = en.AdaBoostClassifier(n_estimators=n_estimators)
    print "Testing Adaboost model..."
    scores = ms.cross_val_score(model, train_data[features], train_data["sleep"], cv=cv_folds)
    print "Adaboost. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
    model.fit(train_data[features],train_data["sleep"])
    return {"model":model,"score":scores.mean()}


def gradient_boosting_classifier(train_data,features,cv_folds=5):
    model = en.GradientBoostingClassifier(loss="exponential")
    print "Testing Gradient Boosting model..."
    scores = ms.cross_val_score(model, train_data[features], train_data["sleep"], cv=cv_folds)
    print "Gradient Boosting. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
    model.fit(train_data[features],train_data["sleep"])
    return {"model":model,"score":scores.mean()}


def logistic_classifier(train_data,features,cv_folds=5):
    model = linear.LogisticRegression()
    print "Testing Logistic Regression model..."
    scores = ms.cross_val_score(model, train_data[features], train_data["sleep"], cv=cv_folds)
    print "Logistic. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
    model.fit(train_data[features],train_data["sleep"])
    return {"model":model,"score":scores.mean()}


def bayes_classifier(train_data,features,cv_folds=5):
    model = nb.BernoulliNB()
    print "Testing Naive Bayes model..."
    scores = ms.cross_val_score(model, train_data[features], train_data["sleep"], cv=cv_folds)
    print "Naive Bayes. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
    model.fit(train_data[features],train_data["sleep"])
    return {"model":model,"score":scores.mean()}


def neural_network_classifier(train_data,features,cv_folds=5):
    model = sk.neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25, 15, 3), random_state=1)
    scores = ms.cross_val_score(model, train_data[features], train_data["sleep"], cv=cv_folds)
    print "Neural Network. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
    model.fit(train_data[features],train_data["sleep"])
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

    results = gradient_boosting_classifier(train_data,features)
    if results["score"] > best_score:
        best_model_name = "GradientBoosting"
        best_model = results["model"]
        best_score = results["score"]

    results = logistic_classifier(train_data,features)
    if results["score"] > best_score:
        best_model_name = "Logistic"
        best_model = results["model"]
        best_score = results["score"]

    results = bayes_classifier(train_data,features)
    if results["score"] > best_score:
        best_model_name = "Bayes"
        best_model = results["model"]
        best_score = results["score"]

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


def predict(model, train_data, features, predict_file):
    model.fit(train_data[features],train_data["sleep"])
    print('Test on data {0}'.format(predict_file))
    predict_csv = pd.read_csv(predict_file)
    test_csv = predict_csv[features]

    # Make prediction and compute accuracy
    predict_csv['predict'] = model.predict(test_csv)
    solution_compare = np.where(predict_csv['sleep'] == predict_csv['predict'], 1, 0)
    total_accurate = sum(solution_compare)
    total_prediction = len(solution_compare)
    accuracy = float(total_accurate)/float(total_prediction)
    print('accuracy on {0} is {1}.'.format(predict_file, accuracy))

    # Make plot
    columns = [#'1_android.sensor.accelerometer_x_avg', '1_android.sensor.accelerometer_y_avg',
              # '1_android.sensor.accelerometer_x_avg',
               #'4_android.sensor.gyroscope_x_avg', '4_android.sensor.gyroscope_y_avg',
               '4_android.sensor.gyroscope_z_avg',
               'sleep', 'predict', 'date_time']
    df = predict_csv[columns]
    df.index = df['date_time']
    df.plot()
    plt.legend(loc='best')
    plt.show()



def main():
    # Generate train data
    all_files = glob.glob("./LabeledData/May_*.csv")
    train_data = pd.DataFrame()
    list_ = []
    for file_ in all_files:
        print ("Reading file {0} into train_data".format(file_))
        df = pd.read_csv(file_, index_col=None, header=0)
        list_.append(df)
    train_data = pd.concat(list_)

    # Generate feature sets
    features = []
    # sensors = ["1_android.sensor.accelerometer", "3_android.sensor.orientation",
    #            "4_android.sensor.gyroscope", "9_android.sensor.gravity",
    #            "10_android.sensor.linear_acceleration"]
    sensors = ["1_android.sensor.accelerometer", "4_android.sensor.gyroscope"]
    variables = ["_x", "_y", "_z", "_m"]
    # variables = ["_x"]
    functions = ["_min","_max","_avg","_delta"]
    for sensor in sensors:
        for variable in variables:
            for function in functions:
                features.append(sensor+variable+function)

    # Get best model from 5 fold cross validation
    # model = get_best_model(train_data, features)

    # Predict on test data
    model = linear.LogisticRegression()
    predict_folder = ['./TestData/Jun_03_2017.csv','./TestData/Jun_05_2017.csv']
    for predict_file in predict_folder:
        predict(model, train_data, features, predict_file)


if __name__ == '__main__':
    main()