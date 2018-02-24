import numpy as np
import pandas as pd
import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import time








def main(args):
    dataDF = pd.read_csv(args.PandasDataFrame)

    # Generate X, Y data sets.
    X = dataDF.as_matrix(dataDF.columns[1:-1])
    Y = np.array(list(map(lambda item: 1 if item == '+' else -1, dataDF[dataDF.columns[-1]])))
    Y = Y.reshape((Y.shape[0], 1))

    # Create each model.
    linearModel = LinearRegression()
    logisticModel = LogisticRegression()
    dtModel = DecisionTreeClassifier(min_samples_leaf=7)
    rfModel = RandomForestClassifier(min_samples_leaf=7, n_estimators=10, max_depth=15)
    abModel = AdaBoostClassifier()#base_estimator=SVC(C=0.1, tol=0.1, verbose=1, kernel='linear', probability=True), n_estimators=10)
    # 50 for data.
    svmModel = SVC(C=0.1, tol=1, verbose=1, kernel='linear')# ( , class_weight='balanced')

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=np.random)
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # Fit each model.
        print('Training linear model...')
        linearModel.fit(X_train, y_train.ravel())
        print('Done!')
        print('Training logistic model...')
        logisticModel.fit(X_train, y_train.ravel())
        print('Done!')
        print('Training decision tree model...')
        dtModel.fit(X_train, y_train.ravel())
        print('Done!')
        print('Training random forrests model...')
        rfModel.fit(X_train, y_train.ravel())
        print('Done!')
        print('Training svm model...')
        startTime = time.time()
        svmModel.fit(X_train, y_train.ravel())
        endTime = time.time()
        print('Done! %s'%(str(endTime-startTime)))
        print('Training adaboost model...')
        abModel.fit(X_train, y_train.ravel())
        print('Done!')

        # Evaluate each model.
        scoring = ['precision_macro', 'recall_macro']
        scores = [0 for i in range(6)]
        print('Testing linear model...')
        predictions = linearModel.predict(X_test)
        predictions = list(map(lambda item: 1 if item >= 0 else -1, predictions))
        scores[0] = (precision_score(y_test, predictions, average='macro'), recall_score(y_test, predictions, average='macro'))
        print('Done!')
        print('Testing logistic model...')
        predictions = linearModel.predict(X_test)
        predictions = list(map(lambda item: 1 if item >= 0.5 else -1, predictions))
        scores[1] = (precision_score(y_test, predictions, average='macro'), recall_score(y_test, predictions, average='macro'))
        print('Done!')
        print('Testing decision tree...')
        scores[2] = (precision_score(y_test, dtModel.predict(X_test), average='macro'), recall_score(y_test, dtModel.predict(X_test), average='macro'))
        print('Done!')
        print('Testing random forrest model...')
        scores[3] = (precision_score(y_test, rfModel.predict(X_test), average='macro'), recall_score(y_test, rfModel.predict(X_test), average='macro'))
        print('Done!')
        print('Testing SVM model...')
        startTime = time.time()
        scores[4] = (precision_score(y_test, svmModel.predict(X_test), average='macro'), recall_score(y_test, svmModel.predict(X_test), average='macro'))
        endTime = time.time()
        print('Done! %s'%(str(endTime-startTime)))
        print('Testing adaboost model...')
        scores[5] = (precision_score(y_test, abModel.predict(X_test), average='macro'), recall_score(y_test, abModel.predict(X_test), average='macro'))
        print('Done!')

        # Print results.
        print(scores)







if __name__ == '__main__':
	#Parse command line arguments
	parser = argparse.ArgumentParser(description="""Entity Classifier Suite. This script trains and tests multiple classifiers
	on the pandas data frame passed to this.""")
	parser.add_argument('PandasDataFrame', metavar='f', type=str)
	args = parser.parse_args()
	main(args)












