import numpy as np
import pandas as pd
import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn import metrics








def main(args):
    dataDF = pd.read_csv(args.PandasDataFrame)

    # Generate X, Y data sets.
    X = dataDF.as_matrix(dataDF.columns[:-1])
    Y = dataDF.as_matrix(dataDF.columns[-1])

    # Create each model.
    linearModel = LinearRegression()
    logisticModel = LogisticRegression()
    dtModel = DecisionTreeClassifier()
    rfModel = RandomForestClassifier()
    svmModel = SVC()

    # Fit each model.
    linearModel.fit(X, Y)
    logisticModel.fit(X, Y)
    dtModel.fit(X, Y)
    rfModel.fit(X, Y)
    svmModel.fit(X, Y)

    # Evaluate each model.
    scoring = ['precision_macro', 'recall_macro']
    scores = cross_validate(clf, iris.data, iris.target, scoring=scoring, cv = 5, return_train_score = False)


    # Print results.








if __name__ == '__main__':
	#Parse command line arguments
	parser = argparse.ArgumentParser(description="""Entity Classifier Suite. This script trains and tests multiple classifiers
	on the pandas data frame passed to this.""")
	parser.add_argument('PandasDataFrame', metavar='f', type=str)
	args = parser.parse_args()
	main(args)












