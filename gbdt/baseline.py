#!/bin/env python
#coding=utf-8
################################################################
# File: a.py
# Created Time: 2020/11/12 10:53:37
################################################################

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , mean_squared_error
from sklearn.datasets import load_boston

def test_classifier():
    X, y = load_iris(return_X_y=True) # 3分类

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, 
        random_state=0)

    clf = DecisionTreeClassifier(random_state=0).fit(X_train, Y_train) 
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_pred, Y_test) # 0.9777
    print(acc)



#with open('train.txt', 'w') as f:
#    for x, y in zip(X_train, Y_train):
#        x = '\t'.join([str(i) for i in list(x)])
#        f.write('%d\t%s\n' % (y, x)
#        )  
#with open('test.txt', 'w') as f:
#    for x, y in zip(X_test, Y_test):
#        x = '\t'.join([str(i) for i in list(x)])
#        f.write('%d\t%s\n' % (y, x)
#        ) 


def test_regressor():
    boston = load_boston()
    X, Xtest, Y, Ytest = train_test_split(boston.data, boston.target, test_size=0.2, random_state=0)

    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X, Y)
    y_pred = regressor.predict(Xtest)
    err = mean_squared_error(Ytest, y_pred)
    print(err) # 32.41637254901961

    from dt import DecisionTree
    mine = DecisionTree(criterion="mse", classifier=False)
    mine.fit(X,Y)
    y_pred = mine.predict(Xtest)
    err = mean_squared_error(Ytest, y_pred)
    print(err) # 32.74450980392157

    #with open('boston.train', 'w') as f:
    #    for x, y in zip(X, Y):
    #        x = '\t'.join([str(i) for i in list(x)])
    #        f.write('%f\t%s\n' % (y, x)
    #        )  
    #with open('boston.test', 'w') as f:
    #    for x, y in zip(Xtest, Ytest):
    #        x = '\t'.join([str(i) for i in list(x)])
    #        f.write('%f\t%s\n' % (y, x)
    #        ) 

test_regressor()
