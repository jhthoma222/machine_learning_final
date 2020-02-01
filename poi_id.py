#!/usr/bin/python

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pprint
from time import time
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


### Task 1: Select what features you'll use.

features_list = ['poi',
 'to_poi_ratio',
 'from_poi_ratio',                
 'bonus',
 'exercised_stock_options',
 'expenses',
 'from_messages',
 'from_poi_to_this_person',
 'from_this_person_to_poi',
 'other',
 'restricted_stock',
 'salary',
 'shared_receipt_with_poi',
 'to_messages',
 'total_payments',
 'total_stock_value']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")

# Correcting the data for Belfer, according to the pdf
data_dict['BELFER ROBERT']['director_fees'] = 102500
data_dict['BELFER ROBERT']['deferred_income'] = -102500
data_dict['BELFER ROBERT']['deferral_payments'] = 'NaN'
data_dict['BELFER ROBERT']['expenses'] = 3285 
data_dict['BELFER ROBERT']['total_payments'] = 102500
data_dict['BELFER ROBERT']['restricted_stock'] = 44093
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093
data_dict['BELFER ROBERT']['total_stock_value'] = "NaN"

# Correcting the data for Belfer, according to the pdf
data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290
data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864
data_dict['BHATNAGAR SANJAY']['other'] = 'NaN'


### Task 3: Create new feature(s)
def create_ratio(data_dict, ratio_name, numerator, denominator):
    for person in data_dict:
        if data_dict[person][numerator] == 'NaN' or data_dict[person][denominator] == 'NaN':
                data_dict[person][ratio_name] = 'NaN'
        else:
            data_dict[person][ratio_name] = float(data_dict[person][numerator])/float(data_dict[person][denominator])
    return data_dict

data_dict = create_ratio(data_dict, 'from_poi_ratio', 'from_poi_to_this_person', 'to_messages' )
data_dict = create_ratio(data_dict, 'to_poi_ratio', 'from_this_person_to_poi', 'from_messages' )

features_list.append('from_poi_ratio')
features_list.append('to_poi_ratio')

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


### Task 4: Try a varity of classifiers

'''
#Naive Bayes
from sklearn.naive_bayes import GaussianNB

t0 = time()

clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
precision = precision_score(labels_test, pred)
recall = recall_score(labels_test, pred)

print "Naive Bayes Accuracy :", accuracy
print "training time:", round(time()-t0, 3), "s"
print
report = classification_report(labels_test, pred)
print report
print "--------------------------------------"
'''

#Decision Tree
from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
precision = precision_score(labels_test, pred)
recall = recall_score(labels_test, pred)

print "Decision Tree Accuracy :", accuracy
print "training time:", round(time()-t0, 2), "s"
print 'Precision :', precision
print 'Recall :', recall
print
report = classification_report(labels_test, pred)
print report
print "--------------------------------------"


'''
#Adaboost
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
precision = precision_score(labels_test, pred)
recall = recall_score(labels_test, pred)


print "AdaBoost Accuracy :", accuracy
print "training time:", round(time()-t0, 3), "s"
print 'Precision :', precision
print 'Recall :', recall
print
report = classification_report(labels_test, pred)
print report
print "--------------------------------------"
'''

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)