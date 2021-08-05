# -*- coding: utf-8 -*-
"""
Created on Fri May 22 19:25:22 2020

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:17:54 2020

@author: user
"""
import pandas as pd
mydataset = pd.read_csv('data.csv')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
mydataset.shape
mydataset.head(7)
# FINDING A NULL VALUES
mydataset.isna().sum()
#DROP THE COLUMN WITH ALL MISSING VALUES
mydataset = mydataset.dropna(axis=1)
#GET the new count of the number of rows and columns
mydataset.shape
mydataset.describe()

mydataset.isnull().values.any()

#GET THE COUNT OF THE NUMBER OF MALIGANT (M) OR BENIGN (B) CELLS
mydataset['diagnosis'].value_counts()

print("Cancer data set dimensions : {}".format(mydataset.shape))
#Understand the target varibale further
mydataset["diagnosis"].value_counts()#Visualize dataset using Histograms

#A histogram is a plot that lets you discover, and show, the underlying frequency distribution 
import matplotlib.pyplot as plt

num_bins = 10
mydataset.hist(bins=num_bins, figsize=(20,15))
plt.show()

#VISUALIZE THE COUNT
sns.countplot(mydataset['diagnosis'], label='count')

#Look at the (dtype) data types to see which columns need to be encoded
mydataset.dtypes

#ENCODE THE CATEGORICAL DATA VALUES
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
mydataset.iloc[:,1] = labelencoder_Y.fit_transform(mydataset.iloc[:,1].values)
mydataset.iloc[:,1]
# create apair plot
sns.pairplot(mydataset.iloc[:,1:6])# 6 is nothing but index
sns.pairplot(mydataset.iloc[:,1:6], hue='diagnosis')


#print the first 5 rows of the new data
mydataset.head(5)
#get the correlation of the columns
mydataset.iloc[:,1:12].corr()
"""
#IF WE WANT IN COLOR HEATMAP WE CAN GO FOR IT#
from matplotlib.colors import ListedColormap
plt.figure(figsize=(8, 8))
 
sns.heatmap(mydataset.iloc[:,1:12].corr(), cbar=True, annot=False, fmt='.0%',
            cmap=ListedColormap(['#C71585', '#DB7093', '#FF00FF', '#FF69B4', '#FFB6C1', '#FFC0CB']),
            )
plt.show()
"""
#visualize the corelation
plt.figure(figsize=(10,10))
sns.heatmap(mydataset.iloc[:,1:12].corr(), annot=True, fmt='.0%')

#box plot to check outlier in each category

#define function can be call later 
def boxPlot(dff):
    d=dff.drop(columns=['diagnosis'])
    for column in d:
        plt.figure(figsize=(5,2))
        sns.boxplot(x=column,data=d,palette="colorblind")
boxPlot(mydataset)

#Box plot for radius mean
#We can use box plots to identify outliers in a dataset. 
#These can be rare occurrence or errors. Sometimes these can provide insight.

melted_data = pd.melt(mydataset,id_vars = "diagnosis",value_vars = ['radius_mean'])
plt.figure(figsize = (15,10))
sns.boxplot(x = "variable", y = "value", hue="diagnosis",data= melted_data)
plt.show()

#From the box plot we can see the range of the radius mean values for bening and malignant tumors. 
#We can also clearly see the outliers in the above visualizations (marked points lying outside the range).

#Box plot for area mean
melted_data = pd.melt(mydataset,id_vars = "diagnosis",value_vars = ['area_mean'])
plt.figure(figsize = (15,10))
sns.boxplot(x = "variable", y = "value", hue="diagnosis",data= melted_data)
plt.show()
#INFERENCE: From the graph, we can say that after a certain threshold for the values of the given features,
# we can classify the tumor as malignant (but we also see outliers, so maybe not with a 100% certainty 
#every time, there could be some false positives).

#Box plot for perimeter mean
melted_data = pd.melt(mydataset,id_vars = "diagnosis",value_vars = ['perimeter_mean'])
plt.figure(figsize = (15,10))
sns.boxplot(x = "variable", y = "value", hue="diagnosis",data= melted_data)
plt.show()


#split the data set into independent (X) and dependent (y) data sets
X=mydataset.iloc[:,2:31].values
Y=mydataset.iloc[:,1].values
#split the dataset into 75% training and 25% testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.25, random_state =0)
#scale the data(feature scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
X_train
  Y_train
 X_test
 Y_test
 
 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


#Model Evaluation
#1 LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
#Train the model
classifier.fit(X_train, Y_train)
#Test it using the test dataset
Y_pred1 = classifier.predict(X_test)

import seaborn as sns
def seaborn_plot_cm(cm, model_name):
  sns.heatmap(cm, annot = True, fmt = '.2f', 
            xticklabels = ['Benign', 'Malignant'], yticklabels = ['Benign', 'Malignant'])
  plt.ylabel('True Class')
  plt.xlabel('Predicted Class')

  plt.title(model_name)
  plt.savefig(model_name)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
def print_classification_report(classifier_name, Y_pred1, Y_test):
  print(classifier_name + ' Accuracy: {:.3f}'.format(accuracy_score(Y_test, 
                                                                   Y_pred1)))

  cm = confusion_matrix(Y_test, Y_pred1)
  print(classification_report(Y_test, Y_pred1))
  seaborn_plot_cm(cm, classifier_name)
  
    print_classification_report('Logistic Regression', Y_pred1, Y_test)
  cm = confusion_matrix(Y_test, Y_pred1)

tp = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]
tn = cm[1,1]
print ('True Positive  >', tp)
print ('False Positive >', fp)
print ('False Negetive >', fn)
print ('True Negetive  >', tn) 
print ('Final prediction >', round((tp+tn)/(len(Y_pred1))*100,2))

# 1.2 roc curve 
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt
fpr, tpr, _ = roc_curve(Y_test, Y_pred1)
print('ROC AUC: %.3f' % roc_auc_score( y_true=Y_test, y_score=Y_pred1))
plt.clf()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()

   
#2 RANDOM FOREST   
from sklearn.ensemble import RandomForestClassifier
rd_forest_classifier = RandomForestClassifier(n_estimators = 10, 
                                              criterion = 'entropy', 
                                              random_state = 0)
rd_forest_classifier.fit(X_train, Y_train)
rd_forest_Y_pred = rd_forest_classifier.predict(X_test)
print_classification_report('Random Forest', rd_forest_Y_pred, Y_test)

#2.1 confusion matrix for random forest
cm = confusion_matrix(Y_test, rd_forest_Y_pred)
tp = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]
tn = cm[1,1]
print ('True Positive  >', tp)
print ('False Positive >', fp)
print ('False Negetive >', fn)
print ('True Negetive  >', tn) 
print ('Final prediction >', round((tp+tn)/(len(rd_forest_Y_pred))*100,2))
  
#2.2 ROC
fpr, tpr, _ = roc_curve(Y_test, rd_forest_Y_pred)
print('ROC AUC: %.3f' % roc_auc_score( y_true=Y_test, y_score=rd_forest_Y_pred))
plt.clf()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()


#3 GAUSSIAN NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
gbNB_classifier = GaussianNB()
gbNB_classifier.fit(X_train, Y_train)
gb_NB_Y_pred = gbNB_classifier.predict(X_test)
print_classification_report('Gaussian Naive Bayes', gb_NB_Y_pred, Y_test)

#3.1confusion matrix
cm = confusion_matrix(Y_test, gb_NB_Y_pred)
tp = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]
tn = cm[1,1]
print ('True Positive  >', tp)
print ('False Positive >', fp)
print ('False Negetive >', fn)
print ('True Negetive  >', tn) 
print ('Final prediction >', round((tp+tn)/(len(gb_NB_Y_pred))*100,2))

#3.2ROC FOR NAIVE BAYES
fpr, tpr, _ = roc_curve(Y_test, gb_NB_Y_pred)
print('ROC AUC: %.3f' % roc_auc_score( y_true=Y_test, y_score=gb_NB_Y_pred))
plt.clf()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()

#4 DECISION TREE

from sklearn.tree import DecisionTreeClassifier
DT_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DT_classifier.fit(X_train, Y_train)
DT_pred = DT_classifier.predict(X_test)
print_classification_report('Decision Tree', DT_pred, Y_test)

#4.1 confusion matrix
cm = confusion_matrix(Y_test, DT_pred)
tp = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]
tn = cm[1,1]
print ('True Positive  >', tp)
print ('False Positive >', fp)
print ('False Negetive >', fn)
print ('True Negetive  >', tn) 
print ('Final prediction >', round((tp+tn)/(len(DT_pred))*100,2))

#4.2 ROC FOR NAIVE BAYES
fpr, tpr, _ = roc_curve(Y_test, DT_pred)
print('ROC AUC: %.3f' % roc_auc_score( y_true=Y_test, y_score=DT_pred))
plt.clf()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()



"""
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = classifier_6.predict_proba(X_test)
Y_pred6 = classifier_6.predict(X_test)

print (roc_auc_score(X_test, predict_proba[:,1])
"""

#lets plot the bar graph
#accuracy score
import numpy as np
import matplotlib.pyplot as plt
ALGORITHM = [95.1, 93.71, 96.5, 94.41]
ACCURACY_SCORE = ('LR', 'DT', 'RF', 'GNB')
y_pos = np.arange(len(ACCURACY_SCORE))
plt.bar(y_pos, ALGORITHM, color=('r', 'b','orange', 'salmon'))
plt.xticks(y_pos, ACCURACY_SCORE)
plt.show()




#roc auc
ALGORITHM = [0.949, 0.942, 0.964, 0.936]
ROC_AUC = ('LR', 'DT', 'RF', 'GNB')
y_pos = np.arange(len(ROC_AUC))
plt.bar(y_pos, ALGORITHM, color=('g', 'r','yellow', 'b'))
plt.xticks(y_pos, ROC_AUC)
plt.show()
