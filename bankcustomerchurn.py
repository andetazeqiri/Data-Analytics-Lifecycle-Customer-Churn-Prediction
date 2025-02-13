

#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

#loading the dataset
dataset =  pd.read_csv('Bank_Customer_Churn_dataset.csv')
dataset.head(10)

# Rownumber,customer id and surname are irrelevent data for finding exited or not
dataset.drop(['RowNumber','CustomerId','Surname'],axis=1,inplace=True)

# Exploratory data analysis
#Relation between Country and Exited
plt.figure(figsize=(8,5))
sns.countplot(x='Exited', hue='Geography', data=dataset)
plt.title('Customer Exits by Geography')
plt.show()
#Thus bank has highest customers in France.Many exited from France and Germany.

plt.figure(figsize=(8,5))
sns.countplot(x='Exited', hue='Gender', data=dataset)
plt.title('Customer Exits by Gender')
plt.show()
#More females are exited from bank than males.

sns.lmplot(x='Balance',y='Age',hue='Exited',data=dataset)
# old people above 45 are tend to exite more than young people below 45.

plt.figure(figsize=(8,5))
sns.countplot(x='Exited', hue='IsActiveMember', data=dataset)
plt.title('Customer Exits by Active Membership Status')
plt.show()
#Active members more tend to leave the bank than inactive members.

plt.figure(figsize=(10,5))
sns.countplot(x='Exited', hue='Tenure', data=dataset)
plt.title('Customer Exits by Tenure')
plt.show()

#people tend to leave the bank more after 1 year on the other way people leaving the bank is less after 9 years.

sns.lmplot(x='Balance',y='EstimatedSalary',hue='Exited',data=dataset)
#people whose salary is between 100000-150000 tends to exite the bank more.

#Feature engineering
#Assigning male=1 and female=0
def gend(val):
    if val == 'Male':
        return 1
    else:
        return 0

dataset['Gender'] = dataset['Gender'].apply(gend)

#rescaling values between 0-1
dataset['NumOfProducts']/=4
dataset['Tenure']/=10

#Applying min max scaling individually to get more accurate range.
dataset['Age'] = minmax_scale(dataset['Age'])
dataset['EstimatedSalary'] = minmax_scale(dataset['EstimatedSalary'])
dataset['Balance'] = minmax_scale(dataset['Balance'])
dataset['CreditScore'] = minmax_scale(dataset['CreditScore'])

#assigning dummy columns by considering each countries separatly.dropping 1 column to avoid overlapping.
df_geo = dataset['Geography']
df_geo = pd.get_dummies(df_geo,drop_first=True)
dataset.drop(['Geography'],axis=1,inplace=True)
df_final = pd.concat([dataset,df_geo],axis=1)

df_final.head(10)

#Training and testing using Random Forest
X = df_final.drop('Exited',axis=1)
y = df_final['Exited']

#25% of the data is allocated for testing, while 75% is used for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=66)
#RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100) #model will create 100 decision trees in the forest
forest.fit(X_train,y_train)
#Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#Artificial Neural Network (ANN)
from sklearn.neural_network import MLPClassifier
ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
ann.fit(X_train, y_train)

#K nearest neighbours algorithm
from sklearn.neighbors import KNeighborsClassifier
# Initialize the KNN model with the desired number of neighbors 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

#pred = logreg.predict(X_test)
# pred=ann.predict(X_test)
# pred=knn.predict(X_test)  
pred = forest.predict(X_test)
print(classification_report(y_test,pred))
print()
print(confusion_matrix(y_test,pred))
