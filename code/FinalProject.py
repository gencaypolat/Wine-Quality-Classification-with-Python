import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("winequality-red.csv")    #Importing the Dataset

print(df.head())            #First 5 rows of the Dataset

plt.figure(figsize=(10,6))          #Figure Size
corr = df.corr()                    #Correlation method
sns.heatmap(corr,annot=True,cmap=sns.diverging_palette(200, 10, as_cmap=True))      #Heat Map (correlation matrix)
plt.show()

#From the graph, we can see that alcohol is most strongly correlated with quality, and the correlation is positive.

#Variations in alcohol levels for wines of different qualities using a bar graph.
plt.bar(df['quality'], df['alcohol'])
plt.title('Relationship between alcohol and quality')
plt.xlabel('Quality')
plt.ylabel('Alcohol')
plt.show()
#Immediately, we know that wines of lower quality tend to have a lower level of alcohol.

#Normalizing the data will transform the data so that its distribution has a uniform range.
# It’s important to equalize the ranges of the data here because in our dataset citric acid
# and volatile acidity, for example, have all of their values between 0 and 1.
# In contrast, total sulfur dioxide has some values over 100 and some values below 10.
# This disparity in ranges may cause a problem since a small change in a feature might not affect the other.
# To address this problem, I normalize the ranges of the dataset to a uniform range between 0 and 1.

scaler = MinMaxScaler(feature_range=(0, 1))
normal_df = scaler.fit_transform(df)
normal_df = pd.DataFrame(normal_df, columns = df.columns)
print(normal_df.head())

#Classification models will finally output “yes” or “no” to predict wine quality.
#"Good wine” equals “yes” when the quality is equal or above 7.
#“Good wine” equals “no” when the quality is less than 7.

df["good wine"] = ["yes" if i >= 7 else "no" for i in df['quality']]
X = normal_df.drop(["quality"], axis = 1)
y = df["good wine"]

#Partition X and y into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

#Supoort Vector Machine

svc = SVC()
svc.fit(X_train, y_train)       # fitting the training data to an SVM

y_pred = svc.predict(X_test)    #predicting the outcomes for the test set
acc_sv = metrics.accuracy_score(y_test, y_pred)
print("SVM Accuracy = ", acc_sv)        #printing the accuracy score of SVM

pred_svc = svc.predict(X_test)
print(classification_report(y_test, pred_svc))

#Random Forest

rf_model = RandomForestClassifier()
rf_model.fit(X_train,y_train)           #fitting the training data to a random forest model

y_pred_rf = rf_model.predict(X_test)        #predicting the outcomes for the test set
acc_rf = accuracy_score(y_test,y_pred_rf)
print('Random Forest Accuracy = ', acc_rf)      #printing the accuracy score of Random Forest

pred_rfc = rf_model.predict(X_test)
print(classification_report(y_test, pred_rfc))

#Feature Importance with Random Forest

imp_rf = pd.DataFrame(zip(X_train.columns, rf_model.feature_importances_),columns = ["feature", "importance"])
imp_rf.set_index("feature", inplace=True)
imp_rf.sort_values(by = "importance", ascending = False, inplace = True)
print(imp_rf.head())

imp_rf.plot.barh(figsize=(10,10))     #creating a horizontal bar graph
plt.show()                            #visualizing the feature importances

#Conclusion

df_good = df[df["good wine"] == "yes"]
df_bad = df[df["good wine"] == "no"]

#Here "Bad Wine" means "Not Good Wine"
print("Good Wine Alcohol Average = " + str(np.average(df_good["alcohol"])))     #Good wine alcohol average
print("Bad Wine Alcohol Average = " + str(np.average(df_bad["alcohol"])))      #Bad wine alcohol average

print("Good Wine Sulphates Average = " + str(np.average(df_good["sulphates"])))   #good wine sulphates average
print("Bad Wine Sulphates Average = " + str(np.average(df_bad["sulphates"])))    #bad wine sulphates average

print("Good Wine Volatile Acidity Average = " + str(np.average(df_good["volatile acidity"])))      #good wine volatile acidity average
print("Bad Wine Volatile Acidity Average = " + str(np.average(df_bad["volatile acidity"])))        #bad wine volatile acidity average