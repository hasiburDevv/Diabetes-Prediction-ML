# importing the libraries 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# loadin the data in a dataframe from .csv file
df = pd.read_csv("pima_native_american_diabetes_dataset.csv")

# preprocessing
# replacing the missing values by mean values 
df.loc[df['serum_insulin'] == 0, 'serum_insulin'] = df['serum_insulin'].mean();
df.loc[df['plasma_glucose'] == 0, 'plasma_glucose'] = df['plasma_glucose'].mean();
df.loc[df['body_mass_index'] == 0, 'body_mass_index'] = df['body_mass_index'].mean();
df.loc[df['diastolic_blood_pressure'] == 0, 'diastolic_blood_pressure'] = df['diastolic_blood_pressure'].mean();
df.loc[df['tricep_skin_fold_thickness'] == 0, 'tricep_skin_fold_thickness'] = df['tricep_skin_fold_thickness'].mean();

# normalization
df = df / df.max()

#spliting the dataset
input = df.drop('class', axis = 'columns')
target = df['class']
X_train, X_test, y_train, y_test = train_test_split(input, target, test_size = 0.25, random_state = 42)

from sklearn.svm import SVC
model = SVC(gamma = 'scale')

model.fit(X_train, y_train)

y_predicted = model.predict(X_test);

#finding the accuracy score
Accuracy = accuracy_score(y_test, y_predicted)
print('*' * 50, '\n')
print("Accuracy Score for Support Vector Machine: ", Accuracy);

# building the confusion matrix
cm = confusion_matrix(y_test, y_predicted)

# printing the confusion Matrix
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['0','1']
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()

# Precision
prec = cm[1][1] / (cm[0][1]+cm[1][1])
print("Precision: ", prec)

#True positive rate
tpr = cm[1][1]/(cm[1][0]+cm[1][1])
print("True Positive Rate: ", tpr)

#True negative rate
tnr = cm[0][0]/(cm[0][0]+cm[0][1])
print("True Negative Rate: ", tnr)

#False positive rate
fpr = cm[0][1]/(cm[0][0]+cm[0][1])
print("False Positive Rate: ", fpr)

#False negative rate
fnr = cm[1][0]/(cm[1][0]+cm[1][1])
print("False Negative Rate: ", fnr)



