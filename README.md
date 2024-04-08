# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Kanimozhi
RegisterNumber:  212222230060
*/
```
import pandas as pd
data = pd.read_csv('dataset/Placement_Data.csv')
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:, :-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
## Output:

![Exp4_1](https://github.com/kanimozhipannerselvam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476060/239a3dc2-ad07-4445-a784-1809d45e4811)

![Exp4_2](https://github.com/kanimozhipannerselvam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476060/558c7f0b-d2fd-4f0f-9c48-7df0d8562f5e)

![Exp4_3](https://github.com/kanimozhipannerselvam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476060/b9f34509-62f8-4c6b-99ea-d07f555db209)

![Exp4_4](https://github.com/kanimozhipannerselvam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476060/0e658745-b1a5-4a41-ac6c-ef62f3cbf406)

![Exp4_5](https://github.com/kanimozhipannerselvam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476060/b014cbcf-28aa-4e0e-add5-0f5a4375ce30)

![Exp4_6](https://github.com/kanimozhipannerselvam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476060/c65d744f-7b06-43bd-adab-e1525e9f10b3)

![Exp4_7](https://github.com/kanimozhipannerselvam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476060/656b940b-25dc-4796-b778-cf7740f59642)

![Exp4_8](https://github.com/kanimozhipannerselvam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476060/914b2aa2-cf25-4b9b-a2b9-0c7d15ae9ec9)

![Exp4_9](https://github.com/kanimozhipannerselvam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476060/b2f21227-7b72-4fd0-84d5-fb4e15d6ce70)

![Exp4_10](https://github.com/kanimozhipannerselvam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476060/2c944cfe-1634-4865-8900-e85c82a9086b)

![Exp4_11](https://github.com/kanimozhipannerselvam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476060/39ad53e8-7d8d-4cff-b053-d8cbe58c3d40)

![Exp4_12](https://github.com/kanimozhipannerselvam/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476060/b1b41322-fb3f-4b1a-b5ef-aa4a9640cb6c)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
