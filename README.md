# Iris_classifier_kNN

# 1. Basic
## 기초
![](https://github.com/davidkorea/Iris_classifier_kNN/blob/master/images/scikit-learn.png)

![](https://github.com/davidkorea/Iris_classifier_kNN/blob/master/images/dataformat.png)

![](https://github.com/davidkorea/Iris_classifier_kNN/blob/master/images/kNN.png)

## 중간
![](https://github.com/davidkorea/Iris_classifier_kNN/blob/master/images/metric.jpg)

![](https://github.com/davidkorea/Iris_classifier_kNN/blob/master/images/n_neighbors.png)

![](https://github.com/davidkorea/Iris_classifier_kNN/blob/master/images/params.png?raw=true)


# 2. 코드요약

0. preparation
```php
DATA_FILE = ''
FEATURE_COL = ['mass','width','height','color_score']
SPECIES_LABEL_DICT = {
  'apple':0,
  'mandarin':1,
  'orange':2,
  'lemon':3  
}
```
1. read data & change 'str'label to 'int'
```php
imort pandas as pd

fruit_data = pd.read_csv(DATA_FILE)
# add a new column in the fruit data.
fruit_data['Label'] = fruit_data['fruit_name'].map(SPECIES_LABEL_DICT)
```
2. define X & y / features & labels
```php
# X = features & y = labels
X = fruit_data[FEATURE_COL].values
y = fruit_data['Label'].values
```
3. split train & test data
```php
from sklearn.model_selection import

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=1/5, random_state=20)
```
4. declare kNN model & fit model & evaluate model
```php
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()
knn_model.fit(X_train,y_train)
accuracy = knn_model.score(X_test,y_test)
print('Accuracy: {:.2f}'.format(accuracy))
```
5. run by one sample
```php
idx = 8
test_sample_feat = [X_test,:]]
true_label = y_test[idx]
pred_label = knn_model.predict(test_sample_feat)
print('sample{}, true:{},predict:{}'.format(idx,true_label,pred_label))
```

# 3. n_neighbors 코드요약
 
```php
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

DATA_FILE = './data_ai/Iris.csv'
FEATURE_COL = ['SepalLengthCm', 'SepalWidthCm',
                             'PetalLengthCm', 'PetalWidthCm']
SPECIES_LABEL_DICT = {
    'Iris-setosa': 0,  # 山鸢尾
    'Iris-versicolor': 1,  # 变色鸢尾
    'Iris-virginica': 2  # 维吉尼亚鸢尾
}

def investigate(iris_data,k_val):
    X = iris_data[FEATURE_COL].values
    y = iris_data['Label'].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=10)

    knn_model = KNeighborsClassifier(n_neighbors=k_val)
    knn_model.fit(X_train,y_train)
    accuracy = knn_model.score(X_test,y_test)
    print('K: {} - Accuracy: {}'.format(k_val,accuracy))

def main():
    iris_data = pd.read_csv(DATA_FILE)
    iris_data['Label'] = iris_data['Species'].map(SPECIES_LABEL_DICT)

    k_vals = [3,5,10,15]
    for k_val in k_vals:
        investigate(iris_data,k_val)

main()

# Result:
# K: 3 - Accuracy: 0.96
# K: 5 - Accuracy: 0.96
# K: 10 - Accuracy: 1.0
# K: 15 - Accuracy: 0.96
```

# 4. n_neighbors 코드요약 & plot boundary

```php
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import ai_utils

DATA_FILE = './data_ai/Iris.csv'
FEATURE_COL = ['SepalLengthCm', 'SepalWidthCm',
                             'PetalLengthCm', 'PetalWidthCm']
SPECIES_LABEL_DICT = {
    'Iris-setosa': 0,  # 山鸢尾
    'Iris-versicolor': 1,  # 变色鸢尾
    'Iris-virginica': 2  # 维吉尼亚鸢尾
}

def investigate(iris_data,k_val,sep_feat_col):
    X = iris_data[sep_feat_col].values
    y = iris_data['Label'].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=10)

    knn_model = KNeighborsClassifier(n_neighbors=k_val)
    knn_model.fit(X_train,y_train)
    accuracy = knn_model.score(X_test,y_test)
    print('K: {} - Sepal - Accuracy: {}'.format(k_val,accuracy))
    ai_utils.plot_knn_boundary(knn_model, X_test, y_test,
                               'Sepal Length vs Sepal Width, k={}'.format(k_val),
                               save_fig='sepal_k={}.png'.format(k_val))

def investigate_pet(iris_data,k_val):
    pet_feat_col = ['PetalLengthCm', 'PetalWidthCm']
    A = iris_data[pet_feat_col].values
    b = iris_data['Label'].values

    A_train,A_test,b_train,b_test = train_test_split(A,b,test_size=1/3,random_state=10)
    knn_model_pet = KNeighborsClassifier()
    knn_model_pet.fit(A_train,b_train)
    accuracy_pet = knn_model_pet.score(A_test,b_test)
    print('K: {} - Petal - Accuracy: {}'.format(k_val,accuracy_pet))
    ai_utils.plot_knn_boundary(knn_model_pet, A_test, b_test,
                               'Petal Length vs Petal Width, k={}'.format(k_val),
                               save_fig='petal_k={}.png'.format(k_val))

def main():
    iris_data = pd.read_csv(DATA_FILE)
    iris_data['Label'] = iris_data['Species'].map(SPECIES_LABEL_DICT)

    sep_feat_col = ['SepalLengthCm', 'SepalWidthCm']
    k_vals = [3,5,10,15]
    for k_val in k_vals:
        investigate(iris_data,k_val,sep_feat_col)
        investigate_pet(iris_data,k_val)

main()

# Result:
# K: 3 - Sepal - Accuracy: 0.66
# K: 3 - Petal - Accuracy: 0.98
# K: 5 - Sepal - Accuracy: 0.68
# K: 5 - Petal - Accuracy: 0.98
# K: 10 - Sepal - Accuracy: 0.78
# K: 10 - Petal - Accuracy: 0.98
# K: 15 - Sepal - Accuracy: 0.78
# K: 15 - Petal - Accuracy: 0.98
# Petal plots as below
```
![](https://github.com/davidkorea/Iris_classifier_kNN/blob/master/images/petal_results.png)
