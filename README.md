# Iris_classifier_kNN

# 1. Basic

![](https://github.com/davidkorea/Iris_classifier_kNN/blob/master/images/scikit-learn.png)

![](https://github.com/davidkorea/Iris_classifier_kNN/blob/master/images/dataformat.png)

![](https://github.com/davidkorea/Iris_classifier_kNN/blob/master/images/kNN.png)
 
# 2. 코드요약

0. preparation
```php
DATA_FILE = ''
FEATURE_COL = ['mass','width','height','color_score']
SPECIES_LABEL_DICT {
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


