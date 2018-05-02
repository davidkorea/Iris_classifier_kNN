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