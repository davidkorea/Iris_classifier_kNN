import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

DATA_FILE = './data_ai/Iris.csv'
FEATURE_COL = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
SPECIES_LABEL_DICT = {
    'Iris-setosa': 0,  # 山鸢尾
    'Iris-versicolor': 1,  # 变色鸢尾
    'Iris-virginica': 2  # 维吉尼亚鸢尾
}

def main():
    iris_data = pd.read_csv(DATA_FILE)
    iris_data['Label'] = iris_data['Species'].map(SPECIES_LABEL_DICT)

    # get features
    X = iris_data[FEATURE_COL].values
    # get labels
    y = iris_data['Label'].values

    # split
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=1/3, random_state=10)


    # declare model
    knn_model = KNeighborsClassifier()
    # fit model
    knn_model.fit(X_train,y_train)
    # evaluate
    accuracy = knn_model.score(X_test,y_test)

    print(accuracy)

    # one sample
    idx = 23
    test_sample_feat = [X_test[idx,:]]
    true_label = y_test[idx]
    pred_label = knn_model.predict(test_sample_feat)
    print('item{}: true:{},predict:{}'.format(idx,true_label,pred_label))

main()