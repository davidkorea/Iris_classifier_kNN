import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


DATA_FILE = './data_ai/fruit_data.csv'
FEATURE_COL = ['mass','width','height','color_score']
SPECIES_LABEL_DICT = {
    'apple':0,
    'mandarin':1,
    'orange':2,
    'lemon':3
}

def main():
    fruit_data = pd.read_csv(DATA_FILE)
    fruit_data['Label'] = fruit_data['fruit_name'].map(SPECIES_LABEL_DICT)

    X = fruit_data[FEATURE_COL].values
    y = fruit_data['Label'].values

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/5,random_state=20)

    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train,y_train)
    accuracy = knn_model.score(X_test, y_test)
    print('Accuracy: {:.2f}'.format(accuracy))

    idx = 2
    test_sample_feat = [X_test[idx,:]]
    true_label = y_test[idx]
    pred_label = knn_model.predict(test_sample_feat)
    print('sample{}, true:{},predict:{}'.format(idx,true_label,pred_label))


main()