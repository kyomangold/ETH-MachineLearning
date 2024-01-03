import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA

pretrain_labels = pd.read_csv('pretrain_labels.csv')
pretrain_features = pd.read_csv('pretrain_features.csv')

train_labels = pd.read_csv('train_labels.csv')
train_features = pd.read_csv('train_features.csv')

test_features = pd.read_csv('test_features.csv')
test_prediction = np.empty((test_features.shape[0],2))
test_prediction[:,0] = test_features.iloc[:,0]

def main():

    pca =  PCA()
    pretrain_feature_pca = pca.fit_transform(pretrain_features.iloc[:,2:])
    train_feature_pca = pca.transform(train_features.iloc[:,2:])
    test_feature_pca = pca.transform(test_features.iloc[:,2:])
    
    pretrain_labels_vector = list(pretrain_labels.iloc[:,1])
    train_labels_vector = list(train_labels.iloc[:,1])

    # train model with pretrained dataset
    x_split_pretrain,x_split_prevalidation,y_split_pretrain,y_split_prevalidation =  train_test_split(pretrain_feature_pca,pretrain_labels_vector,random_state = 42)
    
    model = MLPRegressor(hidden_layer_sizes=(198,),activation='logistic',solver='lbfgs', random_state=7, verbose=1, warm_start=True,early_stopping=True)
    model.fit(x_split_pretrain,y_split_pretrain)

    # retrain model with training dataset
    x_split_train,x_split_validation,y_split_train,y_split_validation =  train_test_split(train_feature_pca,train_labels_vector,random_state = 42)

    train_prediction = np.empty((len(y_split_validation),1))
    model.fit(x_split_train,y_split_train)
    train_prediction = model.predict(x_split_validation)

    test_prediction[:,1] = model.predict(test_feature_pca)

    df = pd.DataFrame(test_prediction, columns = ['Id','y'])
    df.to_csv('output.csv',index=False)

if __name__ == '__main__':

    main()
