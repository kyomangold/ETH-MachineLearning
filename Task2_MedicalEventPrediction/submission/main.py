import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer #fill-in missing values
from sklearn import preprocessing

# Reading in provided datasets
dataset_train = pd.read_csv('train_features.csv')
dataset_test = pd.read_csv('test_features.csv')
training_set = dataset_train.values
test_set = dataset_test.values
labels_train = pd.read_csv('train_labels.csv')
y_train = labels_train.values

# Handling NaN values for training set
pid = y_train[:,0]
imp_zero = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=0)
scaler = preprocessing.MinMaxScaler()
for i in range(len(pid)):
    x = training_set[training_set[:,0]==pid[i]]
    col_mean = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_mean, inds[1])
    training_set[12*i:12*(i+1), :] = x
training_set_filled = imp_zero.fit_transform(training_set)

X_train_scaled = scaler.fit_transform(training_set_filled)
X_train_features = X_train_scaled[:, 2:]

# Handling NaN values for test set
pid_test = pd.unique(dataset_test['pid'])
for i in range(len(pid_test)):
    x = test_set[test_set[:,0]==pid_test[i]]
    col_mean = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_mean, inds[1])
    test_set[12*i:12*(i+1), :] = x
test_set_filled = imp_zero.fit_transform(test_set)

X_test_scaled = scaler.fit_transform(test_set_filled)
X_test_features = X_test_scaled[:, 2:]

y_train = y_train[:,1:]

n_samples = X_train_features.shape[0]
n_features = X_train_features.shape[1]
n_test_samples = X_test_features.shape[0]

X_test = []

a = np.arange(0,n_test_samples,12)
for i in range(np.size(a)):
    X_test.append(X_test_features[a[i]:a[i]+12, :])
X_test = np.array(X_test)
print(X_test.shape)  # (12664, 12, 35)

X = []
y = []

a = np.arange(0,n_samples,12)
for i in range(np.size(a)):
    X.append(X_train_features[a[i]:a[i]+12, :])
    y.append(y_train[i,:])
    
X = np.array(X)
y = np.array(y)

X_train = []
y_train = []
X_val = []
y_val = []

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

# split up training and validation sets into 3 parts (will be re-conactenated later)
y_train1 = y_train[:,0:10]
y_train2 = y_train[:,10:11]
y_train3 = y_train[:,11:]

y_val1 = y_val[:,0:10]
y_val2 = y_val[:,10:11]
y_val3 = y_val[:,11:]


# Data Loader Parameters
BATCH_SIZE = 8
BUFFER_SIZE = 1000
# LSTM Parameters
EPOCHS = 1000
PATIENCE = 10

tf.random.set_seed(13)

train_set1 = tf.data.Dataset.from_tensor_slices((X_train, y_train1))
train_set1 = train_set1.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
val_set1 = tf.data.Dataset.from_tensor_slices((X_val, y_val1))
val_set1 = val_set1.batch(BATCH_SIZE)

train_set2 = tf.data.Dataset.from_tensor_slices((X_train, y_train2))
train_set2 = train_set2.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
val_set2 = tf.data.Dataset.from_tensor_slices((X_val, y_val2))
val_set2 = val_set2.batch(BATCH_SIZE)

train_set3 = tf.data.Dataset.from_tensor_slices((X_train, y_train3))
train_set3 = train_set3.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
val_set3 = tf.data.Dataset.from_tensor_slices((X_val, y_val3))
val_set3 = val_set3.batch(BATCH_SIZE)

 
# Sequentially grouping linear stacks of layers (LSTM) for each of the 3 datasets created above

# for GRU or SimpleRnn replace tf.keras.layers.LSTM by either tf.keras.layers.GRU or tf.keras.layers.SimpleRnn
# for different activation functions replace activation='relu' with e.g. activation='softmax'

model1 = tf.keras.models.Sequential()
model1.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=X_train.shape[-2:]))
model1.add(tf.keras.layers.LSTM(16, activation='relu'))
model1.add(tf.keras.layers.Dense(y_train1.shape[1]))
model1.compile(optimizer = 'adam', loss = 'mean_squared_error')
print(model1.summary())
#model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

model2 = tf.keras.models.Sequential()
model2.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=X_train.shape[-2:]))
model2.add(tf.keras.layers.LSTM(16, activation='relu'))
model2.add(tf.keras.layers.Dense(y_train2.shape[1]))
model2.compile(optimizer = 'adam', loss = 'mean_squared_error')
print(model2.summary())
#model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

model3 = tf.keras.models.Sequential()
model3.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=X_train.shape[-2:]))
model3.add(tf.keras.layers.LSTM(16, activation='relu'))
model3.add(tf.keras.layers.Dense(y_train3.shape[1]))
model3.compile(optimizer = 'adam', loss = 'mean_squared_error')
print(model3.summary())
#model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')



# EarlyStopping if validation loss does not further improve
early_stopping = EarlyStopping(monitor='val_loss', patience = PATIENCE, restore_best_weights=True)
history1 = model1.fit(train_set1,epochs=EPOCHS,validation_data=val_set1,callbacks=[early_stopping])

early_stopping = EarlyStopping(monitor='val_loss', patience = PATIENCE, restore_best_weights=True)
history2 = model2.fit(train_set2,epochs=EPOCHS,validation_data=val_set2,callbacks=[early_stopping])

early_stopping = EarlyStopping(monitor='val_loss', patience = PATIENCE, restore_best_weights=True)
history3 = model3.fit(train_set3,epochs=EPOCHS,validation_data=val_set3, callbacks=[early_stopping])



# Predicting for each of the 3 parts
pred1 = model1.predict(X_test)
df_pred1 = pd.DataFrame(pred1)
df_pred1.insert(loc=0, column='pid', value=pid_test)
df_pred1.columns = labels_train.columns.values[0:11]
print(df_pred1.head())
print("++++++++++++++++++++++++++++++++++++++++++++++++")

pred2 = model2.predict(X_test)
df_pred2 = pd.DataFrame(pred2)
df_pred2.columns = labels_train.columns.values[11:12]
print(df_pred2.head())
print("++++++++++++++++++++++++++++++++++++++++++++++++")

pred3 = model3.predict(X_test)
df_pred3 = pd.DataFrame(pred3)
df_pred3.columns = labels_train.columns.values[12:]
print(df_pred3.head())
print("++++++++++++++++++++++++++++++++++++++++++++++++")



# Concatenate the 3 different predictions into a single one
df_pred = pd.concat([df_pred1, df_pred2, df_pred3],axis=1)
print(df_pred.head())

df_pred.to_csv('prediction.csv', index=False, float_format='%.3f')
