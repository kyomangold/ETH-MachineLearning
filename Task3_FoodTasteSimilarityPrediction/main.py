import numpy as np
import pandas as pd
import random
import os
from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Input
import pickle

img_shape = [299,299]
input_shape = (299,299,3)

seed = 729

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

images_archive = 'food.zip'
img_directory_resized = './food_resized/'
train_triplets = 'train_triplets.txt'
test_triplets = 'test_triplets.txt'
submission_file = 'submission.txt'
features_file = 'features.pckl'


def generate_resized_dataset(zip_file,img_directory_resized,res_shape):

    zip_ref = ZipFile(zip_file, 'r')
    zip_ref.extractall()
    img_dir = zip_ref.filename[:-4]
    zip_ref.close()

    if not os.path.exists(img_directory_resized):
        os.makedirs(img_directory_resized)

    count = 0
    size = len(os.listdir(img_dir))

    for filename in os.listdir(img_dir):

        count += 1
        print('Processed images: {}/{}'.format(count,size),end="\r")

        if filename.endswith('.jpg'):
            img = load_img(img_dir+'/'+filename)
            img = img_to_array(img)
            img = tf.image.resize_with_pad(img,img_shape[0],img_shape[1],antialias=True)
            img = array_to_img(img)
            img.save(img_directory_resized+'/'+str(int(os.path.splitext(filename)[0]))+'.jpg')


def feature_extraction_cnn(input_shape):
    cnn = tf.keras.applications.InceptionResNetV2(pooling='avg',include_top=False)
    cnn.trainable = False

    x = x_in = Input(shape=input_shape)
    x = cnn(x)

    model = Model(inputs=x_in, outputs=x)

    return model

def img_input_preprocessing(directory_name, batch_size):
    num_images = 10000
    i = 0

    while True:
        batch = []

        while len(batch) < batch_size:
            img_name= directory_name + str(int(i)) + ".jpg"
            img = load_img(img_name)
            img = tf.keras.applications.inception_resnet_v2.preprocess_input(img_to_array(img))
            batch.append(img)
            i = (i + 1) % num_images

        batch = np.array(batch)
        labels = np.zeros(batch_size)

        try:
            yield batch, labels
        except StopIteration:
            return

def feature_extraction():
    feature_extraction = feature_extraction_cnn(input_shape)
    res_imgs = img_input_preprocessing(img_directory_resized,1)
    feature_vector = feature_extraction.predict(res_imgs,steps=10000)
    return feature_vector


def build_triplet_tensor(features, triplets_file, gen_labels=False):
    triplets_df = pd.read_csv(triplets_file, delim_whitespace=True, header=None, names=["A", "B", "C"])
    train_tensors = []
    labels = []
    num_triplets = len(triplets_df)

    for i in range(num_triplets):
        triplet = triplets_df.iloc[i]
        A, B, C = triplet['A'], triplet['B'], triplet['C']
        
        tensor_a = features[A]
        tensor_b = features[B]
        tensor_c = features[C]
        
        triplet_tensor = np.concatenate((tensor_a, tensor_b, tensor_c), axis=-1)
        if(gen_labels):
            reverse_triplet_tensor = np.concatenate((tensor_a, tensor_c, tensor_b), axis=-1)
            
            train_tensors.append(triplet_tensor)
            labels.append(1)
            train_tensors.append(reverse_triplet_tensor)
            labels.append(0)
        else:
            train_tensors.append(triplet_tensor)

    train_tensors = np.array(train_tensors)
    
    if(gen_labels):
        labels = np.array(labels)
        return train_tensors, labels
    else:
        return train_tensors


def main():
    print("START: Resizing dataset...")
    generate_resized_dataset(images_archive, img_directory_resized, img_shape)
    print("DONE: Resizing dataset!\n")
    
    print("START: Feature extraction...")
    if(os.path.exists(features_file)):
        with open(features_file, 'rb') as f:
            features = pickle.load(f)
            print("DONE: Feature extraction!\n")
    else:
        features = feature_extraction()
        with open(features_file, 'wb') as f:
            pickle.dump(features, f)
            print("DONE: Feature extraction!\n")

    print("START: Computing feature tensors...")
    train_tensors, labels = build_triplet_tensor(features, train_triplets, gen_labels=True)
    test_tensors = build_triplet_tensor(features, test_triplets, gen_labels=False)
    print("DONE: Computing feature tensors!\n")


    print("START: Creating Model...")
    x = x_in = Input(train_tensors.shape[1:])
    x = Activation('relu')(x)
    x = Dropout(0.7)(x)
    x = Dense(1088)(x)
    x = Activation('relu')(x)
    x = Dense(272)(x)
    x = Activation('relu')(x)
    x = Dense(68)(x)
    x = Activation('relu')(x)
    x = Dense(17)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_in, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    print("DONE: Creating Model!\n")
    
    print("START: Training...")
    model.fit(x = train_tensors, y = labels, epochs=20)
    print("DONE: Training!\n")

    
    print("START: Predicting...")
    y_test = model.predict(test_tensors)
    print("DONE: Predicting!\n")
    
    print("START: Generating submission file...")
    y_test_thresh = np.where(y_test < 0.5, 0, 1)
    np.savetxt(submission_file, y_test_thresh, fmt='%d')
    print("DONE: Generating submission file!\n")
    
    print("DONE: main.py")

if __name__ == '__main__':
    main()

