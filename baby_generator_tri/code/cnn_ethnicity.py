import cv2
from tqdm import tqdm

from keras.layers import Conv2D, Dense, MaxPooling2D, Input, Flatten, Dropout, Activation, BatchNormalization, Add
from keras import Model

import pickle
import numpy as np

import argparse
from PIL import Image

def Model_init():
    inp = Input(shape=(200, 200, 3,))

    x = Conv2D(filters=16, strides=(2,2), kernel_size=(3,3))(inp)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=32, strides=(2,2), kernel_size=(3,3))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=64, strides=(2,2), kernel_size=(3,3))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=64, strides=(2,2), kernel_size=(3,3))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=64, strides=(2,2), kernel_size=(3,3))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)
    out = Dense(5, activation='softmax')(x)

    model = Model(inputs=[inp], outputs=[out])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # print(model.summary())

    return model

# one-hot encoding
def Encode(Y):
    Y_one_hot = np.zeros((len(Y), 5))
    for i,y in enumerate(Y):
        Y_one_hot[i, y] = 1

    Y = Y_one_hot
    return Y

def loadTrainTestData():
    Xtr, Xte, Ytr, Yte = pickle.load(open('train_test_dataset.pkl', 'rb'))
    # print(Ytr, Yte)
    return np.array(Xtr), np.array(Xte), np.int32(Ytr), np.int32(Yte)

def preprocess(path):
    img_arr = np.array(Image.open(path))
    resize_img = cv2.resize(img_arr, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
    return resize_img

def train(epochs=30):

    Xtr, Xte, Ytr, Yte = loadTrainTestData()
    Ytr_encoded, Yte_encoded = Encode(Ytr), Encode(Yte)

    model = Model_init()

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for e in range(epochs):
        history = model.fit(Xtr, Ytr_encoded, batch_size=256)
        model.save_weights('new' + '_' + str(e) + '.h5')
        print('training metrics', history.history)
        train_loss.append(history.history['loss'][0])
        train_acc.append(history.history['accuracy'][0])

        test_history = model.evaluate(x=Xte, y=Yte_encoded)
        test_loss.append(test_history[0])
        test_acc.append(test_history[1])

        print("EPOCH", e, 'test loss', test_history[0], 'test acc:', test_history[1])
        print('train_loss =', train_loss)
        print('train_acc =', train_acc)
        print('test_loss =', test_loss)
        print('test_acc =', test_acc)

# [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others
# (like Hispanic, Latino, Middle Eastern) (from UTKFace dataset: https://susanqq.github.io/UTKFace/)
def predict(img):
    model = Model_init()
    model.load_weights("CNN/1_20.h5")  # load pretrained weights
    img_p = preprocess(img)
    res = model.predict(np.expand_dims(img_p, axis=0))
#     print(res)
#     print(np.argmax(res))
    idx = np.argmax(res)
    print('This classifier is solely used for baby predictor morphing; no offense to anyone')
    print('The predicted race is {}'.format(race_arr_UTK[idx]))
    return np.argmax(res)

race_arr_UTK = ["White", "Black", "Asian", "Indian", "Others"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training or predicting a ethnicity classifier')
    parser = argparse.ArgumentParser(description=
    'This classifier is solely used for baby \
    predictor morphing; no offense to anyone'
    )
    parser.add_argument('--usage', type=str, required=True, help='train or predict')
    parser.add_argument('--epoch', type=int, help='specify if train')
    parser.add_argument('--img_dir', type=str, help='specify if predict')
    args = parser.parse_args()

    if args.usage == 'train':
        train(args.epoch)
    elif args.usage == 'predict':
        predict(args.img_dir)
