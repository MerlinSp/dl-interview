from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
import sys
import csv
import cv2
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

train_img_dir_path = sys.argv[1]
train_gt_path = sys.argv[2]
test_img_dir_path = sys.argv[3]


batch_size = 32
num_classes = 4
epochs = 1000000
data_augmentation = False
num_predictions = 20
save_dir = os.getcwd()
model_name = 'model_weights.hdf5'
not_train = True

class_list = ['upright', 'rotated_left', 'rotated_right', 'upside_down']

num_train = len(os.listdir(train_img_dir_path))
num_test = len(os.listdir(test_img_dir_path))
#Load training data
x_train = np.zeros((num_train, 64, 64, 3))
y_train = np.zeros((num_train, 4))



# The data, split between train and test sets:
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

i = 0
#add train imgs to array
with open(train_gt_path, 'r') as csvfile:
     reader = csv.reader(csvfile)
     for row in reader:
         if row[1] == 'label':
             continue
         img = row[0]
         truth = row[1]
         img_path = os.sep.join((train_img_dir_path, img))
         tmp_img = cv2.imread(img_path)
         tmp_img_array = np.array(tmp_img)
         x_train[i] = tmp_img_array
         y_train[i][class_list.index(truth)] = 1
         i += 1

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_train /= 255

if not data_augmentation:
    callbacks = [
        EarlyStopping(monitor='val_acc', patience=7, mode='max'),
        ModelCheckpoint(os.sep.join(
            (save_dir, 'model_weights_best.hdf5')),
            monitor='val_acc', save_best_only=True, mode='max',
            save_weights_only=True)
    ]
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks,
              validation_split=0.1,
              shuffle=True)

else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.1)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, mode='max'),
        ModelCheckpoint(os.sep.join(
            (save_dir, 'model_weights_best.hdf5')),
                        monitor='val_accuracy', save_best_only=True, mode='max',
                        save_weights_only=True)
    ]
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        callbacks=callbacks,
                        workers=4)

truth_dic = {}

with open('test.preds.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['fn', 'label'])
    for img in os.listdir(test_img_dir_path):
        img_path = os.sep.join((test_img_dir_path, img))
        tmp_img = cv2.imread(img_path)
        tmp_img_array = np.array(tmp_img)
        tmp_img_array = tmp_img_array.astype('float32')
        tmp_img_array /= 255
        tmp_img_array = tmp_img_array.reshape(1, 64, 64, 3)
        pred = model.predict(tmp_img_array)
        true_indx = np.argmax(pred)
        true_class = class_list[true_indx]
        writer.writerow([img, true_class])
        truth_dic[img] = true_class

os.makedirs('test_imgs_fixed')
for img in os.listdir(test_img_dir_path):
    img_path = os.sep.join((test_img_dir_path, img))
    tmp_img = cv2.imread(img_path)
    tmp_img_array = np.array(tmp_img)
    if truth_dic[img] == 'upright':
        cv2.imwrite(os.sep.join(('test_imgs_fixed', img)), tmp_img_array)
    elif truth_dic[img] == 'rotated_left':
        for i in range(3):
            tmp_img_array = np.rot90(tmp_img_array)
        cv2.imwrite(os.sep.join(('test_imgs_fixed', img)), tmp_img_array)
    elif truth_dic[img] == 'rotated_right':
        tmp_img_array = np.rot90(tmp_img_array)
        cv2.imwrite(os.sep.join(('test_imgs_fixed', img)), tmp_img_array)
    elif truth_dic[img] == 'upside_down':
        for i in range(2):
            tmp_img_array = np.rot90(tmp_img_array)
        cv2.imwrite(os.sep.join(('test_imgs_fixed', img)), tmp_img_array)