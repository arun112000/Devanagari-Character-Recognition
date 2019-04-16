from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

img_rows = 32
img_cols = 32
batch_size = 128

# Import dataset
dataset = pd.read_csv('data.csv')
x_train = np.array(dataset.iloc[:, :-1])
y_train = np.array(dataset.iloc[:, -1])
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)
x_train = x_train.reshape(-1, img_rows, img_cols, 1)
x_val = x_val.reshape(-1, img_rows, img_cols, 1)
x_test = x_test.reshape(-1, img_rows, img_cols, 1)
y_train = pd.get_dummies(y_train)
y_val = pd.get_dummies(y_val)
y_test = pd.get_dummies(y_test)

# Construct model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', kernel_initializer='he_uniform',
                 input_shape=(32, 32, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation="relu", kernel_initializer='he_uniform'))
model.add(Dropout(0.5))
model.add(Dense(46, activation="softmax"))
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
datagen = ImageDataGenerator(rescale=1. / 255)
training_set = datagen.flow(x_train, y_train, batch_size=batch_size)
test_set = datagen.flow(x_val, y_val, batch_size=batch_size)
history = model.fit_generator(training_set,
                              steps_per_epoch=518,
                              epochs=20,
                              validation_data=test_set,
                              validation_steps=144)

# Save the model
model.save('devanagari.h5')

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

score = model.evaluate(x_test / 255, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
