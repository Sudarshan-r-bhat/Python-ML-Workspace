import pandas as pd
import numpy as np
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential, model_from_json
from keras.preprocessing.image import ImageDataGenerator


# define a model
model = Sequential()

# build the model
model.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# add tunning/ error correction factors
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# preprocess before fitting the model
train_gen = ImageDataGenerator(zoom_range=0.2, rotation_range=0.2, rescale=1./255, shear_range=0.2, horizontal_flip=True)
test_gen = ImageDataGenerator()

train_set = train_gen.flow_from_directory(directory='./datasets/cats_dogs/training_set',
                              target_size=(64, 64),
                              batch_size=32,
                              class_mode='binary')

test_set = test_gen.flow_from_directory(directory='./datasets/cats_dogs/test_set',
                              target_size=(64, 64),
                              batch_size=32,
                              class_mode='binary')
model.fit_generator(generator=train_set,
                    validation_data=test_set,
                    steps_per_epoch=8000,
                    epochs=1,
                    validation_steps=2000)

# save model and weights
saved_model = model.to_json()
with open('./models/cat_dog_model.json', 'w') as json_file:
    json_file.write(saved_model)
model.save_weights('./models/cat_dog_weights.h5')
json_file.close()
print('Saved the model to the disk   =======>>>>>>>>')

# to load model and weight

# json_file = open('./model/cat_dog_model.json', 'r')
# loaded_model = json_file.read()
# loaded_model = model_from_json(loaded_model)
# loaded_model.load_weights('./models/cat_dog_weights.h5')
# print('loaded model and weights from the disk ==========>>>>>>')


path = 'C:\\workstation\\PycharmProjects\\Machine Learning Workspace\\neural_networks\\CNN\\datasets\\cats_dogs\\training_set\\cats\cat.1.jpg'
from keras.preprocessing import image
img = image.load_img(path, target_size=(64, 64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
class_name = ['cat', 'dog']
pred = model.predict(img)
print(class_name[int(pred[0])])





