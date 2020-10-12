# this is the training part. we are going to take the name of image-folder as labels. detect their faces and train
# the model.  We require os library to get the directory path of the images and to get the labels for classification.
import os
import cv2
from PIL import Image
import numpy as np
import pickle  # used for serializing. recall serializable from php.
from PIL.Image import ANTIALIAS

base_dir = os.path.dirname(os.path.abspath('C:/ML_datasets/face_recognition/images'))  # return current working dir with backward slashes.
# or simply put: os.path.dirname(__file__) # it gives the current working directory name. with forward slashes.
image_dir = os.path.join(base_dir, 'images')
# print(base_dir, '=>', image_dir)

recognizer = cv2.face.LBPHFaceRecognizer_create() # this is face recognizer. we can model the data using ANN as well.
face_cascade = cv2.CascadeClassifier('C:/ML_datasets/face_recognition/cascades/haarcascade_frontalface_alt.xml')
y_labels = []
x_train = []

label_ids, current_id = dict(), 0
count = 0

for subdir, sub_subdir, all_files in os.walk(image_dir):
    for file in all_files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(subdir, file)
            label = os.path.basename(subdir).replace(' ', '-').lower()
            # print(label, '->', path)

            if label not in label_ids:  # each labels are given some consecutive numbers.
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]  # else:  you will however use the id's present for the labels.

            pil_image = Image.open(path).convert('L')  # L is used to convert to grayscale.
            size = (200, 200)
            final_image = pil_image.resize(size, ANTIALIAS)
            image_array = np.array(final_image, 'uint8')
            # print(image_array)

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.4, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y: y + h, x: x + w]
                cv2.imwrite(f'{count}.jpg', roi)
                count += 1
                x_train.append(roi)
                y_labels.append(id_)


with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)  # we are going to store the label ids in the pickle file. to help recognize images later.

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")  # this is our trained model.
