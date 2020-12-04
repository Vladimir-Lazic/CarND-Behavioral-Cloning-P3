import cv2
import csv
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D

lines = []
with open('./data/driving_log.csv') as cvsfile:
    reader = csv.reader(cvsfile)
    for line in reader:
        lines.append(line)

images = []
steering_angles = []

for line in lines[1:]:
    image_center_file = line[0]
    image_left_file = line[1]
    image_right_file = line[2]

    image_center_path = image_center_file.split('/')[-1]
    image_center_path = './data/IMG/' + image_center_path

    image_left_path = image_left_file.split('/')[-1]
    image_left_path = './data/IMG/' + image_left_path

    image_right_path = image_right_file.split('/')[-1]
    image_right_path = './data/IMG/' + image_right_path

    image_center = cv2.imread(image_center_path)
    image_left = cv2.imread(image_left_path)
    image_right = cv2.imread(image_right_path)

    images.append(image_center)
    images.append(image_left)
    images.append(image_right)

    # create adjusted steering measurements for the side camera images
    steering_center = float(line[3])

    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    steering_angles.append(steering_center)
    steering_angles.append(steering_left)
    steering_angles.append(steering_right)


X_train = np.array(images)
y_train = np.array(steering_angles)

nvidia_model = Sequential([
    Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)),
    Cropping2D(cropping=((70,20), (0,0))),
    Conv2D(24, 5, 2, input_shape = (66, 200, 3), activation = 'relu'),
    Conv2D(36, 5, 2, activation = 'relu'),
    Conv2D(48, 5, 2, activation = 'relu'),
    Conv2D(64, 3, activation = 'relu'),
    Conv2D(64, 3, activation = 'relu'),
    Flatten(),
    Dense(100),
    Dense(50),
    Dense(10),
    Dense(1)
])

nvidia_model.compile(loss='mse', optimizer='adam')
nvidia_model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

nvidia_model.save('./models/model.h5')
