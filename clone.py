import cv2
import csv
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense

lines = []
with open('./simulator_data/driving_log.csv') as cvsfile:
    reader = csv.reader(cvsfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './simulator_data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurements.append(float(line[3]))


X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)

model.save('./models/model.h5')
