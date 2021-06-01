from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import *
from keras.models import *
from keras.layers import Input, Dense
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
#from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
from keras.utils.np_utils import to_categorical
import glob
import os.path
from keras.models import load_model
from multiprocessing import Process

import os
import time
from imutils.video import VideoStream
import imutils
import cv2
import tensorflow as tf

categorie =["intruder", "authorized"]

def image_loader():
	detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	vs = VideoStream(src = 0).start()
	total = 0
	while True:
		frame = vs.read()
		orig = frame.copy()
		frame = imutils.resize(frame, width = 400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rects = detector.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30,30))
		for (x, y, w, h) in rects:
			cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)

		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		p = os.path.sep.join(["live_feed", "{}.jpg".format(str(total))])
		cv2.imwrite(p, frame[y:y+h,x:x+w])
		total += 1
		time.sleep(0.5)
	cv2.destroyAllWindows()
	vs.stop()


def load_data(path):
    x_train = []
    y_train = []

    images = glob.glob(path+"/**/*")
    for photo in images:
        img = image.load_img(photo, target_size=(299, 299))
        tr_x = image.img_to_array(img)
        tr_x = preprocess_input(tr_x)
        label = (photo.split("\\"))[1]
        label_place = categorie.index(label)

        x_train.append(tr_x)
        y_train.append(label_place)
    
    return np.array(x_train), to_categorical(y_train)

#Add images in dataset folder.. Add random face images from internet for intruder with name of the file intruder{x}.jpg.. 
#{x} means a number starting from '1'.. for example: intruder1.jpg*/
#Similarly add authorized person's images in authorized folder. criteria to add image is same.. for example authorized1.jpg
X_train, Y_train = load_data("dataset")

print(type(Y_train))
print(Y_train.shape)    # 808,4
print(X_train.shape)    # 808,299,299,3

input = Input(shape=(299, 299, 3))
#print(X_train.shape)
#raise
if (os.path.isfile("my_model.h5")):
    print("Model exists")
    model = load_model("my_model.h5")
else:
    print("Model not present, beginning training")
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=input, input_shape=(299, 299, 3), pooling='avg', classes=1000)
    for l in base_model.layers:
        l.trainable = False

    t = base_model(input)
    o = Dense(len(categorie), activation='softmax')(t)
    model = Model(inputs=input, outputs=o)


    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    

    model.fit(X_train, Y_train,
                  batch_size=30,
                  epochs=4,
                  shuffle=True,
                  verbose=1
                  )

    model.save("my_model.h5")

print(model.summary())
def main_code():
    total = 0
    while True:
        k = os.path.sep.join(["live_feed", "{}.jpg".format(str(total))])
        img_path = k
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        
        print("predictions: (intruder | authorized)")
        #x = preds[0]
        x = preds
        print(x)
        #y = np.array2string(x)
        #print(y[1:11])
        #i = np.argmax(preds)
        #lb = labels.classes_[i]
        #print(lb)
        total += 1
        os.remove(k)

if __name__ == "__main__":
    Process(target = image_loader).start()
    time.sleep(3)
    Process(target = main_code).start()