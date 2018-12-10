import cv2
import numpy as np
import keras
from keras.models import load_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

'''
Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
for this error the below 2 statements are included
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# CPU-GPU configuration
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

video_capture = cv2.VideoCapture(0)
model = load_model('my_model.hdf5')
model.get_config()

target = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
	ret, frame = video_capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1)

	# drawing rectangle around faces
	for(x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 5)
		face_crop = frame[y:y + h, x:x + w]
		face_crop = cv2.resize(face_crop, (48, 48))
		face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
		face_crop = face_crop.astype('float32') / 255
		face_crop = np.asarray(face_crop)
		face_crop = face_crop.reshape(1, face_crop.shape[0], face_crop.shape[1], 1)
		result = target[np.argmax(model.predict(face_crop))]
		cv2.putText(frame, result, (x, y), font, 1, (200, 0, 0), 3, cv2.LINE_AA)

	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_capture.release()
cv2.destroyAllWindows()
