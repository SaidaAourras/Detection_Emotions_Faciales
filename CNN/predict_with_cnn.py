from tensorflow.keras.models import load_model
import numpy as np
import cv2

emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
model = load_model('CNN_model.keras')

model.summary()

img_path = '../datasets/test/angry/im4.png'
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (48, 48))
input_array = resized / 255.0
input_array = np.expand_dims(input_array, axis=0)  # batch dimension
input_array = np.expand_dims(input_array, axis=-1)

pred = model.predict(input_array)
for label, prob in zip(emotion_labels, pred[0]):
    print(f"{label}: {prob:.2f}")
predicted_class = emotion_labels[np.argmax(pred)]
print(f"Émotion prédite : {predicted_class}")
# print(input_array.shape)





