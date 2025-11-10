import cv2
import tensorflow as tf
import numpy as np
print(cv2.__version__)

model = tf.keras.models.load_model('CNN_model.keras')
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
# img = cv2.imread('../datasets/test/angry/im8.png')
img = cv2.imread('./images/im.png')

gray_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

# print(img.shape)
# read the file xml
facecscade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# detecte the emotion
faces = facecscade.detectMultiScale(gray_img , scaleFactor=1.1 , minNeighbors=5) # array of [x y w h] ex: [[ 5  4 38 38]]
print(faces)
for (x,y,w,h) in faces:
    cv2.rectangle(img , (x,y) , (x+w,y+h) , (0,128,0),2)
    # extraire la region
    img_ = gray_img[x:x+w , y:y+h]
    # print(img_.shape)
    if img_.shape[0] == 0:
         continue
    # redimentionne a la taille (48,48)
    img_ = cv2.resize(img_, (48,48))
    print(img_.shape)
    img_ = np.reshape(img_, (1,48,48,1))

    pred = model.predict(img_)
    emotion = emotion_labels[np.argmax(pred)]
    
       
    cv2.putText(img , emotion , (x,y) , cv2.FONT_HERSHEY_COMPLEX , 0.5 , (0,128,0) , 1)
    
print(img.shape)
cv2.imshow("emotions" , img)
cv2.waitKey(0)
cv2.destroyAllWindows()
    

# print(f'the numbere of faces is : {len(faces)}') 