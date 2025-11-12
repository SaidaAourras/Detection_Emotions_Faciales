import cv2
import tensorflow as tf
import numpy as np



def detect_and_predict(img , model):
    model = tf.keras.models.load_model(model)
    emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

    # img = cv2.imread(img)

    gray_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

    # read the file xml
    facecscade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # detecte the emotion
    faces = facecscade.detectMultiScale(gray_img , scaleFactor=1.1 , minNeighbors=5) # array of [x y w h] ex: [[ 5  4 38 38]]
    emotions = []
    scores = []
    for (x,y,w,h) in faces:
        cv2.rectangle(img , (x,y) , (x+w,y+h) , (0, 160, 0),2)
        # extraire la region
        img_ = gray_img[x:x+w , y:y+h]
        
        if img_.shape[0] == 0:
            continue
        # redimentionne a la taille (48,48)
        img_ = cv2.resize(img_, (48,48))

        img_ = np.reshape(img_, (1,48,48,1))

        pred = model.predict(img_)
        emotion = emotion_labels[np.argmax(pred)]
        score = np.max(pred)
        
        emotions.append(emotion)
        scores.append(score)
        
        # label
        cv2.putText(img , f'{emotion}\n {score*100:.2f}%' , (x,y-5) , cv2.FONT_HERSHEY_COMPLEX , 0.3 , (0, 160, 0) , 1)
        
    return img , emotions , scores


# img , emotions , scores = detect_and_predict('./images/frt.jpg')
# print(emotions)
# print(scores[0])
# cv2.imshow("emotions" , img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
    
