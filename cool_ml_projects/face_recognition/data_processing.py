import cv2
import pickle
face_cascade = cv2.CascadeClassifier('C:/ML_datasets/face_recognition/cascades/haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')
labels = {}
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}


record = cv2.VideoCapture(0) # 0 means take the input from the camera itself. but if we mention the directory that becomes the video source.

while True:
    flag, frame = record.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y: y + h, x: x + w]  # region of interest in the matrix.
        roi_color = frame[y: y + h, x: x + w]  # select only the faces we need.

        id_, conf = recognizer.predict(roi_gray)  # conf = confidence level
        if conf >= 55:
            print(labels[id_])
            print(id_)
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 0, 0)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        image_item = 'temp_face.png'
        cv2.imwrite(image_item, roi_color)
        print('image captured')

        color, stroke = (122, 255, 0), 1
        end_cord_x, end_cord_y = (x + w), (y + h)

        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        # cv2.ellipse(frame, (x, y), (end_cord_x, end_cord_y), 0, 0, 360, color, stroke)

    cv2.imshow('selfie', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
record.release()
cv2.destroyAllWindows()
