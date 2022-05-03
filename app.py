from flask import Flask, render_template, Response
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np


import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)

#loading weight fules for face_detector network and mask_classifier
prototxtPath = "checkpoints/deploy.prototxt"
weightsPath = "checkpoints/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("checkpoints/face_mask.h5")


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (250, 250),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # print(detections.shape)

    # initialize our list of faces, their corresponding locations and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.3:  # assuming minimum confidence to be 30%
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # verifying that the bounding boxes fall within the dimensions the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 180x180, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (180, 180))
            face = img_to_array(face)
            face = preprocess_input(face)

            # adding the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # minimum number of faces to start predictions
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        # print(faces)
        preds = maskNet.predict(faces, batch_size=32)

    # return a tuple of the face locations and the mask or No mask predictions
    return (locs, preds)


def generate_frames():
    while True:
        ## read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # getting coordinates and predictions for faces and if mask is present on them
            locs, preds = detect_and_predict_mask(frame, faceNet=faceNet, maskNet=maskNet)

            # manipulating each frame in the for loop
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                mask = pred
                mask = mask[0]
                # assigning labels according to predictions value (probability of wearing a mask)
                label = "Mask" if mask > 0.65 else "No Mask"

                # color coding the label: green for mask, red for No Mask
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability and label
                label = "{}: {:.2f}%".format(label, mask * 100)

                # display the label and bounding box rectangle on the output frame
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()


        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') # returns each frame everytime to the html page


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)