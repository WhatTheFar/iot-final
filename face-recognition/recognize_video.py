# USAGE
# python recognize_video.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os


def start_recognize(detector_path, embedding_model_path, recognizer_path, label_encoder_path, min_confidence=0.5,
                    debug=False, callback=None):
    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([detector_path, "deploy.prototxt"])
    modelPath = os.path.sep.join([detector_path,
                                  "res10_300x300_ssd_iter_140000.caffemodel"])
    detector_path = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(embedding_model_path)

    # load the actual face recognition model along with the label encoder
    recognizer_path = pickle.loads(open(recognizer_path, "rb").read())
    label_encoder = pickle.loads(open(label_encoder_path, "rb").read())

    # initialize the video stream, then allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # start the FPS throughput estimator
    fps = FPS().start()

    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        frame = vs.read()

        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector_path.setInput(imageBlob)
        detections = detector_path.forward()

        # all detected face with label and probability, to be passed as an output to callback
        output = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > min_confidence:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer_path.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = label_encoder.classes_[j]

                output.append({"name": name, "probability": proba, "box": (startX, startY, endX, endY)})

                if debug:
                    # draw the bounding box of the face along with the
                    # associated probability
                    text = "{}: {:.2f}%".format(name, proba * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # update the FPS counter
        fps.update()

        if debug:
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        if callback is not None:
            callback(output)

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


# if __name__ == '__main__':
#     # construct the argument parser and parse the arguments
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--detector", required=True,
#                     help="path to OpenCV's deep learning face detector")
#     ap.add_argument("-m", "--embedding-model", required=True,
#                     help="path to OpenCV's deep learning face embedding model")
#     ap.add_argument("-r", "--recognizer", required=True,
#                     help="path to model trained to recognize faces")
#     ap.add_argument("-l", "--le", required=True,
#                     help="path to label encoder")
#     ap.add_argument("-c", "--confidence", type=float, default=0.5,
#                     help="minimum probability to filter weak detections")
#     ap.add_argument("--debug", default=False, action="store_true",
#                     help="non-headless and show frames for debugging")
#     args = vars(ap.parse_args())
#
#     start_recognize(detector_path=args["detector"], embedding_model_path=args["embedding_model"],
#                     recognizer_path=args["recognizer"], label_encoder_path=args["le"],
#                     min_confidence=args["confidence"], debug=args["debug"])
