import argparse
import numpy as np
import microgear.client as microgear
import logging
from logging.handlers import TimedRotatingFileHandler

import recognize_video
import netpie_utils

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True,
                    help="path to OpenCV's deep learning face detector")
    ap.add_argument("-m", "--embedding-model", required=True,
                    help="path to OpenCV's deep learning face embedding model")
    ap.add_argument("-r", "--recognizer", required=True,
                    help="path to model trained to recognize faces")
    ap.add_argument("-l", "--le", required=True,
                    help="path to label encoder")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("--debug", default=False, action="store_true",
                    help="non-headless and show frames for debugging")

    # Raspberry Pi
    ap.add_argument("--pi", "--use-pi-camera", default=False, action="store_true",
                    help="use Pi camera instead of webcam")

    # netpie
    ap.add_argument("--key", required=True, help="NETPIE key")
    ap.add_argument("--secret", required=True, help="NETPIE secret")
    ap.add_argument("--appid", required=True, help="NETPIE appid")

    args = vars(ap.parse_args())

    # initialize logging configuration
    logging_handlers = []
    if args['debug'] is True:
        logging_level = logging.DEBUG
        logging_handlers.append(logging.StreamHandler())
    else:
        logging_level = logging.INFO
        logging_handlers.append(TimedRotatingFileHandler('./log/app.log', when="midnight", interval=1))

    logging.basicConfig(level=logging_level,
                        handlers=logging_handlers,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    last_recognized_label = None
    last_label_count = 0
    current_recognizing_label = None
    new_label_debounce_count = 0

    is_match_processed = False


    def on_recognition_match():
        global last_recognized_label, is_match_processed

        if is_match_processed is False:
            logging.info("Face recognized: {}".format(last_recognized_label))
            if last_recognized_label is not None and last_recognized_label != "unknown":
                logging.info("Turn off security command dispatched by {}".format(last_recognized_label))
                microgear.publish("/security/off", "true")
            is_match_processed = True


    def process_unrecognized_label():
        global last_recognized_label, last_label_count, current_recognizing_label, new_label_debounce_count, is_match_processed

        last_recognized_label = None
        last_label_count = 0
        current_recognizing_label = None
        new_label_debounce_count = 0
        is_match_processed = False


    def process_recognized_label(label, probability):
        global last_recognized_label, last_label_count, current_recognizing_label, new_label_debounce_count, is_match_processed

        if probability < 0.80:
            label = "unknown"

        if label != last_recognized_label:
            if label != current_recognizing_label:
                new_label_debounce_count = 0
                current_recognizing_label = label
            else:
                new_label_debounce_count += 1
                if new_label_debounce_count >= 3:
                    new_label_debounce_count = 0
                    last_recognized_label = label
                    last_label_count = 0
                    is_match_processed = False
        else:
            new_label_debounce_count = 0
            if last_label_count >= 10:
                on_recognition_match()
            else:
                last_label_count += 1


    def face_recognition_callback(output):
        if len(output) == 0:
            process_unrecognized_label()
            return

        def sum_size(v):
            (start_x, start_y, end_x, end_y) = v["box"]
            return (end_x - start_x) * (end_y - start_y)

        size = map(sum_size, output)
        max_size_index = np.argmax(size)

        label = output[max_size_index]["name"]
        proba = output[max_size_index]["probability"]
        process_recognized_label(label=label, probability=proba)


    netpie_utils.start_netpie(key=args["key"], secret=args["secret"], appid=args["appid"], debug=args["debug"])

    recognize_video.start_recognize(detector_path=args["detector"], embedding_model_path=args["embedding_model"],
                                    recognizer_path=args["recognizer"], label_encoder_path=args["le"],
                                    min_confidence=args["confidence"], debug=args["debug"], use_pi_camera=args['pi'],
                                    callback=face_recognition_callback)
