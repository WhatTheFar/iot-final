import argparse
import numpy as np

import recognize_video

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
    args = vars(ap.parse_args())

    last_recognized_label = None
    last_label_count = 0
    current_recognizing_label = None
    new_label_debounce_count = 0


    def on_recognition_match():
        pass


    def process_recognized_label(label=None):
        global last_recognized_label, last_label_count, current_recognizing_label, new_label_debounce_count

        if label != last_recognized_label:
            if label != current_recognizing_label:
                new_label_debounce_count = 0
                current_recognizing_label = label
            else:
                new_label_debounce_count += 1
                if new_label_debounce_count >= 5:
                    new_label_debounce_count = 0
                    last_recognized_label = label
                    last_label_count = 0
        else:
            last_label_count += 1
            if last_label_count >= 24:
                on_recognition_match()


    def face_recognition_callback(output):
        if len(output) == 0:
            process_recognized_label()
            return

        def sum_size(v):
            (start_x, start_y, end_x, end_y) = v["box"]
            return (end_x - start_x) * (end_y - start_y)

        size = map(sum_size, output)
        max_size_index = np.argmax(size)

        label = output[max_size_index]["name"]
        process_recognized_label(label)


    recognize_video.start_recognize(detector_path=args["detector"], embedding_model_path=args["embedding_model"],
                                    recognizer_path=args["recognizer"], label_encoder_path=args["le"],
                                    min_confidence=args["confidence"], debug=args["debug"],
                                    callback=face_recognition_callback)
