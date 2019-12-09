echo "### Extracting embeddings"
python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7

echo
echo
echo
echo

echo "### Training model"
python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle