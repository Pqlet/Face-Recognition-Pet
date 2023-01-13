# Face-Recognition-Pet
- Constructing a full pipeline for face recognition using CelebA dataset with 1000 classes.
- Preprocessing bounding boxes of the CelebA dataset and training YOLOv5 detection model for the face detection task.
- Training a distinct model for face landamrks regression. Utilizing the estimated landmarks for face rotation and cropping.
- Fine-tuning CNN pre-trained on ImageNet to classify faces and evaluating the results:
  - Plotting cosine similarities between embeddings corresponding to the pictures of the same person and different people, calculating TPR@FPR metric.
  - Implementing and calculating TPR@FPR metric.
---
В репозитории:

1) Main_notebook.ipynb - Main notebook
2) Alignment_training.ipynb - Notebook with face landmarks alignment model training and alignment itself.
3) Prepare_data_for_YoloV5.ipynb - Notebook for bbox preparation for YOLOv5
4) utils_detal.py - Helper functions and classes
5) trained_models - Folder with models used in the main notebook
6) Best_accscore_1000 - Folder with the notebook with fine-tunning CNN to score 88.67 % accuracy on 1000 classes.

---
TODO: Robust landmarks rotation - https://github.com/PyImageSearch/imutils/blob/master/imutils/face_utils/facealigner.py

TODO: Retrain Alignment NN with more custom random augmentations 

TODO: ArcFace
