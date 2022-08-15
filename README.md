# Face-Recognition-Pet

В репозитории:

1) Главный ноутбук с обучением модели на 500 классов, не дающими качественный результат наработками Triplt loss и ArcFace loss, работающим в силу обученности внутренних моделей пайплайн Face Recognition(classification).
2) Alignment_training.ipynb - Ноутбук с обучением модели нахождения лэндмарков на изображении лица.
3) Prepare_data_for_YoloV5.ipynb - Ноутбук подготовки разметки для Yolov5
4) utils.py - Функции и классы хелперы.
5) trained_models - Папка с моделями, использующимися в главном ноутбуке
6) Папка Best_accscore_1000 с ноутбуком обучения модели на CE на 1000 классов с accuracy > 0.85 на валидации.

---
TODO: Robust landmarks rotation

TODO: Retrain Alignment NN with more custom random augmentations 

TODO: ArcFace
