# Face-Recognition-Pet

В репозитории:

1) Главный ноутбук с обучением модели на 500 классов, не дающими качественный результат наработками Triplt loss и ArcFace loss, работающим в силу обученности внутренних моделей пайплайн Face Recognition(classification).
2) Папка Best_accscore_1000 с ноутбуком обучения модели на CE на 1000 классов с accuracy > 0.85 на валидации.
3) Alignment_training.ipynb - Ноутбук с обучением модели нахождения лэндмарков на изображении лица.
4) Prepare_data_for_YoloV5.ipynb - Ноутбук подготовки разметки для Yolov5
5) utils.py - Функции и классы хелперы.
6) trained_models - Папка с моделями, использующимися в главном ноутбуке

---
TODO: Robust landmarks rotation

TODO: Retrain Alignment NN with more custom random augmentations 

TODO: ArcFace
