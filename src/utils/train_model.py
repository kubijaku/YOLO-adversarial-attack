from ultralytics import YOLO
import os


def train_model(model_path: str = '../model/yolov8n.pt', data_yaml: str = './car/data.yaml', device: str = 'mps',
                epochs: int = 30, batch: int = 10) -> dict:
    '''
    Function to train the YOLO model
    :param epochs:
    :param batch:
    :param model_path:
    :param data_yaml:
    :param device:
    :return: directory
    '''
    model = YOLO(model_path)

    # Training The Final Model
    final_model = model.train(data=data_yaml, epochs=epochs, batch=batch, optimizer='auto', device=device,
                              project=os.path.dirname(model_path))

    model_dir = final_model.save_dir
    return model_dir
