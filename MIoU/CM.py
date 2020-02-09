from sklearn.metrics import confusion_matrix
from PIL import Image
from MIoUData import MIoU_dataloader
import numpy as np

# 求MIoU之前，需要先求混淆矩阵
# 该函数只能一幅label和predict进行求解混淆矩阵，效率太低

def generate_ConfusionMatrix():
    for index, (label, predict) in enumerate(MIoU_dataloader):
        label_arr = np.array(label)
        predict_arr = np.array(predict)
        label_arr_fla = label_arr.flatten()
        predict_arr_fla = predict_arr.flatten()
        ConfusionMatrix = confusion_matrix(label_arr_fla, predict_arr_fla)
