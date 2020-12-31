import numpy as np
from MIoUData import MIoU_dataloader
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    # 计算每个类的正确预测的比例，求所有类的平均
    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        #Acc = np.nanmean(Acc)
        return Acc

    def Precision(self):
        precesion_arr = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        return precesion_arr

    def Recall(self):
        recall_arr = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return recall_arr

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
    
    def kappa(self):
        pe_rows = np.sum(self.confusion_matrix, axis=0)
        pe_cols = np.sum(self.confusion_matrix, axis=1)
        sum_total = sum(pe_cols)
        pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
        # np.trace求矩阵的迹，即对角线的和
        po = np.trace(self.confusion_matrix) / float(sum_total)
        return (po - pe) / (1 - pe)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

if __name__ == "__main__":
    miou = Evaluator(4)

    acc_0 =[]
    acc_1 =[]
    acc_2 =[]
    acc_3 =[]
    for index, (label, predict) in enumerate(MIoU_dataloader):
        label = label.cpu().numpy()
        predict = predict.cpu().numpy()
        miou.add_batch(label,predict)
        acc_0.append(miou.Pixel_Accuracy_Class()[0])
        acc_1.append(miou.Pixel_Accuracy_Class()[1])
        acc_2.append(miou.Pixel_Accuracy_Class()[2])
        acc_3.append(miou.Pixel_Accuracy_Class()[3])
    print('kappaVal is:{}'.format(miou.kappa()))
    print('all acc and miou are {},{}'.format(miou.Pixel_Accuracy(),miou.Mean_Intersection_over_Union()))
    print('precesion_arr:{}'.format(miou.Precision()))
    print('recall_arr:{}'.format(miou.Recall()))
    print('F1_score:{}'.format((2*miou.Precision()*miou.Recall())/(miou.Precision()+miou.Recall())))
    print('acc_0:{}'.format(np.nanmean(acc_0)))
    print('acc_1:{}'.format(np.nanmean(acc_1)))
    print('acc_2:{}'.format(np.nanmean(acc_2)))
    print('acc_3:{}'.format(np.nanmean(acc_3)))
