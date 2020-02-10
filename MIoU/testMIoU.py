import numpy as np
#from MIoUData import MIoU_dataloader
from sklearn.metrics import confusion_matrix

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
    '''
    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask].astype(int), minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist
    '''
    def _fast_hist(self, label_pred,label_true):
        hist = confusion_matrix(label_true, label_pred)
        return hist


    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            lp = lp.numpy()
            lt = lt.numpy()
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        #acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        #acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        #freq = self.hist.sum(axis=1) / self.hist.sum()
        #fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, mean_iu

if __name__ == "__main__":
    miou = IOUMetric(3)
    y_true = np.array([2, 0, 2, 2, 0, 1])
    y_pred = np.array([0., 0., 2., 2., 0., 2.])
    cm = miou._fast_hist(y_pred, y_true)
    print(cm)
    # batch=4，每4个算一个miou，最后求平均miou

    '''
    miouVal = 0
    accVal = 0
    for index, (predict, label) in enumerate(MIoU_dataloader):
        miou.add_batch(predict,label)
        accVal += miou.evaluate()[0]
        miouVal += miou.evaluate()[1]
        print('acc and miou are {},{}'.format(miou.evaluate()[0],miou.evaluate()[1]))
    print('all acc and miou are {},{}'.format(accVal/len(MIoU_dataloader),miouVal/len(MIoU_dataloader)))
    '''