import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from data.data_loader import DataLoader
from data.dictionary import Dictionary


def test(models, test_loader):
    print('models number: ' + str(len(models)))
    for model in models:
        model.eval()
    batch_size = 128
    pred, y = [], []
    print(len(test_loader.data))
    for i in range(0, len(test_loader.data), batch_size):
        network_inputs = test_loader.get_hatt_input(i, i + batch_size)

        outputs = torch.zeros(network_inputs['inputs'].size(0), 2).to('cuda')
        for model in models:
            tmp = model(network_inputs['inputs'], network_inputs['tags'])
            outputs += torch.exp(tmp)
        outputs /= len(models)

        pred += outputs[:, 1].tolist()
        y += network_inputs['labels'].tolist()
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


datapath = 'D:/data/defect-detection/'

dictionary = Dictionary(datapath + 'vocab.txt')
test_loader = DataLoader(datapath + 'bpe_test.txt', dictionary)

model1 = torch.load(datapath + 'model/hatt_1.pt')
model2 = torch.load(datapath + 'model/hatt_2.pt')

fpr, tpr, roc_auc = test([model2], test_loader)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig(datapath + 'auc_model_2.png')
