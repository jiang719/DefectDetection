import random
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc

from data.data_loader import DataLoader
from data.dictionary import Dictionary
from model.transfo_classifier import TransfoClassifier


def valid(model, valid_loader):
    model.eval()
    batch_size = 64
    pred, y = [], []
    for i in range(0, len(valid_loader.data[:100]), batch_size):
        network_inputs = valid_loader.get_transfo_input(i, i + batch_size)
        outputs = model(network_inputs['inputs'], network_inputs['tags'])
        outputs = torch.exp(outputs)
        pred += outputs[:, 1].tolist()
        y += network_inputs['labels'].tolist()
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    model.train()
    return roc_auc


def train(model, train_loader, valid_loader, batch_size, epoches):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-4)
    loss_fct = nn.NLLLoss()
    old_roc_auc = 0
    for epoch in range(epoches):
        print('epoch ' + str(epoch + 1))
        random.shuffle(train_loader.data)
        for i in range(0, len(train_loader.data), batch_size):
            network_inputs = train_loader.get_transfo_input(i, i + batch_size)
            outputs = model(network_inputs['inputs'], network_inputs['tags'])
            optimizer.zero_grad()
            loss = loss_fct(outputs.view(-1, 2), network_inputs['labels'].view(-1))
            loss.backward()
            optimizer.step()
            print(loss.item())

            if (i / batch_size) % 10 == 0 and i > 0:
                roc_auc = valid(model, valid_loader)
                print('valid roc: ' + str(roc_auc))
                if roc_auc > old_roc_auc:
                    old_roc_auc = roc_auc
                    print('better model saved')
                    torch.save(model, datapath + 'model/transfo_1.pt')


datapath = 'D:/data/defect-detection/'

dictionary = Dictionary(datapath + 'bpe_vocab.txt')
print('dictionary loaded')
train_loader = DataLoader(datapath + 'bpe_train.txt', dictionary)
valid_loader = DataLoader(datapath + 'bpe_valid.txt', dictionary)
print('data loader initialized')

model = TransfoClassifier(dictionary, embed_dim=256, hidden_dim=256,
                          head_num=4, layer_num=3, inner_dim=256)

train(model, train_loader, valid_loader, batch_size=4, epoches=5)
