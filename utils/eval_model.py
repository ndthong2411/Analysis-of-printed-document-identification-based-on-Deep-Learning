import torch
from tqdm import tqdm
from sklearn.metrics import f1_score,confusion_matrix
import numpy as np

def eval(model, device, have_prj, loader, metric_loss, miner, criterion, split):
    model.eval()
    print('Evaluating model on ' + split + ' data')

    ce_loss_sum = 0
    metric_loss_sum = 0
    correct = 0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            if have_prj:
                p, logits = model(images)
                pminer = miner(p, labels)
                p_mloss = metric_loss(p, labels, pminer)
                ce_loss = criterion(logits, labels)
            else:
                _, logits = model(images)
                p_mloss = torch.tensor([0.0])
                ce_loss = criterion(logits, labels)

            ce_loss_sum += ce_loss.item()
            metric_loss_sum += p_mloss.item()

            pred = logits.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

            predicted_labels.extend(pred.cpu().numpy().tolist())
            true_labels.extend(labels.cpu().numpy().tolist())
    classes = ('Type1', 'Type2','Type3','Type4','Type5','Type6',)
    loss_avg = ce_loss_sum / (i+1)
    metric_loss_avg = metric_loss_sum / (i+1)
    accuracy = correct / len(loader.dataset)

    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    cf_matrix_l = confusion_matrix(true_labels, predicted_labels)
    cm = cf_matrix_l.astype('float') / cf_matrix_l.sum(axis=1)[:, np.newaxis]
    return loss_avg, metric_loss_avg, accuracy, f1,cm
