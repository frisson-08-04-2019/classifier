import logging
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.models.densenet import densenet121
import matplotlib.pyplot as plt
import torchvision
import os
from shutil import copyfile
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image

from .transforms import normalize
from .utils import get_device

SUBSETS = ('train', 'validation', 'test')


class ImageFolderWithPath(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPath, self).__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple + (path,)


class ClassifierNet(nn.Module):
    def __init__(self, outputs, units, dropout):
        super().__init__()
        self.outputs = outputs
        if dropout is None:
            dropout = 0.
        self.dropout = dropout
        self.basenet = densenet121(pretrained=True)
        self.lin1 = nn.Linear(1024, units)
        self.lin2 = nn.Linear(units, units)
        self.lin3 = nn.Linear(units, outputs)

    def forward(self, x):
        x = self.basenet.features.forward(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1).view(x.size(0), -1)
        if self.dropout > 0:
            x = nn.Dropout(self.dropout)(x)
        x = F.relu(self.lin1(x), inplace=True)
        if self.dropout > 0:
            x = nn.Dropout(self.dropout)(x)
        x = F.relu(self.lin2(x), inplace=True)
        x = self.lin3(x)
        x = nn.Softmax(dim=1)(x)
        return x


class Classifier:
    def __init__(self,
                 data_path=None,
                 learning_rate=None,
                 momentum=None,
                 epochs=None,
                 dropout=None,
                 batch_size=None,
                 results_path=None,
                 units=None,
                 load_net_path=None):
        self.data_path = data_path
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.dropout = dropout
        self.batch_size = batch_size
        self.results_path = results_path
        if results_path is not None:
            self.net_path = os.path.join(results_path, 'net.p')
        else:
            self.net_path = None
        self.train_acc_history = {ds: [] for ds in SUBSETS}
        self.datasets = {}
        self.data_loaders = {}
        self.class_labels = None
        self.device = get_device()
        self.units = units

        if load_net_path is None:
            self.init_subset('train')
            self.outputs = len(self.datasets['train'].classes)
        else:
            self.load_checkpoint(path=load_net_path, params=True, net=False)

        self.net = ClassifierNet(outputs=self.outputs, units=self.units, dropout=dropout)
        self.using_gpu = 'cuda' in str(self.device)
        self.net.to(self.device)

        if load_net_path is not None:
            self.load_checkpoint(load_net_path)

        self.get_class_labels()

    def init_train(self):
        for ds in SUBSETS:
            self.init_subset(ds)

        self.optimizer = optim.SGD([{'params': self.net.lin1.parameters(), 'lr': self.learning_rate},
                                    {'params': self.net.lin2.parameters(), 'lr': self.learning_rate},
                                    {'params': self.net.lin3.parameters(), 'lr': self.learning_rate},
                                    {'params': self.net.basenet.parameters(), 'lr': self.learning_rate}],
                                   momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()

    def init_subset(self, ds):
        def make_weights_for_balanced_classes(images, nclasses):
            count = [0] * nclasses
            for item in images:
                count[item[1]] += 1
            weight_per_class = [0.] * nclasses
            N = float(sum(count))
            for i in range(nclasses):
                weight_per_class[i] = N / float(count[i])
            logging.debug('Training sampling weights per class: %s' % str(weight_per_class))
            weight = [0] * len(images)
            for idx, val in enumerate(images):
                weight[idx] = weight_per_class[val[1]]
            return weight

        if ds not in self.datasets.keys():
            self.datasets[ds] = ImageFolderWithPath(root=os.path.join(self.data_path, ds), transform=normalize)

            if ds == 'train':
                weights = make_weights_for_balanced_classes(self.datasets[ds].imgs, len(self.datasets[ds].classes))
                sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
                self.data_loaders[ds] = DataLoader(self.datasets[ds], batch_size=self.batch_size, sampler=sampler,
                                                   num_workers=1)
            else:
                self.data_loaders[ds] = DataLoader(self.datasets[ds], batch_size=self.batch_size, shuffle=True,
                                                   num_workers=1)

    def train(self):
        self.init_train()

        results = {'max_val_acc': 0,
                   'max_val_test_acc': 0,
                   'max_val_acc_ep': 0,
                   'max_test_acc': 0,
                   'max_test_acc_ep': 0,
                   'max_train_acc': 0,
                   'max_train_acc_ep': 0,
                   'min_train_loss': float('Inf'),
                   'min_train_loss_ep': 0}

        for epoch in range(self.epochs):
            self.net.train()
            start = time.time()
            loss_train = 0
            for batch_num, (inputs, target, _) in tqdm(enumerate(self.data_loaders['train']),
                                                       desc='epoch %d: train' % epoch,
                                                       total=len(self.datasets['train'].samples) // self.batch_size):
                if self.using_gpu:
                    inputs, target = inputs.cuda(), target.cuda()
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, target)
                loss_train += loss.item() / inputs.shape[0]
                loss.backward()
                self.optimizer.step()

            # _, acc_train_avg = self.test(ds='train')
            acc_train_avg = 0
            self.train_acc_history['train'].append(acc_train_avg)

            _, acc_val_avg = self.test(ds='validation')
            self.train_acc_history['validation'].append(acc_val_avg)

            _, acc_test_avg = self.test(ds='test')
            self.train_acc_history['test'].append(acc_test_avg)

            if results['max_val_acc'] < acc_val_avg:
                results['max_val_acc'] = acc_val_avg
                results['max_val_acc_ep'] = epoch
                results['max_val_test_acc'] = acc_test_avg
                self.save_checkpoint()

            if results['max_test_acc'] < acc_test_avg:
                results['max_test_acc'] = acc_test_avg
                results['max_test_acc_ep'] = epoch

            if results['max_train_acc'] < acc_train_avg:
                results['max_train_acc'] = acc_train_avg
                results['max_train_acc_ep'] = epoch

            loss_train = loss_train / len(self.data_loaders['train'])
            if results['min_train_loss'] > loss_train:
                results['min_train_loss'] = loss_train
                results['min_train_loss_ep'] = epoch

            logging.debug('Max val acc: %.2f%% (ep %d). Max val test acc: %.2f%%. ' \
                          'Max test acc: %.2f%% (ep %d). Max train acc: %.2f%% (ep %d). ' \
                          'Min train loss: %.5f (ep %d).' %
                          (results['max_val_acc'], results['max_val_acc_ep'], results['max_val_test_acc'],
                           results['max_test_acc'], results['max_test_acc_ep'], results['max_train_acc'],
                           results['max_train_acc_ep'], results['min_train_loss'], results['min_train_loss_ep']))

            logging.debug('Finished epoch %d in %ds. Average train loss: %.5f' %
                          (epoch, time.time() - start, loss_train))

            self.plot_training_acc()

        self.load_checkpoint()

    def plot_training_acc(self):
        f = plt.figure()
        [plt.plot(history, label=ds) for ds, history in self.train_acc_history.items()]
        plt.legend()
        f.savefig(os.path.join(self.results_path, 'training_acc.png'))
        plt.close(f)

    def plot_confusion_matrix(self, cm, ds, normalize=True, cmap=plt.cm.Blues):
        cm = cm.astype('int')

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=self.class_labels, yticklabels=self.class_labels,
               ylabel='True label',
               xlabel='Predicted label',
               title=ds)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        fig.savefig(os.path.join(self.results_path, 'confusion_matrix_%s.png' % ds))
        plt.close(fig)

    def test(self, ds):
        cm = np.zeros((self.outputs, self.outputs))
        self.init_subset(ds)
        self.net.eval()
        class_correct = list(0. for i in range(self.outputs))
        class_total = list(0. for i in range(self.outputs))
        with torch.no_grad():
            for batch_num, (inputs, target, _) in tqdm(enumerate(self.data_loaders[ds]), desc='testing %s' % ds,
                                                       total=len(self.datasets[ds].samples) // self.batch_size):
                if self.using_gpu:
                    inputs, target = inputs.cuda(), target.cuda()
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == target)
                for i in range(len(c)):
                    label = target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

                cm += confusion_matrix(target.cpu(), predicted.cpu(), labels=list(range(self.outputs)))

        cls_acc = tuple(100 * correct / total for correct, total in zip(class_correct, class_total))
        # avg_acc = 100 * sum(class_correct) / sum(class_total)
        avg_acc = sum(cls_acc) / len(cls_acc)
        logging.debug('%s - accuracy: %.2f%%' % (ds, avg_acc))
        logging.debug(('%s - class accuracies: ' % ds) + '/'.join(['%.2f%%' % acc for acc in cls_acc]))

        self.plot_confusion_matrix(cm, ds)

        return cls_acc, avg_acc

    def eval(self, ds='eval'):
        self.init_subset(ds)
        self.net.eval()
        with torch.no_grad():
            for batch_num, (inputs, target, paths) in tqdm(enumerate(self.data_loaders[ds]), desc='evaluating %s' % ds,
                                                           total=len(self.datasets[ds].samples) // self.batch_size):
                if self.using_gpu:
                    inputs, target = inputs.cuda(), target.cuda()
                outputs = self.net(inputs)
                certainty, predicted = torch.max(outputs, 1)

                for predicted_img, path_img, certainty_img in zip(predicted, paths, certainty):
                    name = '%.6f_' % certainty_img + os.path.basename(path_img)
                    path_to = os.path.join(self.results_path, 'eval', self.class_labels[int(predicted_img)], name)
                    os.makedirs(os.path.dirname(path_to), exist_ok=True)
                    copyfile(path_img, path_to)

    def get_misclassified(self, ds='train'):
        self.init_subset(ds)
        self.net.eval()
        with torch.no_grad():
            for batch_num, (inputs, target, paths) in tqdm(enumerate(self.data_loaders[ds]),
                                                           desc='computing misclassified %s' % ds,
                                                           total=len(self.datasets[ds].samples) // self.batch_size):
                if self.using_gpu:
                    inputs, target = inputs.cuda(), target.cuda()
                outputs = self.net(inputs)
                certainty, predicted = torch.max(outputs, 1)

                for predicted_img, target_img, path_img, certainty_img in zip(predicted, target, paths, certainty):
                    label_correct = 'correct' if predicted_img == target_img else 'incorrect'
                    label_target = self.class_labels[int(target_img)]
                    label_predicted = self.class_labels[int(predicted_img)]
                    name = '%s_%.3f_' % (label_target, certainty_img) + os.path.basename(path_img)
                    path_to = os.path.join(self.results_path, 'eval', label_correct, label_predicted, name)
                    os.makedirs(os.path.dirname(path_to), exist_ok=True)
                    copyfile(path_img, path_to)

    def eval_single_img(self, path):
        t0 = time.time()
        image = Image.open(path)
        image = normalize(image).float()
        image = Variable(image)
        image = image.unsqueeze(0)
        self.net.eval()
        with torch.no_grad():
            if self.using_gpu:
                image = image.cuda()
            output = self.net(image)
            certainty, predicted = torch.max(output, 1)
            certainty = float(certainty)
            predicted = int(predicted)
            label = self.class_labels[predicted]
            scores = {l: val for l, val in zip(self.class_labels, output.tolist()[0])}

        return predicted, certainty, label, scores, time.time() - t0

    def save_checkpoint(self):
        logging.info('Saving model to %s' % self.net_path)
        torch.save({'state': self.net.state_dict(),
                    'classes': self.outputs},
                   self.net_path)

    def load_checkpoint(self, path=None, params=False, net=True):
        if path is None:
            path = self.net_path

        log_str = 'Loading model from %s' % path
        if params:
            log_str += ' - params'
        if net:
            log_str += ' - net'
        logging.info(log_str)

        checkpoint = torch.load(path, map_location=self.device)
        if params:
            self.outputs = checkpoint['classes']
            self.units = len(checkpoint['state']['lin1.bias'])
        if net:
            self.net.load_state_dict(checkpoint["state"])

    def get_class_labels(self):
        def get_num_labels():
            return ['%d' % i for i in range(self.outputs)]

        if self.data_path is None:
            self.class_labels = get_num_labels()
        else:
            for ds in SUBSETS + ('eval',):
                try:
                    self.init_subset(ds)
                    self.class_labels = self.datasets[ds].classes
                    break
                except FileNotFoundError:
                    self.class_labels = get_num_labels()

        logging.debug('Class labels: %s' % self.class_labels)
