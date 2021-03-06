import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import VGG, make_layers
import time
from torchvision.models import resnet18

class VGG16(VGG):

    def __init__(self, num_classes=10, batch_norm=False, dropout=0.5):
        cfgs = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                  'M'],
        }
        super(VGG16, self).__init__(features=make_layers(cfgs['D'], batch_norm=batch_norm),
                                    num_classes=num_classes, init_weights=True)
        self.classifier[2] = nn.Dropout(dropout)
        self.classifier[5] = nn.Dropout(dropout)

        def get_conv_layers():
            conv_list = []
            for i in range(1, 3):
                for j in range(1, 3):
                    conv_list.append('conv_weight_{}_{}'.format(i, j))
            for i in range(3, 6):
                for j in range(1, 4):
                    conv_list.append('conv_weight_{}_{}'.format(i, j))
            return conv_list
        self.layers = dict()
        conv_idx = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
        fc_idx = [0, 3, 6]
        for idx, layer_name in enumerate(get_conv_layers()):
            self.layers[layer_name] = self.features[conv_idx[idx]].weight
        for idx, fc_id in enumerate(fc_idx):
            self.layers['fc_weight_{}'.format(idx + 1)] = self.classifier[fc_id].weight

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 7*7*512)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

    def load_from_pretrained_model(self, load_state_dict_path=''):
        self.load_state_dict(torch.load(load_state_dict_path))

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)

    def compute_acc(self, data_loader, batch_num=-1):
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):
            logits, probas = self.forward(features)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size()[0]
            correct_pred += (predicted_labels == targets).sum()
            if batch_num > 0 and batch_num == i - 1:
                break
        return correct_pred.float() / num_examples * 100

    def adjust_learning_rate(self, optimizer, epoch, lr):
        """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
        lr = lr * (0.5 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train_model(self, start_epoch, epochs, learning_rate, momentum, weight_decay,
                    train_loader, valid_loader, test_loader, save_path):
        optimizer = torch.optim.SGD(self.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
        start_time = time.time()
        best_test_acc = 0

        for epoch in range(start_epoch, epochs):
            self.adjust_learning_rate(optimizer, epoch, learning_rate)
            self.train()
            for batch_idx, (features, targets) in enumerate(train_loader):
                '''
                Forward probagation
                '''
                logits, probas = self.forward(features)
                cost = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
                if not batch_idx % 100:
                    print(f'Epoch: {epoch + 1:03d}/{epochs:03d} | '
                          f'Batch {batch_idx:03d}/{len(train_loader):03d} |'
                          f' Cost: {cost:.4f}')
            self.eval()
            with torch.set_grad_enabled(False):  # save memory during inference

                train_acc = self.compute_acc(train_loader)
                valid_acc = self.compute_acc(valid_loader)
                test_acc = self.compute_acc(test_loader)

                print(f'Epoch: {epoch + 1:03d}/{epochs:03d}\n'
                      f'Train ACC: {train_acc:.2f} | Validation ACC: {valid_acc:.2f} | Test ACC:{test_acc:.2f}')
                if test_acc > best_test_acc:
                    print('Attain best test accuracy. Save model in {}'.format(save_path))
                    self.save_model(save_path)
                    best_test_acc = test_acc

            elapsed = (time.time() - start_time) / 60
            print(f'Time elapsed: {elapsed:.2f} min')