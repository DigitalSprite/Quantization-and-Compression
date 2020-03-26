import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
import time

class ResNet18(ResNet):

    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        if num_classes == 10:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layers = dict()
        # self.layers['conv_weight_1'] = self.conv1.weight.data
        for i in range(1, 3):
            self.layers['block 1 Conv {} 1'.format(i)] = self.layer1[i-1].conv1.weight
            self.layers['block 1 Conv {} 2'.format(i)] = self.layer1[i-1].conv2.weight
        for i in range(1, 3):
            self.layers['block 2 Conv {} 1'.format(i)] = self.layer2[i-1].conv1.weight
            self.layers['block 2 Conv {} 2'.format(i)] = self.layer2[i-1].conv2.weight
        for i in range(1, 3):
            self.layers['block 3 Conv {} 1'.format(i)] = self.layer3[i-1].conv1.weight
            self.layers['block 3 Conv {} 2'.format(i)] = self.layer3[i-1].conv2.weight
        for i in range(1, 3):
            self.layers['block 4 Conv {} 1'.format(i)] = self.layer4[i-1].conv1.weight
            self.layers['block 4 Conv {} 2'.format(i)] = self.layer4[i-1].conv2.weight
        self.layers['fc'] = self.fc.weight


    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 512)
        logits = self.fc(x)
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), 7*7*512)
        # logits = self.classifier(x)
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
        best_test_acc = self.compute_acc(test_loader)

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