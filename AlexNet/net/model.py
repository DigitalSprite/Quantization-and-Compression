import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

class AlexNet(nn.Module):
    def __init__(self, num_class, dropout=0.5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_class)
        )
        self.layers = dict()
        self.clusters = dict()
        self.code_books = dict()
        for idx, param in enumerate(self.parameters()):
            if idx in [0, 2, 4, 6, 8]:
                self.layers['conv_weight_{}'.format( idx // 2 + 1)] = param
            if idx in [1, 3, 5, 7, 9]:
                self.layers['conv_bias_{}'.format( idx // 2 + 1)] = param
            if idx in [10, 12, 14]:
                self.layers['fc_weight_{}'.format((idx - 10) // 2 + 1)] = param
            if idx in [11, 13, 15]:
                self.layers['fc_bias_{}'.format((idx - 10) // 2 + 1)] = param
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        logits =self.classifier(x)
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
            if batch_num > 0 and batch_num == i-1:
                break
        return correct_pred.float()/num_examples * 100
    
    def train_model(self, epochs, learning_rate, train_loader, valid_loader, test_loader, save_path):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        start_time = time.time()
        best_test_acc = 0

        for epoch in range(epochs):
            self.train()
            for batch_idx, (features, targets) in enumerate(train_loader):
                ### FORWARD AND BACK PROP
                logits, probas = self.forward(features)
                cost = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
                if not batch_idx % 100:
                    print (f'Epoch: {epoch+1:03d}/{epochs:03d} | '
                   f'Batch {batch_idx:03d}/{len(train_loader):03d} |' 
                   f' Cost: {cost:.4f}')
            self.eval()
            with torch.set_grad_enabled(False): # save memory during inference
                
                train_acc = self.compute_acc(train_loader)
                valid_acc = self.compute_acc(valid_loader)
                test_acc = self.compute_acc(test_loader)
                
                print(f'Epoch: {epoch+1:03d}/{epochs:03d}\n'
                    f'Train ACC: {train_acc:.2f} | Validation ACC: {valid_acc:.2f} | Test ACC:{test_acc:.2f}')
                if test_acc > best_test_acc:
                    print('Attain best test accuracy. Save model in {}'.format(save_path))
                    self.save_model(save_path)
                    best_test_acc = test_acc
                
            elapsed = (time.time() - start_time)/60
            print(f'Time elapsed: {elapsed:.2f} min') 

    def prune_weight(self, layer, prune_ratio):
        self.eval()
        original_weight = layer.weight.detach().numpy()
        original_shape = list(original_weight.shape)
        total_num = 1
        for i in range(len(original_shape)):
            total_num *= original_shape[i]
        threshold_weight = sorted(np.abs(original_weight.reshape(-1)))[int(total_num * prune_ratio)]
        mask = np.abs(original_weight) > threshold_weight
        layer.weight.data = torch.from_numpy(mask * original_weight)
        return mask
    
    #################################
    # how to prune a selected layer #
    #################################
    def prune(self, layer, prune_ratio, retrain_epoch, train_loader, valid_loader, test_loader):
        for epoch in range(retrain_epoch):
            print('############################# Retraining Epoch:[{}] #############################'.format(epoch))
            mask = self.prune_weight(layer, prune_ratio)
            self.train()
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
            for batch_idx, (images, labels) in enumerate(train_loader):
                logits, probas = self.forward(images)
                cost = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                cost.backward()
                layer.weight.grad *= mask
                optimizer.step()
                if not batch_idx % 50:
                    with torch.no_grad():
                        test_acc = self.compute_acc(test_loader)
                        print('Epoch: {:0>3d}/{:0>3d} | Batch {:0>3d}/{:0>3d} | Test Accuracy: {:.2f}%'.format(epoch+1, retrain_epoch, batch_idx, len(train_loader), test_acc))

    ########################################
    # Use linear intialization to quantize #
    ########################################
    def k_means_quantization(self, layer, cluster_num, retrain_epoch, train_loader, valid_loader, test_loader,
                             retrain=True, show_log=False):
        original_weight = layer.weight.data.numpy()
        original_shape = original_weight.shape
        interval = (np.max(original_weight) - np.min(original_weight)) / (cluster_num - 1)
        initial_centroids = np.array([np.min(original_weight) + i * interval for i in range(cluster_num)], dtype=np.float32).reshape(-1,1)
        predictor = KMeans(n_clusters=cluster_num, init=initial_centroids)
        code_book = predictor.fit_predict(original_weight.reshape(-1, 1)).reshape(original_shape)
        for id in range(cluster_num):
            centroid = np.sum(original_weight * (code_book == id)) / np.sum(code_book == id)
            original_weight[code_book == id] = centroid
        layer.weight.data = torch.from_numpy(original_weight.astype(np.float32))
        if retrain:
            for epoch in range(retrain_epoch):
                for batch_idx, (images, labels) in enumerate(train_loader):
                    logits, probas = self.forward(images)
                    cost = F.cross_entropy(logits, labels)
                    cost.backward()
                    weights = layer.weight.data.numpy()
                    for idx in range(cluster_num):
                        centroid = np.sum((code_book == idx) * weights) / np.sum(code_book == idx)
                        grad = np.sum(layer.weight.grad.numpy() * (code_book == idx)) / np.sum(code_book == idx)
                        weights[code_book == idx] = centroid - grad * 1e-4
                    layer.weight.data = torch.from_numpy(weights)
                    if not (batch_idx+1) % 100 and show_log:
                        with torch.no_grad():
                            test_acc = self.compute_acc(test_loader)
                            print('Epoch: {:0>3d}/{:0>3d} | Batch {:0>3d}/{:0>3d} | '
                                  'Test Accuracy: {:.2f}%'.format(epoch+1, retrain_epoch,
                                                                  batch_idx+1, len(train_loader), test_acc))
                    if batch_idx >= 100:
                        break
        with torch.no_grad():
            test_acc = self.compute_acc(test_loader)
            valid_acc = self.compute_acc(valid_loader)
            print('Clusters: {:0>3d} | Test Accuracy: {:.2f}% | Valid Accuracy:'
                  ' {:.2f}%'.format(cluster_num, test_acc, valid_acc))
        return valid_acc.item()
