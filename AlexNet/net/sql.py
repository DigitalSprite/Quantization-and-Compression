import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from net.model import AlexNet

class SLQ(AlexNet):

    def __init__(self, num_class, dropout=0.5):
        super(SLQ, self).__init__(num_class=num_class, dropout=dropout)
    
    def slq_quantization(self, cluster_num, partition, retrain_epoch, train_loader, valid_loader, test_loader,
                         retrain=True, show_log=False):

        self.single_level_quantization(self.features[0], cluster_num, partition, retrain_epoch, train_loader,
                                       valid_loader,
                                       test_loader, retrain, show_log)
        self.single_level_quantization(self.features[3], cluster_num, partition, retrain_epoch, train_loader,
                                       valid_loader,
                                       test_loader, retrain, show_log)
        self.single_level_quantization(self.features[6], cluster_num, partition, retrain_epoch, train_loader,
                                       valid_loader,
                                       test_loader, retrain, show_log)
        self.single_level_quantization(self.features[8], cluster_num, partition, retrain_epoch, train_loader,
                                       valid_loader,
                                       test_loader, retrain, show_log)
        self.single_level_quantization(self.features[10], cluster_num, partition, retrain_epoch, train_loader,
                                       valid_loader,
                                       test_loader, retrain, show_log)
        self.single_level_quantization(self.classifier[1], cluster_num, partition, retrain_epoch, train_loader,
                                       valid_loader,
                                       test_loader, retrain, show_log)
        self.single_level_quantization(self.classifier[4], cluster_num, partition, retrain_epoch, train_loader,
                                       valid_loader,
                                       test_loader, retrain, show_log)
        self.single_level_quantization(self.classifier[6], cluster_num, partition, retrain_epoch, train_loader,
                                       valid_loader,
                                       test_loader, retrain, show_log)

        test_acc = self.compute_acc(test_loader)
        print(test_acc)
        return 0

    def single_level_quantization(self, layer, cluster_num, partition, retrain_epoch, train_loader, valid_loader,
                                  test_loader,
                                  retrain=True, show_log=False):
        original_weight = layer.weight.data.numpy()
        original_shape = original_weight.shape

        mask = np.ones(original_shape, dtype=int)
        rest_num = cluster_num
        quantizated_clusters = np.ones(cluster_num)

        interval = (np.max(original_weight) - np.min(original_weight)) / (cluster_num - 1)
        initial_centroids = np.array([np.min(original_weight) + i * interval for i in range(cluster_num)],
                                     dtype=np.float32).reshape(-1, 1)
        predictor = KMeans(n_clusters=cluster_num, init=initial_centroids)
        code_book = predictor.fit_predict(original_weight.reshape(-1, 1)).reshape(original_shape)
        for p in partition:
            if rest_num == 0:
                break

            centroid_all = []
            for id in range(cluster_num):
                centroid = np.sum(original_weight * (code_book == id)) / np.sum(code_book == id)
                centroid_all.append(centroid)

            quantiztion_loss_all = []
            original_acc = self.compute_acc(valid_loader)
            for id in range(cluster_num):
                if quantizated_clusters[id] == 0:
                    quantiztion_loss_all.append(0)
                    continue

                weight_copy = original_weight.copy()
                weight_copy[code_book == id] = centroid_all[id]
                layer.weight.data = torch.from_numpy(weight_copy.astype(np.float32))
                acc = self.compute_acc(valid_loader)
                quantiztion_loss = abs(original_acc - acc)
                quantiztion_loss_all.append(quantiztion_loss)
                layer.weight.data = torch.from_numpy(original_weight.astype(np.float32))

                # quantiztion_loss = np.sum(abs(original_weight[code_book == id] - centroid_all[id])) / np.sum(code_book == id)
                # quantiztion_loss_all.append(quantiztion_loss)
            quantiztion_loss_all = np.asarray(quantiztion_loss_all)
            # print("quantiztion_loss_all:", quantiztion_loss_all)
            quantiztion_loss_index = np.argsort(-quantiztion_loss_all)
            # print("quantiztion_loss_index:", quantiztion_loss_index)

            for id in range(p):
                original_weight[code_book == quantiztion_loss_index[id]] = centroid_all[quantiztion_loss_index[id]]
                mask[code_book == quantiztion_loss_index[id]] = 0
                quantizated_clusters[quantiztion_loss_index[id]] = 0

            layer.weight.data = torch.from_numpy(original_weight.astype(np.float32))
            if retrain:
                for epoch in range(retrain_epoch):
                    for batch_idx, (images, labels) in enumerate(train_loader):
                        logits, probas = self.forward(images)
                        cost = F.cross_entropy(logits, labels)
                        cost.backward()
                        weights = layer.weight.data.numpy()
                        grad = layer.weight.grad.numpy() * (mask == 1)
                        weights -= grad * 1e-4
                        layer.weight.data = torch.from_numpy(weights)
                        if not (batch_idx + 1) % 100 and show_log:
                            with torch.no_grad():
                                test_acc = self.compute_acc(test_loader)
                                print('Epoch: {:0>3d}/{:0>3d} | Batch {:0>3d}/{:0>3d} | '
                                      'Test Accuracy: {:.2f}%'.format(epoch + 1, retrain_epoch, batch_idx + 1,
                                                                      len(train_loader), test_acc))
                        if batch_idx >= 100:
                            break

            rest_num -= p

        with torch.no_grad():
            test_acc = self.compute_acc(test_loader)
            valid_acc = self.compute_acc(valid_loader)
            print('Clusters: {:0>3d} | Test Accuracy: {:.2f}% | Valid Accuracy:'
                  ' {:.2f}%'.format(cluster_num, test_acc, valid_acc))
        return valid_acc.item()