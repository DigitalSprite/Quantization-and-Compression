from net.resnet18 import ResNet18
import numpy as np
import torch
import torch.nn.functional as F
import copy
import time
import math
import json
from collections import OrderedDict
import os

class HCQ(ResNet18):

    def __init__(self, num_classes):
        super(HCQ, self).__init__(num_classes)
        self.linkage = {}
        def single_linkage(a, b):
            return np.min(b) - np.max(a)
        def complete_linkage(a, b):
            return np.max(b) - np.min(a)
        def avg_linkage(a, b):
            dist = 0
            for i in a:
                for j in b:
                    dist += abs(j - i)
            return dist / (len(a) * len(b))
        # Use distance of clusters' mean value
        def designed_linkage(a, b):
            return np.mean(b) - np.mean(a)
        self.linkage['single'] = single_linkage
        self.linkage['complete'] = complete_linkage
        self.linkage['avg'] = avg_linkage
        self.linkage['designed'] = designed_linkage
        self.code_book = {}
    
    def cal_group_dist(self, clusters, linkage_name):
        dist_list = []
        for i in range(len(clusters) - 1):
            dist_list.append(self.linkage[linkage_name](clusters[i], clusters[i+1]))
        return np.array(dist_list)

    def hcq_initialization(self, layer_name, linkage_function):
        '''
        Initialize HCQ to 8 bits status
        :param layer_name: conv/fc_weight/bias_num
        :param linkage_function: How to compute distance between clusters
        :return:
        '''
        original_weights = self.layers[layer_name].data.numpy().reshape(-1)
        sorted_data = sorted([(idx, value) for idx, value in enumerate(original_weights)], key=lambda i:i[1])
        weights = [[i[1]] for i in sorted_data]
        index = [[i[0]] for i in sorted_data]
        del sorted_data
        total_num = len(index)
        while (len(weights) >= 256):
            weights_dist = self.cal_group_dist(weights, linkage_function)
            clustering_index = []
            jump = False
            for i in range(len(weights)-1):
                if jump:
                    jump = False
                    continue
                if i == 0:
                    if weights_dist[0] < weights_dist[1]:
                        clustering_index.append([0, 1])
                        jump = True
                elif i == len(weights) - 2:
                    if weights_dist[i] < weights_dist[i-1]:
                        clustering_index.append([i, i+1])
                        jump = True
                elif weights_dist[i] <= weights_dist[i-1] and weights_dist [i] <= weights_dist[i+1]:
                    clustering_index.append([i, i+1])
                    jump = True
            for idx in clustering_index:
                weights[idx[0]] = list(np.concatenate((weights[idx[0]], weights[idx[1]]), axis=0))
                # print(weights[idx[0]], weights[idx[1]])
                index[idx[0]] = list(np.concatenate((index[idx[0]], index[idx[1]]), axis=0))
            weights = np.delete(weights, [i[1] for i in clustering_index])
            index = np.delete(index, [i[1] for i in clustering_index])
        # Generate Code Book according to hcq
        code_book = np.zeros(total_num, dtype=np.uint8)
        changed_weights = np.zeros(total_num)
        for i in range(len(index)):
            code_book[index[i]] = i
        code_book = code_book.reshape(self.layers[layer_name].data.numpy().shape)
        for i in range(len(index)):
            changed_weights[index[i]] = np.mean(weights[i])
        changed_weights = changed_weights.reshape(self.layers[layer_name].data.numpy().shape)
        self.layers[layer_name].data = torch.from_numpy(changed_weights.astype(np.float32))
        return code_book, original_weights.reshape(self.layers[layer_name].data.numpy().shape)
        
    def set_layers_status(self, status=False):
        '''
        Fix all layers, which immune from gradients update
        :return:
        '''
        for layer in self.layers:
            self.layers[layer].requires_grad = status
    
    def fine_tune(self, layer_name, code_book, retrain_epoch, learning_rate, train_loader,
                  valid_loader, test_loader, show_interval=50, sample_rate=1):
        # optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        optimizer = torch.optim.SGD(self.parameters(), learning_rate, momentum=0.9, weight_decay=1e-5)
        baseline_valid_acc = self.compute_acc(valid_loader)
        self.layers[layer_name].requires_grad = True
        start_time = time.time()
        for epoch in range(retrain_epoch):
            for batch_idx, (images, labels) in enumerate(train_loader):
                if batch_idx > sample_rate * len(train_loader):
                    break
                logits, probas = self.forward(images)
                cost = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
                weight = self.layers[layer_name].data.numpy()
                for idx in range(np.max(code_book)+1):
                    centroid = np.sum((code_book == idx) * weight) / np.sum(code_book == idx)
                    weight[code_book == idx] = centroid
                self.layers[layer_name].data = torch.from_numpy(weight.astype(np.float32))
                if not (batch_idx+1) % show_interval:
                    with torch.no_grad():
                        test_acc = self.compute_acc(test_loader)
                        valid_acc = self.compute_acc(valid_loader)
                        print('Epoch: {:0>3d}/{:0>3d} | '
                              'Batch {:0>3d}/{:0>3d} | '
                              'Test Accuracy: {:.2f}% | '
                              'Validation Accuracy: {:.2f}% | '
                              'Time: {:.2f} mins'.format(epoch+1, retrain_epoch,
                                                         batch_idx+1, len(train_loader), test_acc, valid_acc, (time.time() - start_time) / 60))
                        if valid_acc > baseline_valid_acc + 0.5:
                            print()
                            break
        self.layers[layer_name].requires_grad = False
    
    def quantize_layer_under_acc_loss(self, layer_name, code_book, linkage_name, clusters_num_lower_bound,  baseline_acc, valid_loader):
        '''
        QUantize Layer with target accuracy loss
        :param layer_name:
        :param target_acc:
        :param train_loader:
        :param valid_loader:
        :param test_loader:
        :return:
        '''
        self.layers[layer_name].requires_grad = True
        valid_acc = original_acc = self.compute_acc(valid_loader)
        clusters = []
        cluster_num_list = []
        acc_list = []
        for idx in range(np.max(code_book)+1):
            clusters.append(self.layers[layer_name].data.numpy()[code_book == idx])
        start_time = time.time()
        while valid_acc >= baseline_acc and np.max(code_book)+1 > clusters_num_lower_bound:
            original_code_book = copy.deepcopy(code_book)
            weight_dist = self.cal_group_dist(clusters, linkage_name = linkage_name)
            idx = np.where(weight_dist == np.min(weight_dist))[0][0]
            clusters[idx] = list(np.concatenate((clusters[idx], clusters[idx+1]), axis=0))
            code_book[code_book == idx+1] = idx
            clusters = np.delete(clusters, idx+1)
            bias = 0
            # modify code book
            for idx in range(np.max(code_book)+1):
                if np.sum(code_book == idx) == 0:
                    bias += 1
                else:
                    code_book[code_book == idx] = idx - bias
            # update weights
            original_weights = self.layers[layer_name].data.numpy()
            update_weights = np.zeros(code_book.shape)
            for idx in range(np.max(code_book)+1):
                update_weights[code_book == idx] = np.mean(clusters[idx])
            self.layers[layer_name].data = torch.from_numpy(update_weights.astype(np.float32))
            num = np.max(code_book)+1
            if num >= 200 and  num % 20 == 0 or num >= 100 and num < 200 and num % 10 == 0 or  num < 100 and num >= 64 and num % 5 == 0 or num < 64 and num % 2 == 0 and num >= 16 or num < 16:
                valid_acc = self.compute_acc(valid_loader)
                if valid_acc < baseline_acc:
                    self.layers[layer_name].data = torch.from_numpy(original_weights.astype(np.float32))
                    code_book = original_code_book
                    print('End quantization')
                    break
                print('Clusters:{:>3d} | Validation Accuracy:{:.2f}% | '
                    'Accuracy change: {:.2f}% | Time: {:.2f} mins'.format(np.max(code_book) + 1, valid_acc.item(),
                                                    (valid_acc - original_acc).item(), (time.time() - start_time) / 60))
                cluster_num_list.append(np.max(code_book))
                acc_list.append(valid_acc.item())
        centroids = np.zeros(np.max(code_book)+1, dtype=np.float32)
        for idx in range(np.max(code_book)+1):
            centroids[idx] = np.sum((code_book == idx) * self.layers[layer_name].data.numpy()) / \
                             np.sum((code_book == idx))
        self.layers[layer_name].requires_grad = False
        return code_book, centroids, cluster_num_list, acc_list

    def reduce(self, layer_name, code_book, centroids):
        reduce_weight = np.zeros(code_book.shape, dtype=np.float32)
        for idx in range(len(centroids)):
            reduce_weight[code_book == idx] = centroids[idx]
        self.layers[layer_name].data = torch.from_numpy(reduce_weight.astype(np.float32))

    def set_quantization_sequence(self, type, linkage_function):
        print('############################## Set Quantization Sequence By {} ##############################'.format(type))
        score_list = []
        if type == 'l1 norm':
            for layer_name in self.layers:
                original_weights = self.layers[layer_name].data.numpy()
                code_book, weights = self.hcq_initialization(layer_name, linkage_function)
                quantized_weights = self.layers[layer_name].data.numpy()
                score = np.abs((np.abs(quantized_weights) - np.abs(original_weights))).sum() * len(quantized_weights.reshape(-1)) / math.log2(len(quantized_weights.reshape(-1)))
                score_list.append((layer_name, score))
                print('Layer Name: {} | Score: {:.2f}'.format(layer_name, score))
                self.code_book[layer_name] = code_book
        sequence = sorted(score_list, key=lambda x: x[1], reverse=True)
        self.sequence = [i[0] for i in sequence]
        print('############################## End Quantization Sequence ##############################'.format(type))
    
    def quantize_layers_by_sequence(self, type, load_path, save_path, acc_loss, linkage_function, train_loader, valid_loader, test_loader, sample_rate=0.2, reload=False, reload_code_book_file=None, sequence=None, reload_model_file=None):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.load_from_pretrained_model(load_path)
        valid_acc = self.compute_acc(valid_loader)
        test_acc = self.compute_acc(test_loader)

        if reload:
            # Load from existig records
            with open(reload_code_book_file, 'r') as f:
                self.code_book = json.loads(f.read())
                for layer_name in self.code_book:
                    self.code_book[layer_name] = np.array(self.code_book[layer_name])
                f.close()
            self.load_from_pretrained_model(reload_model_file)
            self.sequence = sequence
        else:
            # Set sequence and make force-grained hcq
            self.set_quantization_sequence(type, linkage_function)
        print('Orignal valid accuracy:{:.2f}% | Original Test accuracy:{:.2f}% | Minimal valid accuracy:{:.2f}%'.format(valid_acc, test_acc, valid_acc - acc_loss))
        for epoch in range(3, 11):
            # accuracy loss adjustment
            real_acc_loss = acc_loss * (1 - 0.5**epoch)
            # Bit width adjustment
            bit_width_limit = 2
            if epoch <= 1:
                bit_width_limit = 4
            elif epoch <= 3:
                bit_width_limit = 3
            print('\n############################## Quantization Epoch {} | Acc Loss: {} ##############################\n'.format(epoch, real_acc_loss))
            for layer_name in self.sequence:
                print('---------------------------------- Quantize Layer {} ----------------------------------'.format(layer_name))
                code_book = self.code_book[layer_name]
                # Weights Adjustment
                self.fine_tune(layer_name, code_book, 1, 1e-4, train_loader, valid_loader, test_loader, show_interval=20, sample_rate=sample_rate)
                cluster_num = np.max(code_book)+1
                code_book, centroids, clusters_num_list, acc_list = self.quantize_layer_under_acc_loss(layer_name, code_book, linkage_function, 2**bit_width_limit, valid_acc - real_acc_loss, valid_loader)
                if np.max(code_book)+1 < cluster_num:
                    # Fine Tuning
                    # If clusters number is changed, we then need fine tune the weights
                    self.fine_tune(layer_name, code_book, 1, 1e-4, train_loader, valid_loader, test_loader, show_interval=20, sample_rate=sample_rate)
                self.code_book[layer_name] = code_book
                self.save_model(os.path.join(save_path, layer_name+' model.pth'))
                new_dict = {}
                for name in self.code_book:
                    new_dict[name] = self.code_book[name].tolist()
                with open(os.path.join(save_path, layer_name + ' code book.json'), 'w') as f:
                    f.write(json.dumps(new_dict))
                    f.close()
