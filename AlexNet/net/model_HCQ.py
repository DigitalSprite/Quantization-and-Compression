from net.model import AlexNet
import numpy as np
import torch
import torch.nn.functional as F
import copy

class HCQ(AlexNet):

    def __init__(self, num_class, dropout=0.5):
        super(HCQ, self).__init__(num_class, dropout=dropout)

    def cal_group_dist(self, clusters, linkage_name, enlarge=False):

        def single_linkage(a, b, enlarge):
            if enlarge:
                a = np.exp(a)
                b = np.exp(b)
            return np.min(b) - np.max(a)
        def complete_linkage(a, b, enlarge):
            if enlarge:
                a = np.exp(a)
                b = np.exp(b)
            return np.max(b) - np.min(a)
        def avg_linkage(a, b, enlarge):
            if enlarge:
                a = np.exp(a)
                b = np.exp(b)
            dist = 0
            for i in a:
                for j in b:
                    dist += abs(j - i)
            return dist / (len(a) * len(b))
        # Use distance of clusters' mean value
        def designed_linkage(a, b, enlarge):
            if enlarge:
                a = np.exp(a)
                b = np.exp(b)
            return np.mean(b) - np.mean(a)
        linkage = {}
        linkage['single'] = single_linkage
        linkage['complete'] = complete_linkage
        linkage['avg'] = avg_linkage
        linkage['designed'] = designed_linkage
        dist_list = []
        for i in range(len(clusters) - 1):
            dist_list.append(linkage[linkage_name](clusters[i], clusters[i+1], enlarge))
        return np.array(dist_list)


    def hcq_initialization(self, layer_name, linkage_function, enlarge=False):
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
        total_num = len(index)
        while (len(weights) >= 256):
            weights_dist = self.cal_group_dist(weights, linkage_function, enlarge=enlarge)
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
            if 'weight' in layer:
                self.layers[layer].requires_grad=status

    def fine_tune(self, layer_name, code_book, retrain_epoch, learning_rate, train_loader,
                  valid_loader, test_loader, show_interval = 50):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        baseline_valid_acc = self.compute_acc(valid_loader)
        self.set_layers_status(False)
        self.layers[layer_name].requires_grad = True
        for epoch in range(retrain_epoch):
            for batch_idx, (images, labels) in enumerate(train_loader):
                logits, probas = self.forward(images)
                cost = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
                weight = self.layers[layer_name].data.numpy()
                for idx in range(np.max(code_book)+1):
                    centroid = np.sum((code_book == idx) * weight) / np.sum(code_book == idx)
                    # grad = np.sum(self.layers[layer_name].grad.numpy() * (code_book == idx)) / np.sum(code_book == idx)
                    # centroid = centroid - grad * learning_rate
                    weight[code_book == idx] = centroid
                self.layers[layer_name].data = torch.from_numpy(weight.astype(np.float32))
                if not (batch_idx+1) % show_interval:
                    with torch.no_grad():
                        test_acc = self.compute_acc(test_loader)
                        valid_acc = self.compute_acc(valid_loader)
                        print('Epoch: {:0>3d}/{:0>3d} | '
                              'Batch {:0>3d}/{:0>3d} | '
                              'Test Accuracy: {:.2f}% |'
                              'Validation Accuracy: {:.2f}%'.format(epoch+1, retrain_epoch,
                                                              batch_idx+1, len(train_loader),
                                                                    test_acc, valid_acc))
                        if valid_acc > baseline_valid_acc + 0.8:
                            break
        self.set_layers_status(True)
        # centroids = np.zeros(np.max(code_book) + 1, dtype=np.float32)
        # for idx in range(np.max(code_book) + 1):
        #     centroids[idx] = np.sum((code_book == idx) * self.layers[layer_name].data.numpy()) / \
        #                      np.sum((code_book == idx))
        return code_book

    def quantize_layer_under_acc_loss(self, layer_name, code_book, linkage_name, baseline_acc, valid_loader, enlarge=False):
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
        while valid_acc >= baseline_acc:
            original_code_book = copy.deepcopy(code_book)
            weight_dist = self.cal_group_dist(clusters, linkage_name=linkage_name, enlarge=enlarge)
            idx = int((np.where(weight_dist==np.min(weight_dist)))[0])
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
            valid_acc = self.compute_acc(valid_loader)
            if valid_acc < baseline_acc:
                self.layers[layer_name].data = torch.from_numpy(original_weights.astype(np.float32))
                code_book = original_code_book
                print('End quantization')
                break
            print('Clusters:{:>3d} | Validation Accuracy:{:.2f}% | '
                  'Accuracy change:{:.2f}%'.format(np.max(code_book) + 1, valid_acc.item(),
                                                   (valid_acc - original_acc).item()))
            cluster_num_list.append(np.max(code_book))
            acc_list.append(valid_acc.item())
        centroids = np.zeros(np.max(code_book)+1, dtype=np.float32)
        for idx in range(np.max(code_book)+1):
            centroids[idx] = np.sum((code_book == idx) * self.layers[layer_name].data.numpy()) / \
                             np.sum((code_book == idx))
        return code_book, centroids, cluster_num_list, acc_list

    def reduce(self, layer_name, code_book, centroids):
        reduce_weight = np.zeros(code_book.shape, dtype=np.float32)
        for idx in range(len(centroids)):
            reduce_weight[code_book == idx] = centroids[idx]
        self.layers[layer_name].data = torch.from_numpy(reduce_weight.astype(np.float32))

    # def get_centroids(self):
    #
