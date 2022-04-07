import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
from DGCNN_embedding import DGCNN
from mlp_dropout import MLPClassifier, MLPRegression
from sklearn import metrics
from util import cmd_args, load_data, GNNGraph
import networkx as nx

option = 1

class Classifier(nn.Module):
    def __init__(self, regression=False):
        super(Classifier, self).__init__()
        self.regression = regression
        if cmd_args.gm == 'DGCNN':
            model = DGCNN
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        if cmd_args.gm == 'DGCNN':
            self.gnn = model(latent_dim=cmd_args.latent_dim,
                            output_dim=cmd_args.out_dim,
                            num_node_feats=cmd_args.feat_dim+cmd_args.attr_dim,
                            num_edge_feats=cmd_args.edge_feat_dim,
                            k=cmd_args.sortpooling_k, 
                            conv1d_activation=cmd_args.conv1d_activation)
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            if cmd_args.gm == 'DGCNN':
                out_dim = self.gnn.dense_dim
            else:
                out_dim = cmd_args.latent_dim
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class, with_dropout=cmd_args.dropout)
        if regression:
            self.mlp = MLPRegression(input_size=out_dim, hidden_size=cmd_args.hidden, with_dropout=cmd_args.dropout)

    def PrepareFeatureLabel(self, batch_graph):
        if self.regression:
            labels = torch.FloatTensor(len(batch_graph))
        else:
            labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        if cmd_args.edge_feat_dim > 0:
            edge_feat_flag = True
            concat_edge_feat = []
        else:
            edge_feat_flag = False

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag == True:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag == True:
                tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)
            if edge_feat_flag == True:
                if batch_graph[i].edge_features is not None:  # in case no edge in graph[i]
                    tmp = torch.from_numpy(batch_graph[i].edge_features).type('torch.FloatTensor')
                    concat_edge_feat.append(tmp)

        if node_tag_flag == True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, cmd_args.feat_dim)
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag == True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features
        
        if edge_feat_flag == True:
            edge_feat = torch.cat(concat_edge_feat, 0)

        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()
            if edge_feat_flag == True:
                edge_feat = edge_feat.cuda()

        if edge_feat_flag == True:
            return node_feat, edge_feat, labels
        return node_feat, labels

    def forward(self, batch_graph, tags, features, labels):
        gs = [nx.from_numpy_array(tensor.numpy()) for tensor in batch_graph]
        batch_graph = [GNNGraph(g, labels[idx], tags[idx], features) for idx, g in enumerate(gs)]
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return self.mlp(embed, labels)

    def output_features(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return embed, labels
        

def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]


        # Try modify batch graph data type
        batch_graph = [torch.from_numpy(nx.to_numpy_array(g_list[idx].g)) for idx in selected_idx]
        tag_lists = [g_list[idx].node_tags for idx in selected_idx]
        node_features = None
        # End edition

        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        if classifier.regression:
            pred, mae, loss = classifier(batch_graph, tag_lists, node_features, targets)
            all_scores.append(pred.cpu().detach())  # for binary classification
        else:
            logits, loss, acc = classifier(batch_graph, tag_lists, node_features, targets)
            all_scores.append(logits[:, 1].cpu().detach())  # for binary classification

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().detach().numpy()
        if classifier.regression:
            pbar.set_description('MSE_loss: %0.5f MAE_loss: %0.5f' % (loss, mae) )
            total_loss.append( np.array([loss, mae]) * len(selected_idx))
        else:
            pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )
            total_loss.append( np.array([loss, acc]) * len(selected_idx))


        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()
    
    # np.savetxt('test_scores.txt', all_scores)  # output test predictions
    
    if not classifier.regression and cmd_args.printAUC:
        all_targets = np.array(all_targets)
        fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        avg_loss = np.concatenate((avg_loss, [auc]))
    else:
        avg_loss = np.concatenate((avg_loss, [0.0]))
    
    return avg_loss


def old_main():
    print(cmd_args)
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    train_graphs, test_graphs = load_data()
    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
        cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
        print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
        if not cmd_args.printAUC:
            avg_loss[2] = 0.0
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

        classifier.eval()
        test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
        if not cmd_args.printAUC:
            test_loss[2] = 0.0
        print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2]))

    with open(cmd_args.data + '_acc_results.txt', 'a+') as f:
        f.write(str(test_loss[1]) + '\n')

    if cmd_args.printAUC:
        with open(cmd_args.data + '_auc_results.txt', 'a+') as f:
            f.write(str(test_loss[2]) + '\n')

    if cmd_args.extract_features:
        features, labels = classifier.output_features(train_graphs)
        labels = labels.type('torch.FloatTensor')
        np.savetxt('extracted_features_train.txt', torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')
        features, labels = classifier.output_features(test_graphs)
        labels = labels.type('torch.FloatTensor')
        np.savetxt('extracted_features_test.txt', torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')


def new_main():

    print(cmd_args)
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    clean_train_graphs, test_graphs = load_data()
    print('# train: %d, # test: %d' % (len(clean_train_graphs), len(test_graphs)))

    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in clean_train_graphs + test_graphs])
        cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
        print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

    base_model = Classifier()
    if cmd_args.mode == 'gpu':
        base_model = base_model.cuda()
    optimizer_theta = optim.Adam(base_model.parameters(), lr=cmd_args.learning_rate)
    ##################################


    # Training the EM noise



    ##################################
    criterion = torch.nn.CrossEntropyLoss()
    condition = True
    train_idx = 0
    noise_len = len(clean_train_graphs)
    def get_inc_shape(graph):
        x = graph.num_nodes
        return [x+1, x+1]

    adj_noise = [torch.zeros(get_inc_shape(clean_train_graphs[i])) for i in range(noise_len)]
    tag_noise = [-1 for i in range(noise_len)]


    while condition:
        train_idxes = list(range(len(clean_train_graphs)))
        random.shuffle(train_idxes)

        # Set loop batch size
        bsize=cmd_args.batch_size
        total_iters = (len(train_idxes) + (bsize - 1) * (optimizer_theta is None)) // bsize
        pbar = tqdm(range(total_iters), unit='batch')

        for pos in pbar:
            selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]
            # optimize theta for M steps
            if pos % 21 != 0:
                base_model.train()
                for param in base_model.parameters():
                    param.requires_grad = True
                
                # for j in range(0, 10):
                #     try:
                #         graph = next(data_iter)
                #     except:
                #         train_idx = 0
                #         data_iter = iter(clean_train_graphs)
                #         graph = next(data_iter)
                    
                    
                    # Update noise to graphs
                    for loop_idx in selected_idx:
                        if tag_noise[loop_idx] == -1:
                            tag_noise[loop_idx] = 0
                            g_list[loop_idx].append_node()

                    batch_graph = [torch.from_numpy(nx.to_numpy_array(g_list[idx].g))+adj_noise[train_idx] for idx in selected_idx]
                    tag_lists = [g_list[idx].node_tags.append(tag_noise[idx]) for idx in selected_idx]
                    node_features = None
                    labels = [g_list[idx].label for idx in selected_idx]


                    
                    base_model.zero_grad()
                    optimizer_theta.zero_grad()
                    logits = base_model(batch_graph, tag_lists, node_features)
                    loss = criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), 5.0)
                    optimizer_theta.step()

                    quit()
            
            else:
                # Perturbation over entire dataset
                idx = 0
                for param in base_model.parameters():
                    param.requires_grad = False
                for i, graph in tqdm(enumerate(clean_train_loader), total=len(clean_train_loader)):
                    batch_start_idx, batch_noise = idx, []
                    
                    # Update noise to images
                    batch_noise.append(noise[idx])
                    idx += 1
                    batch_noise = torch.stack(batch_noise).cuda()
                    
                    # Update sample-wise perturbation
                    base_model.eval()
                    images, labels = images.cuda(), labels.cuda()
                    perturb_img, eta = noise_generator.min_min_attack(images, labels, base_model, optimizer, criterion, 
                                                                      random_noise=batch_noise)
                    for i, delta in enumerate(eta):
                        noise[batch_start_idx+i] = delta.clone().detach().cpu()
                
            # Eval stop condition
            eval_idx, total, correct = 0, 0, 0
            for i, (images, labels) in enumerate(clean_train_loader):
                for i, _ in enumerate(images):
                    # Update noise to images
                    images[i] += noise[eval_idx]
                    eval_idx += 1
                images, labels = images.cuda(), labels.cuda()
                with torch.no_grad():
                    logits = base_model(images)
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = correct / total
            print('Accuracy %.2f' % (acc*100))
            if acc > 0.99:
                condition=False      



if __name__ == '__main__':
    if option == 0:
        old_main()
    elif option == 1:
        new_main()
