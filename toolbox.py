import numpy as np
import torch
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class PerturbationTool():
    def __init__(self, seed=0, epsilon=0.03137254901, num_steps=20, step_size=0.00784313725):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.seed = seed
        np.random.seed(seed)

    def random_noise(self, noise_shape):
        random_feature_noise = torch.FloatTensor(*noise_shape).uniform_(-100, 100).to(device)
        random_node_noise = torch
        return random_noise

    def min_min_attack(self, batch_graph, tag_lists, node_features, labels, base_model, optimizer, criterion, 
                                                                          batch_adj_noise, batch_tag_noise, num_nodes):
        
        perturb_batch_graph = Variable(torch.stack(batch_graph), requires_grad=True)
        perturb_batch_graph = Variable(torch.clamp(perturb_batch_graph, 0, 1), requires_grad=True)
        perturb_tag_lists = Variable(torch.stack(tag_lists), requires_grad=True)
        eta_adj = batch_adj_noise
        eta_tag = batch_tag_noise
        for _ in range(self.num_steps):
            base_model.zero_grad()
            
            if base_model.regression:
                pred, mae, loss = base_model(batch_graph, tag_lists, node_features, labels)
                #all_scores.append(pred.cpu().detach())  # for binary classification
            else:
                logits, loss, acc = base_model(batch_graph, tag_lists, node_features, labels)
                #all_scores.append(logits[:, 1].cpu().detach())  # for binary classification

            perturb_batch_graph.retain_grad()
            perturb_tag_lists.retain_grad()
            loss.backward()
            eta_adj = self.step_size * perturb_batch_graph.grad.data.sign() * (-1)
            eta_tag = self.step_size * perturb_tag_lists.grad.data.sign() * (-1)
            quit()

            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)

        return perturb_img, eta
