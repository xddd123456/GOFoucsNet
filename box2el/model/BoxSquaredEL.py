import numpy as np
import torch.nn as nn
import torch
from torch.nn.functional import relu
from boxes import Boxes
import os
import json

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class BoxSquaredEL(nn.Module):
    def __init__(self, device, embedding_dim, num_classes, num_roles, batch_size=256, pretrained_embeddings=None):
        super(BoxSquaredEL, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_roles = num_roles
        self.batch_size = batch_size
        self.margin = 0.1
        # self.individual_embeds = self.init_embeddings(self.num_individuals, embedding_dim)
        self.bumps = self.init_embeddings(self.num_classes, embedding_dim)
        self.relation_heads = self.init_embeddings(num_roles, embedding_dim * 2)
        self.relation_tails = self.init_embeddings(num_roles, embedding_dim * 2)

        if pretrained_embeddings is not None:
            self.class_embeds = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.class_embeds = nn.Embedding(num_classes, embedding_dim*2)

    def init_embeddings(self, num_embeddings, dim):
        if num_embeddings == 0:
            return None
        embeddings = nn.Embedding(num_embeddings, dim)
        return embeddings

    def get_boxes(self, embedding):
        embedding = embedding.weight
        return Boxes(embedding[:, :self.embedding_dim], torch.abs(embedding[:, self.embedding_dim:]))

    def get_class_boxes(self, nf_data, *indices):
        return (self.get_boxes(self.class_embeds(nf_data[:, i])) for i in indices)

    def get_relation_boxes(self, nf_data, *indices):
        boxes = []
        for i in indices:
            boxes.append(self.get_boxes(self.relation_heads(nf_data[:, i])))
            boxes.append(self.get_boxes(self.relation_tails(nf_data[:, i])))
        return tuple(boxes)

    def inclusion_loss(self, boxes1, boxes2):
        diffs = torch.abs(boxes1.centers - boxes2.centers)
        dist = torch.reshape(torch.linalg.norm(relu(diffs + boxes1.offsets - boxes2.offsets - self.margin), axis=1),
                             [-1, 1])
        return dist

    def nf1_loss(self, nf1_data):
        c_boxes, d_boxes = self.get_class_boxes(nf1_data, 0, 1)  # change
        return self.inclusion_loss(c_boxes, d_boxes)

    def nf2_loss(self, nf2_data):
        c_boxes, d_boxes, e_boxes = self.get_class_boxes(nf2_data, 0, 1, 2)
        intersection, lower, upper = c_boxes.intersect(d_boxes)
        return self.inclusion_loss(intersection, e_boxes) + torch.linalg.norm(relu(lower - upper), axis=1)

    def nf3_loss(self, nf3_data):
        c_boxes, d_boxes = self.get_class_boxes(nf3_data, 0, 2)
        c_bumps, d_bumps = self.bumps(nf3_data[:, 0]), self.bumps(nf3_data[:, 2])
        head_boxes, tail_boxes = self.get_relation_boxes(nf3_data, 1)

        dist1 = self.inclusion_loss(c_boxes.translate(d_bumps), head_boxes)
        dist2 = self.inclusion_loss(d_boxes.translate(c_bumps), tail_boxes)
        return (dist1 + dist2) / 2



    def get_data_batch(self, train_data, key):
        if len(train_data[key]) <= self.batch_size:
            return train_data[key].to(self.device)
        else:
            rand_index = np.random.choice(len(train_data[key]), size=self.batch_size, replace=False)
            return train_data[key][rand_index].to(self.device)


    def forward(self, train_data):
        loss = 0
        nf1_data = self.get_data_batch(train_data, 'nf1')
        loss += self.nf1_loss(nf1_data).square().mean()

        if len(train_data['nf2']) > 0:
            nf2_data = self.get_data_batch(train_data, 'nf2')
            loss += self.nf2_loss(nf2_data).square().mean()

        nf3_data = self.get_data_batch(train_data, 'nf3')
        loss += self.nf3_loss(nf3_data).square().mean()

        return loss
