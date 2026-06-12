import torch
import torch.nn as nn
from parse import args
from client import FedRecClient
import numpy as np
from parse import GENERATORCONFIGS
from time import time
#import wandb
import random
import math



class BaselineAttackClient(FedRecClient):
    def __init__(self, train_ind, m_item, dim, client_id):
        super().__init__(train_ind, [], [], m_item, dim, client_id)

    def train_(self, items_emb, linear_layers):
        a, b, c, _ = super().train_(items_emb, linear_layers)
        return a, b, c, None

    def eval_(self, _items_emb, _linear_layers):
        return None, None

class AttackClient(nn.Module):
    def __init__(self, target_items, m_item, dim, interested_items = [], client_id=0):
        super().__init__()
        self._target_ = target_items
        self.m_item = m_item
        self.dim = dim
        self._user_emb = nn.Embedding(1, self.dim)
        self._other_user_emb = nn.Embedding(1, self.dim)
        self._interested_ = interested_items
        self._interested_origin = interested_items
        self._other_user_item_pool = list(range(self.m_item))
        self.client_id = client_id

        self.interested_neg_items = []
        for _ in self._interested_:
            for _ in range(args.num_neg):
                neg_item = np.random.randint(self.m_item)
                while neg_item in self._interested_:
                    neg_item = np.random.randint(self.m_item)
                self.interested_neg_items.append(neg_item)
        

    def forward(self, user_emb, items_emb, linear_layers):
        user_emb = user_emb.repeat(len(items_emb), 1)
        v = torch.cat((user_emb, items_emb), dim=-1)

        for i, (w, b) in enumerate(linear_layers):
            v = v @ w.t() + b
            if i < len(linear_layers) - 1:
                v = v.relu()
            else:
                v = v.sigmoid()
        return v.view(-1)

    def train_on_user_emb(self, user_emb, items_emb, linear_layers, labels=None):#user:(batch,dim), items:(batch, n, dim)
        if(labels == None):
            labels = torch.ones(len(self._target_)).to(args.device)
        predictions = self.forward(user_emb.requires_grad_(False), items_emb, linear_layers)
        loss = nn.BCELoss()(predictions, labels).to(args.device)
        return loss

    def train_(self, items_emb, linear_layers):
        target_items_emb_notrain = items_emb[self._target_].clone().detach()
        target_linear_layers_notrain = [[w.clone().detach(), b.clone().detach()] for w, b in linear_layers]
        target_items_emb_train = items_emb[self._target_].clone().detach().requires_grad_(True)
        target_linear_layers_train = [[w.clone().detach().requires_grad_(True),
                          b.clone().detach().requires_grad_(True)]
                         for (w, b) in linear_layers]

        
        '''group attack'''
        interested_items_emb_notrain = items_emb[self._interested_].clone().detach()
        combined_items_emb_notrain = torch.cat([interested_items_emb_notrain, target_items_emb_notrain], dim=0)

        interested_neg_items_emb_notrain = items_emb[self.interested_neg_items].clone().detach()
        combined_items_emb_notrain2 = torch.cat([interested_items_emb_notrain, interested_neg_items_emb_notrain, target_items_emb_notrain], dim=0)

        
        s = 10
        total_loss = 0
        for _ in range(s):
            nn.init.normal_(self._user_emb.weight, std=0.01)
            
            if args.attack == 'A-ra':
                total_loss += (1 / s) * self.train_on_user_emb(self._user_emb.weight, target_items_emb_train, target_linear_layers_train)
            
            if args.attack == 'A-hum' or args.attack == 'test':#hard sample
                for __ in range(30):
                    predictions = self.forward(self._user_emb.weight.requires_grad_(True),
                                               target_items_emb_notrain, target_linear_layers_notrain)
                    loss = nn.BCELoss()(predictions, torch.zeros(len(self._target_)).to(args.device))

                    self._user_emb.zero_grad()
                    loss.backward()
                    self._user_emb.weight.data.add_(self._user_emb.weight.grad, alpha=-args.lr)
                total_loss += (1 / s) * self.train_on_user_emb(self._user_emb.weight, target_items_emb_train, target_linear_layers_train)
            
        total_loss.backward()

        items_emb_grad = target_items_emb_train.grad
        linear_layers_grad = [[w.grad, b.grad] for (w, b) in target_linear_layers_train]
        return self._target_, items_emb_grad, linear_layers_grad, None

    def eval_(self, _items_emb, _linear_layers):
        return None, None



class OursAttackClient(AttackClient):
    def __init__(self, target_items, m_item, dim, interested_items = [], client_id=0, items_popularity=[]):
        super().__init__(target_items, m_item, dim, interested_items)
        self.user_batch_size = 10#10
        self._user_emb = nn.Embedding(self.user_batch_size, self.dim)
        self._other_user_emb = nn.Embedding(self.user_batch_size, self.dim)
        self.user_optimizer = torch.optim.Adam(
            list(self._user_emb.parameters()) + list(self._other_user_emb.parameters()),
            lr=args.generator_lr)
        self.group_user_optimizer = torch.optim.Adam(
            list(self._user_emb.parameters()),
            lr=args.generator_lr)
        self.other_user_optimizer = torch.optim.Adam(
            list(self._other_user_emb.parameters()),
            lr=args.generator_lr)
        
        self.client_id = client_id
        self.global_epoch = 1
        self.dist_loss = self.contrastive_loss
        self.items_popularity = items_popularity

        self.prev_group_weight = None
        self.prev_others_weight = None
        
    
    def forward(self, user_emb, items_emb, linear_layers):#user:(batch,dim), items:(batch, n, dim)
        user_emb = user_emb.unsqueeze(1).repeat(1, items_emb.shape[1], 1)
        v = torch.cat((user_emb, items_emb), dim=-1)#(batch, n, 2*dim)

        for i, (w, b) in enumerate(linear_layers):
            v = v @ w.t() + b
            if i < len(linear_layers) - 1:
                v = v.relu()
            else:
                v = v.sigmoid()
        return v.squeeze(-1)#(batch, n)
    

    def contrastive_loss(self, group, others, margin=15.0):#

        group_expand = group.unsqueeze(1)      # [B, 1, D]
        others_expand = others.unsqueeze(0)    # [1, B, D]
        dists = torch.norm(group_expand - others_expand, p=2, dim=2)  # [B, B]
        loss = 0.5 * torch.clamp(margin - dists, min=0).pow(2).mean()
        return loss
    
    def contrastive_loss_cosine(self, group, others, margin=1.5):
        # group: [B, D], others: [B, D]
        group_expand = group.unsqueeze(1)      # [B, 1, D]
        others_expand = others.unsqueeze(0)    # [1, B, D]
        cos = torch.nn.CosineSimilarity(dim=2)
        sim_matrix = cos(group_expand, others_expand)  # [B, B]
        cos_dist = 1 - sim_matrix#
        loss = 0.5 * torch.clamp(margin - cos_dist, min=0).pow(2).mean()
        return loss
    
    def similarity_loss(self, target_emb, interested_emb):
        '''余弦相似度距离'''
        cos = nn.CosineSimilarity(dim=-1)
        sim = cos(target_emb, interested_emb.requires_grad_(False))  # (n,)
        return (1 - sim).mean()
        '''平方距离'''
        #return torch.mean((target_emb - interested_emb.requires_grad_(False)).pow(2).sum(dim=1))
        '''欧几里得距离'''
        #return torch.mean(torch.norm(target_emb - interested_emb.requires_grad_(False), p=2, dim=1))

    
    def cluster(self, items_emb, n, m):
        # items_emb: (m_item, dim)
        all_indices = torch.arange(self.m_item, device=args.device)
        mask = torch.ones(self.m_item, dtype=torch.bool, device=args.device)
        mask[self._interested_+self._target_] = False 
        candidate_indices = all_indices[mask]  # (num_candidates,)
        interested_emb = items_emb[self._interested_].clone().detach()  # (num_interested, dim)
        mean_emb = interested_emb.mean(dim=0, keepdim=True)  # (1, dim)
        candidate_emb = items_emb[mask].clone().detach()  # (num_candidates, dim)
        cos = nn.CosineSimilarity(dim=1)
        sims = cos(candidate_emb, mean_emb)  # (num_candidates,)
        topk_sim = torch.topk(sims, n)
        most_similar_ids = candidate_indices[topk_sim.indices].tolist()
        most_similar_sims = topk_sim.values.tolist()
        least_similar_ids, least_similar_sims = None, None
        return most_similar_ids, most_similar_sims, least_similar_ids, least_similar_sims
    
    
    def weighted_popularity_at_k_normalized(self, rating_k, items_popularity):
        pop_min = items_popularity.min()
        pop_max = items_popularity.max()
        def pop_norm(i):
            return (items_popularity[i] - pop_min) / (pop_max - pop_min) if pop_max > pop_min else 0.0

        weights = [1 / np.log2(j + 2) for j in range(len(rating_k[0]))]  # j从0开始
        user_scores = []
        for r in rating_k:
            score = sum(pop_norm(i) * w for i, w in zip(r, weights))
            user_scores.append(score)
        return sum(user_scores) / len(user_scores)

    def get_popularity_rank(self, items_popularity):
        sorted_indices = np.argsort(-items_popularity)
        rank = np.empty_like(sorted_indices)
        rank[sorted_indices] = np.arange(1, len(items_popularity) + 1)
        return rank  #

    def train_(self, items_emb, linear_layers):

        
        nn.init.normal_(self._user_emb.weight, std=0.01)
        nn.init.normal_(self._other_user_emb.weight, std=0.01)

        target_items_emb_notrain = items_emb[self._target_].clone().detach()
        target_linear_layers_notrain = [[w.clone().detach(), b.clone().detach()] for w, b in linear_layers]
        target_items_emb_train = items_emb[self._target_].clone().detach().requires_grad_(True)
        target_linear_layers_train = [[w.clone().detach().requires_grad_(True),
                        b.clone().detach().requires_grad_(True)]
                        for (w, b) in linear_layers]

        '''cluster'''
        if(args.is_cluster==True):
            similar_ids, _, dissimilar_ids, _ = self.cluster(items_emb, args.cluster_num, int(self.m_item*0.1))
            self._interested_ = self._interested_origin + similar_ids

        
        '''group attack'''
        interested_items_emb_notrain = items_emb[self._interested_].clone().detach()
        combined_items_emb_notrain = torch.cat([interested_items_emb_notrain, target_items_emb_notrain], dim=0)
        combined_items_emb_notrain = combined_items_emb_notrain.unsqueeze(0).repeat(self.user_batch_size, 1, 1)
        
        '''train generator'''
        for gen_epoch in range(args.generator_epochs):
            t1 = time()
            interested_neg_items = []
            for _ in range(self.user_batch_size):
                neg_items = []
                for _ in range(len(self._interested_)):#100
                    neg_item = np.random.randint(self.m_item)
                    while neg_item in self._interested_ or neg_item in self._target_:
                        neg_item = np.random.randint(self.m_item)
                    neg_items.append(neg_item)
                interested_neg_items.append(neg_items)
            
            interested_neg_items_tensor = torch.tensor(interested_neg_items, dtype=torch.long)  # shape: [user_batch_size, num_neg_items_per_user]
            interested_neg_items_emb_notrain = items_emb[interested_neg_items_tensor].clone().detach()  # shape: [user_batch_size, num_neg_items_per_user, dim]
            

            self.user_optimizer.zero_grad()
            predictions_group = self.forward(self._user_emb.weight.requires_grad_(True), combined_items_emb_notrain, target_linear_layers_notrain)
            labels_group = torch.cat([torch.ones(len(self._interested_)), torch.zeros(len(self._target_))]).to(args.device)
            labels_group = labels_group.unsqueeze(0).expand(self.user_batch_size, -1)#（batch, 4+1)
            prediction_loss_group = nn.BCELoss()(predictions_group, labels_group)

            predictions_others = self.forward(self._other_user_emb.weight.requires_grad_(True), interested_neg_items_emb_notrain, target_linear_layers_notrain)
            labels_others = torch.ones(len(self._interested_)).to(args.device)
            labels_others = labels_others.unsqueeze(0).expand(self.user_batch_size, -1)
            prediction_loss_others = nn.BCELoss()(predictions_others, labels_others)

            distance_loss = self.dist_loss(self._user_emb.weight.requires_grad_(True), self._other_user_emb.weight.requires_grad_(True), margin=args.distance_margin)
            if(args.is_distance==True):
                loss = prediction_loss_group + prediction_loss_others + distance_loss  
            else:
                loss = prediction_loss_group + prediction_loss_others
    
            loss.backward()
            self.user_optimizer.step()
            t2 = time()


        '''update target item and linear layer'''
        others_items_emb_train_expand = target_items_emb_train.unsqueeze(0).expand(self.user_batch_size, -1, -1)#batch, 1, dim
        group_items_emb_train_expand = others_items_emb_train_expand
        interaction_loss_group = self.train_on_user_emb(self._user_emb.weight, group_items_emb_train_expand, target_linear_layers_train, labels=torch.ones((self.user_batch_size, len(self._target_))).to(args.device))
        interaction_loss_others = self.train_on_user_emb(self._other_user_emb.weight, others_items_emb_train_expand, target_linear_layers_train, labels=torch.zeros((self.user_batch_size, len(self._target_))).to(args.device))
        
        
        '''target item在group user rating_k中的排名'''
        rating = self.forward(self._user_emb.weight.clone().detach(), items_emb.unsqueeze(0).repeat(self.user_batch_size, 1, 1).clone().detach(), target_linear_layers_notrain)
        _, all_rating = torch.topk(rating, self.m_item)#(n,5)
        all_rating = all_rating.cpu().tolist()#(n,5)
        target_rank = [items.index(self._target_[0]) for items in all_rating]
        avg_rank_group = sum(target_rank) / len(target_rank)
        '''target item在other user rating_k中的排名'''
        rating = self.forward(self._other_user_emb.weight.clone().detach(), items_emb.unsqueeze(0).repeat(self.user_batch_size, 1, 1).clone().detach(), target_linear_layers_notrain)
        _, all_rating = torch.topk(rating, self.m_item)#(n,5)
        all_rating = all_rating.cpu().tolist()#(n,5)
        '''target item在other user rating_k中的排名'''
        target_rank = [items.index(self._target_[0]) for items in all_rating]
        avg_rank_others = sum(target_rank) / len(target_rank)

        max_rank = self.m_item  
        min_rank = 0
        group_weight = (avg_rank_group - min_rank) / (max_rank - min_rank)
        others_weight = (avg_rank_others - min_rank) / (max_rank - min_rank)

        total = group_weight + others_weight
        if(total == 0):
            group_weight = 0
            others_weight = 0
        else:
            group_weight /= total
            others_weight /= total
        
        x = (self.global_epoch / 30)
        epoch_factor = x**2
        if(args.is_adaptive_coef==True):
            interaction_loss = (1+group_weight*epoch_factor+args.epoch_coef*self.global_epoch)*interaction_loss_group + (1-others_weight*epoch_factor)*interaction_loss_others
        else:
            interaction_loss = interaction_loss_group + interaction_loss_others
        
        similarity_loss = self.similarity_loss(target_items_emb_train, interested_items_emb_notrain)
        total_loss = interaction_loss + args.similarity_coef * similarity_loss
        total_loss.backward()
        items_emb_grad = target_items_emb_train.grad
        embs_grad = items_emb_grad
        updated_items = self._target_
        linear_layers_grad = [[w.grad, b.grad] for (w, b) in target_linear_layers_train]

        self.global_epoch += 1
        return updated_items, embs_grad, linear_layers_grad, None 
    

