import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
import scipy.sparse as sp
from .model_0320 import Encoder_overall, Discriminator
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
import torch.optim.lr_scheduler as lr_scheduler
from .preprocess import adjacent_matrix_preprocessing, preprocess_graph
from sklearn.cluster import SpectralClustering
from collections import Counter


# from .optimal_clustering_HLN import R5

class Train_SpaMode:
    def __init__(self,
                 data,
                 datatype='SPOTS',
                 device=torch.device('cpu'),
                 random_seed=2024,
                 learning_rate=0.01,
                 weight_decay=5e-2,
                 epochs=200,
                 dim_input=3000,
                 dim_output=64,
                 weight_factors=[1, 5, 1, 1],
                 Arg=None
                 ):
        '''\

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input, Our current model supports 'SPOTS', 'Stereo-CITE-seq', and 'Spatial-ATAC-RNA-seq'. We plan to extend our model for more data types in the future.
            The default is 'SPOTS'.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 2022.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight decay to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 1500.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        weight_factors : list, optional
            Weight factors to balance the influcences of different omics data on model training.

        Returns
        -------
        The learned representation 'self.emb_combined'.

        '''
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors

        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)


        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to_dense().to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to_dense().to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to_dense().to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to_dense().to(self.device)

        import copy
        mask_feature_omics1 = copy.deepcopy(self.adj_feature_omics1)
        mask_feature_omics2 = copy.deepcopy(self.adj_feature_omics2)
        mask_spatial = copy.deepcopy(self.adj_spatial_omics1)

        mask_feature_omics1[mask_feature_omics1 > 0] = 1
        mask_feature_omics2[mask_feature_omics2 > 0] = 1
        mask_spatial[mask_spatial > 0] = 1

        mask_core = mask_feature_omics1 + mask_feature_omics2
        self.mask_adj = mask_core + mask_spatial
        mask_core[mask_core != 2] = 0
        core_adj = (self.adj_feature_omics1 + self.adj_feature_omics2) / 2 * mask_core
        differ_1 = torch.nn.functional.cosine_similarity(self.adj_feature_omics1.view(-1), core_adj.view(-1), dim=0)
        differ_2 = torch.nn.functional.cosine_similarity(self.adj_feature_omics2.view(-1), core_adj.view(-1), dim=0)

        differ_stack = torch.stack([differ_1, differ_2], dim=0)

        weight = F.softmax(differ_stack)
        print(differ_stack, weight)
        # weight = [1.0, 1.0]


        self.fuse_adj = (
                self.adj_feature_omics1 * weight[0] + self.adj_feature_omics2 * weight[1])

        mask_spatial = copy.deepcopy(self.adj_spatial_omics1)
        mask_spatial[mask_spatial != 0] = 1
        self.local_adj = self.fuse_adj * mask_spatial
        self.local_adj[self.local_adj != 0] = 1

        self.global_adj = self.fuse_adj + mask_spatial
        self.global_adj[self.global_adj != 0] = 1

        self.paramed_fuse_adj = Parametered_Graph(self.fuse_adj + self.adj_spatial_omics1, self.device)


        self.EMA_coeffi = 0.9
        self.K = 5
        self.T = 4
        self.arg = Arg

        # self.clustering = R5(self.datatype, self.arg)

        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)

        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs

        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output

        self.Biloss_fn = nn.CrossEntropyLoss()

        self.epochs = self.arg["training_epoch"]
        self.weight_factors = [self.arg["weight1"], self.arg["weight2"]]  # if RNA Slice2: [1, 10]
        self.weight_bi = self.arg["weight_bi"]
        self.weight_adv = self.arg["weight_adv"]
        self.weight_kl = self.arg["weight_kl"]
        self.scheduler = False
        self.learning_rate = self.arg["learning_rate"]
        self.early_stop = self.arg["earlystop"]
        self.early_stop_patience = self.arg["early_stop_patience"]  # if RNA Slice2: 5
        self.weight_decay = self.arg["weight_decay"]


    def train(self):

        self.ADV = Discriminator(self.dim_output1).to(self.device)

        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2).to(
            self.device)

        print('optimizer: ', self.arg["optimizer"])
        if self.arg["optimizer"] == 'SGD':
            self.optimizer = torch.optim.SGD(list(self.model.parameters()) +
                                             list(self.ADV.parameters()),
                                             lr=self.learning_rate,
                                             momentum=0.9,
                                             weight_decay=self.weight_decay)
        elif self.arg["optimizer"] == 'AdamW':
            self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.ADV.parameters()),
                                             lr=self.learning_rate,
                                             weight_decay=self.weight_decay
                                              )

        scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        self.model.train()
        min_loss = 100
        early_stop_count = 0
        progress_bar = tqdm(range(self.epochs))

        for epoch in progress_bar:
            self.model.train()
            adj = self.paramed_fuse_adj().to(self.device)
            self.zero_indices, self.one_indices = self.get_heterogeneous_label(self.global_adj)
            N = self.zero_indices.shape[0]
            random_indices = torch.randint(0, len(self.one_indices), (N,))
            sampled_indices = self.one_indices[random_indices]

            results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1,
                                 adj, self.adj_spatial_omics2, adj, sampled_indices, self.zero_indices)

            loss, recon_loss, kl_loss, Biloss, adv_loss, moe_loss = self.loss(results, self.features_omics1, self.features_omics2)

            loss = loss

            if self.early_stop:
                if early_stop_count >= self.early_stop_patience:
                    print('Early Stop')
                    break
                elif Biloss < min_loss:
                    min_loss = Biloss
                    early_stop_count = 0
                else:
                    early_stop_count += 1

            # print(recon_loss, Biloss, adv_loss)
            progress_bar.set_postfix(
                recon=recon_loss.item(),
                Biloss=Biloss.item(),
                kl_loss=kl_loss.item(),
                moe_loss=moe_loss.item(),
                adv_loss=adv_loss.item(),
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler:
                scheduler.step()

        print("Model training finished!\n")

        emb = results['emb_latent_combined']

        emb_combined = F.normalize(emb, p=2, eps=1e-12, dim=1)

        adj_combined = self.fuse_adj

        output = {
                  'Smoe': emb_combined.detach().cpu().numpy(),
                  'adj': adj_combined.detach().cpu().numpy(),
                  }

        return output

    def imputation(self, data, omics):
        adata_omics1 = data['adata_omics1']
        adata_omics2 = data['adata_omics2']
        adj = adjacent_matrix_preprocessing(adata_omics1, adata_omics2)

        adj_spatial_omics1 = adj['adj_spatial_omics1'].to_dense().to(self.device)
        adj_spatial_omics2 = adj['adj_spatial_omics2'].to_dense().to(self.device)
        adj_feature_omics1 = adj['adj_feature_omics1'].to_dense().to(self.device)
        adj_feature_omics2 = adj['adj_feature_omics2'].to_dense().to(self.device)

        self.tmp_adj_spatial_omics1 = adj_spatial_omics1

        import copy
        mask_feature_omics1 = copy.deepcopy(adj_feature_omics1)
        mask_feature_omics2 = copy.deepcopy(adj_feature_omics2)
        mask_spatial = copy.deepcopy(adj_spatial_omics1)

        mask_feature_omics1[mask_feature_omics1 > 0] = 1
        mask_feature_omics2[mask_feature_omics2 > 0] = 1
        mask_spatial[mask_spatial > 0] = 1

        mask_core = mask_feature_omics1 + mask_feature_omics2
        mask_adj = mask_core + mask_spatial
        mask_core[mask_core != 2] = 0
        core_adj = (adj_feature_omics1 + adj_feature_omics2) / 2 * mask_core
        differ_1 = torch.nn.functional.cosine_similarity(adj_feature_omics1.view(-1), core_adj.view(-1), dim=0)
        differ_2 = torch.nn.functional.cosine_similarity(adj_feature_omics2.view(-1), core_adj.view(-1), dim=0)

        differ_stack = torch.stack([differ_1, differ_2], dim=0)

        weight = F.softmax(differ_stack)
        print(differ_stack, weight)

        fuse_adj = (
                adj_feature_omics1 * weight[0] + adj_feature_omics2 * weight[1])

        mask_spatial = copy.deepcopy(self.adj_spatial_omics1)
        mask_spatial[mask_spatial != 0] = 1

        paramed_fuse_adj = Parametered_Graph(fuse_adj + adj_spatial_omics1, self.device).to(self.device)

        adj = paramed_fuse_adj()
        if omics == 1:
            features_omics1 = torch.FloatTensor(adata_omics1.obsm['feat'].copy()).to(self.device)
            results = self.model(features_omics1, None, adj_spatial_omics1,
                                 adj, adj_spatial_omics2, adj, train=False)
        else:
            features_omics2 = torch.FloatTensor(adata_omics2.obsm['feat'].copy()).to(self.device)
            results = self.model(None, features_omics2, adj_spatial_omics1,
                                 adj, adj_spatial_omics2, adj, train=False)

        return results

    def loss(self, results, features_omics1, features_omics2):
        # reconstruction loss
        loss_recon_omics1 = F.mse_loss(features_omics1, results['emb_recon_omics1'])
        loss_recon_omics2 = F.mse_loss(features_omics2, results['emb_recon_omics2'])

        recon_loss = self.arg["weight1"] * loss_recon_omics1 + self.arg["weight2"] * loss_recon_omics2

        one_hot_labels = torch.nn.functional.one_hot(results['mixed_labels'].long(), num_classes=2).to(self.device)

        Biloss = self.BCE_loss_0213(results['discriminator_pred'], one_hot_labels.float()) * self.arg["weight_bi"]

        adv_loss = self.adv_loss(results) * self.arg["weight_adv"]

        kl_loss = results['kl_div'] * self.arg["weight_kl"]
        moe_loss = results['moe_loss'] * self.arg["weight_moe"]

        loss = recon_loss + kl_loss + Biloss + adv_loss + moe_loss

        return loss, recon_loss, kl_loss, Biloss, adv_loss, moe_loss

    def adv_loss(self, results):
        # assert not torch.isnan(results['logits_combine']).any(), "Input contains NaN"
        # assert not torch.isinf(results['logits_combine']).any(), "Input contains Inf"

        logits_omics1 = self.ADV(results['emb_latent_omics1'], if_grl=False)
        logits_omics2 = self.ADV(results['emb_latent_omics2'], if_grl=False)

        inv_logits_omics1 = self.ADV(results['inv_emb_latent_omics1'], if_grl=True)
        inv_logits_omics2 = self.ADV(results['inv_emb_latent_omics2'], if_grl=True)

        N = results['emb_latent_omics1'].shape[0]
        # criterion = nn.BCELoss()
        criterion = nn.CrossEntropyLoss().to(self.device)

        omics1_labels = torch.zeros((N,), dtype=torch.long).to(self.device)
        omics2_labels = torch.ones((N,), dtype=torch.long).to(self.device)

        loss_omics1 = criterion(logits_omics1, omics1_labels) + criterion(inv_logits_omics1, omics1_labels)
        loss_omics2 = criterion(logits_omics2, omics2_labels) + criterion(inv_logits_omics2, omics2_labels)

        # 总损失
        loss_D = loss_omics1 + loss_omics2

        return loss_D




    def cal_BiCE(self, embedding, thr=0.99, pos_weight=None, loss_weight=10):
        self_similarity = torch.mm(embedding, embedding.T)

        ss_metric = torch.sigmoid(self_similarity)

        binary_metric = (ss_metric > thr).int().detach()

        return self_similarity, binary_metric

    def cal_KL(self, S, T, tau=2):
        S = F.log_softmax(S / tau, dim=1)
        T = F.softmax(T / tau, dim=1)
        KL = F.kl_div(S, T, reduction='batchmean') * tau * tau
        return KL

    def BCE_loss_0213(self, logits, targets, weight=0.9):
        epsilon = 1e-10
        logits = torch.sigmoid(logits)
        # logits = torch.softmax(logits, dim=1)

        Biloss = -torch.mean(
            targets * torch.log(logits + epsilon) * weight + (1 - targets) * torch.log(1 - logits + epsilon) * 0.1)
        return Biloss

    def get_heterogeneous_label(self, adj):
        # print(adj)
        binary_tensor = (adj == 0).to(torch.float)

        zero_indices = torch.nonzero(binary_tensor == 0)

        one_indices = torch.nonzero(binary_tensor == 1)

        return zero_indices, one_indices



class Parametered_Graph(nn.Module):
    def __init__(self, adj, device):
        super(Parametered_Graph, self).__init__()
        self.adj = adj
        self.device = device

        self.tau = 1
        self.hard = False
        # self.adj = F.gumbel_softmax(adj, tau=self.tau, hard=self.hard)

        n = self.adj.shape[0]
        self.paramed_adj_omics = nn.Parameter(torch.FloatTensor(n, n))
        self.paramed_adj_omics.data.copy_(self.adj)
        # torch.nn.init.xavier_uniform_(self.paramed_adj_omics)

    def forward(self):
        adj = self.paramed_adj_omics

        # _temp = self.paramed_adj_omics.clone()
        # #
        # for i in range(10):
        #     adj = adj + F.gumbel_softmax(_temp, tau=self.tau, hard=self.hard)

        adj = (adj + adj.t()) / 2
        adj = nn.ReLU(inplace=False)(adj)
        adj = self._normalize(adj.to(self.device) + torch.eye(adj.shape[0]).to(self.device))

        return adj.to(self.device)

    def normalize(self, A=None):

        if A is None:
            adj = (self.paramed_adj_omics + self.paramed_adj_omics.t()) / 2
            adj = nn.ReLU(inplace=True)(adj)
            # adj.fill_diagonal_(0)
            normalized_adj = self._normalize(adj.to(self.device) + torch.eye(adj.shape[0]).to(self.device))
        else:
            adj = (A + A.t()) / 2
            adj = nn.ReLU(inplace=True)(adj)
            # adj.fill_diagonal_(0)
            normalized_adj = self._normalize(adj.to(self.device) + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx







