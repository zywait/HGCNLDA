import numpy as np
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import nn, optim
from functools import partial

# from ..loss_fn import FocalLoss
from ..model_help import BaseModel
from .dataset import FullGraphData
from .. import MODEL_REGISTRY
from .gnn import *
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import GraphAttentionLayer, AttentionLayer
from pytorch_lightning.loggers import TensorBoardLogger
# from ..loss_fn import FocalLoss
import torch.nn.init as init


class EdgeDropout(nn.Module):
    def __init__(self, keep_prob=0.5):
        super(EdgeDropout, self).__init__()
        assert keep_prob > 0
        self.keep_prob = keep_prob
        self.register_buffer("p", torch.tensor(keep_prob))

    def forward(self, edge_index, edge_weight):
        if self.training:
            mask = torch.rand(edge_index.shape[1], device=edge_weight.device)  # 随机产生edge_index.shape[1]个0~1之间的数
            mask = torch.floor(mask + self.p).type(torch.bool)  # 以self.p的概率进行dropout
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask] / self.p
        return edge_index, edge_weight

    def forward2(self, edge_index):
        if self.training:
            mask = ((torch.rand(edge_index._values().size()) + (self.keep_prob)).floor()).type(torch.bool)
            rc = edge_index._indices()[:, mask]
            val = edge_index._values()[mask] / self.p
            return torch.sparse.FloatTensor(rc, val)
        return edge_index

    def __repr__(self):
        return '{}(keep_prob={})'.format(self.__class__.__name__, self.keep_prob)


class ShareGCN(nn.Module):
    def __init__(self, size_u, size_v, size_w, in_channels=64, out_channels=64, share=True, normalize=True,
                 dropout=0.4, use_sparse=True, act=nn.ReLU, cached=False, bias=False, add_self_loops=False,
                 **kwargs):
        super(ShareGCN, self).__init__()
        self.size_u = size_u
        self.size_v = size_v
        self.size_w = size_w
        self.num_nodes = size_u + size_v + size_w
        self.share = share
        self.use_sparse = use_sparse
        self.dropout = nn.Dropout(dropout)
        self.u_encoder = GCNConv(in_channels=in_channels, out_channels=out_channels,
                                     normalize=normalize, add_self_loops=add_self_loops,
                                     cached=cached, bias=bias, **kwargs)
        if not self.share:
            self.v_encoder = GCNConv(in_channels=in_channels, out_channels=out_channels,
                                         normalize=normalize, add_self_loops=False,
                                         cached=cached, bias=bias, **kwargs)
            self.w_encoder = GCNConv(in_channels=in_channels, out_channels=out_channels,
                                         normalize=normalize, add_self_loops=False,
                                         cached=cached, bias=bias, **kwargs)
        self.act = act(inplace=True) if act else nn.Identity()

    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight, w_edge_index=None, w_edge_weight=None):
        x = self.dropout(x)
        if self.share:
            edge_index = torch.cat([u_edge_index, v_edge_index], dim=1)
            edge_weight = torch.cat([u_edge_weight, v_edge_weight], dim=0)
            if w_edge_index is not None and w_edge_weight is not None:
                edge_index = torch.cat([edge_index, w_edge_index], dim=1)
                edge_weight = torch.cat([edge_weight, w_edge_weight], dim=0)
            if self.use_sparse:
                node_nums = self.num_nodes
                edge_index = SparseTensor(row=edge_index[0], col=edge_index[1],
                                          value=edge_weight,
                                          sparse_sizes=(node_nums, node_nums)).t()
            feature = self.u_encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        else:
            if self.use_sparse:
                node_nums = self.num_nodes
                u_edge_index = SparseTensor(row=u_edge_index[0], col=u_edge_index[1],
                                            value=u_edge_weight,
                                            sparse_sizes=(node_nums, node_nums)).t()
                v_edge_index = SparseTensor(row=v_edge_index[0], col=v_edge_index[1],
                                            value=v_edge_weight,
                                            sparse_sizes=(node_nums, node_nums)).t()
                if w_edge_index is not None and w_edge_weight is not None:
                    w_edge_index = SparseTensor(row=w_edge_index[0], col=w_edge_index[1],
                                                value=w_edge_weight,
                                                sparse_sizes=(node_nums, node_nums)).t()
            feature_u = self.u_encoder(x=x, edge_index=u_edge_index, edge_weight=u_edge_weight)
            feature_v = self.v_encoder(x=x, edge_index=v_edge_index, edge_weight=v_edge_weight)
            feature_w = torch.zeros_like(feature_v)
            if w_edge_index is not None and w_edge_weight is not None:
                feature_w = self.w_encoder(x=x, edge_index=w_edge_index, edge_weight=w_edge_weight)
            feature = feature_u + feature_v + feature_w
        output = self.act(feature)
        return output

class MetaGCN(nn.Module):
    def __init__(self, size_u, size_v, size_w, in_channels=64, out_channels=64, normalize=True,
                 dropout=0.4, use_sparse=True, act=nn.ReLU, cached=False, bias=False, add_self_loops=False,
                 metapaths=None, **kwargs):
        super(MetaGCN, self).__init__()
        if metapaths is None:
            metapaths = ["vv", "uu", "ww"]
        self.size_u = size_u
        self.size_v = size_v
        self.size_w = size_w
        self.num_nodes = size_u + size_v + size_w
        self.use_sparse = use_sparse
        self.dropout = nn.Dropout(dropout)
        mp_num = 0
        self.mps = metapaths
        encoders = []
        for mp in self.mps:
            encoders.append(GCNConv(in_channels=in_channels, out_channels=out_channels,
                                     normalize=normalize, add_self_loops=add_self_loops,
                                     cached=cached, bias=bias, **kwargs))
            mp_num += 1
        self.mp_num = mp_num
        self.encoders = nn.ModuleList(encoders)
        self.act = act(inplace=True) if act else nn.Identity()

    def forward(self, x, edge_index_dict, edge_weight_dict):
        assert len(self.encoders) == len(edge_index_dict)
        node_nums = self.num_nodes
        x = self.dropout(x)
        outputs = []
        mp_labels = []
        for i, mp in enumerate(self.mps):
            edge_index = edge_index_dict[mp]
            edge_weight = edge_weight_dict[mp]
            if self.use_sparse:
                edge_index = SparseTensor(row=edge_index[0], col=edge_index[1],
                                            value=edge_weight,
                                            sparse_sizes=(node_nums, node_nums)).t()
            feature = self.encoders[i](x=x, edge_index=edge_index, edge_weight=edge_weight)
            output = self.act(feature)
            outputs.append(output)
            mp_labels.append(mp)
        return outputs, mp_labels


class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""

    def __init__(self, size_u, size_v, input_dim=None, dropout=0.4, act=nn.Sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.size_u = size_u
        self.size_v = size_v

        self.dropout = nn.Dropout(dropout)
        if input_dim:
            self.weights = nn.Linear(input_dim, input_dim, bias=False)
            nn.init.xavier_uniform_(self.weights.weight)
        self.act = act

    def forward(self, feature):
        feature = self.dropout(feature)
        L = feature[:self.size_u]
        D = feature[self.size_u:self.size_u + self.size_v]
        M = feature[self.size_u + self.size_v:]
        if hasattr(self, "weights"):
            D = self.weights(D)
        x = L @ D.T
        outputs = self.act(x)
        return outputs, L, D, M

class IMCDecoder(nn.Module):
    """Decoder model layer for link prediction."""

    def __init__(self, size_u, size_v, input_dim, dropout=0.4, act=nn.Sigmoid):
        super(IMCDecoder, self).__init__()
        self.size_u = size_u
        self.size_v = size_v

        self.dropout = nn.Dropout(dropout)
        self.weights = nn.Linear(input_dim, input_dim, bias=False)

        nn.init.xavier_uniform_(self.weights.weight)
        self.act = act() if act is not None else nn.Identity()

    def forward(self, feature):
        feature = self.dropout(feature)
        L = feature[:self.size_u]
        D = feature[self.size_u:self.size_u + self.size_v]
        M = feature[self.size_u + self.size_v:]
        D = self.weights(D)
        x = L @ D.T
        outputs = self.act(x)
        return outputs, L, D, M

class BiLinearDecoder(nn.Module):
    def __init__(self, size_u, size_v, input_dim=None, dropout=0.4, num_weights=2, num_classes=2, activation=F.relu):
        super(BiLinearDecoder, self).__init__()
        self.size_u = size_u
        self.size_v = size_v
        self.input_dim = input_dim
        self.num_weights = num_weights
        self.num_classes = num_classes
        self.activation = activation

        self.weight = nn.Parameter(torch.Tensor(num_weights, input_dim, input_dim))
        self.weight_classifier = nn.Parameter(torch.Tensor(num_weights, num_classes))
        self.reset_parameters()

        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        init.kaiming_uniform_(self.weight_classifier)

    def forward(self, feature):
        feature = self.dropout(feature)
        lnc_inputs = feature[:self.size_u]
        dis_inputs = feature[self.size_u:self.size_u + self.size_v]
        M = feature[self.size_u + self.size_v:]
        row_idx, col_idx = torch.meshgrid(torch.arange(lnc_inputs.shape[0]), torch.arange(dis_inputs.shape[0]))
        lnc_indices = row_idx.reshape(-1)
        dis_indices = col_idx.reshape(-1)
        lnc_inputs = lnc_inputs[lnc_indices]
        dis_inputs = dis_inputs[dis_indices]

        basis_outputs = []
        for i in range(self.num_weights):
            tmp = torch.matmul(lnc_inputs, self.weight[i])
            out = torch.sum(tmp * dis_inputs, dim=1, keepdim=True)
            basis_outputs.append(out)

        basis_outputs = torch.cat(basis_outputs, dim=1)

        outputs = torch.matmul(basis_outputs, self.weight_classifier)
        outputs = self.activation(outputs)

        return outputs, lnc_inputs, dis_inputs, M

@MODEL_REGISTRY.register()
class SGALDA(BaseModel):
    DATASET_TYPE = "FullGraphDataset"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SGALDA model config")
        parser.add_argument("--embedding_dim", default=128, type=int)
        parser.add_argument("--layer_num", default=1, type=int)
        parser.add_argument("--homo_metapaths", default="uu,vv,uvu,vuv,uwu,vwv", type=str,  # u：lncRNA、v：disease、w：miRNA
                            choices=["uu", "vv", "uvu", "vuv", "uwu", "vwv"])
        parser.add_argument("--heter_metapaths", default="uv,uwv,uvwv", type=str,
                            choices=["uv", "uwv", "uvwv"])
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument("--bias", type=bool, default=True)
        parser.add_argument("--add_self_loops", type=bool, default=True)
        parser.add_argument("--use_sim", type=bool, default=True)
        parser.add_argument("--fl_alpha", type=float, default=0.25)
        parser.add_argument("--fl_gamma", type=float, default=2)
        return parent_parser

    def __init__(self, size_u, size_v, size_w, lr=0.05, layer_num=3, embedding_dim=64, share=False, normalize=True,
                 dropout=0.4, edge_dropout=0.2, pos_weight=1.0, use_sparse=True, act=nn.ReLU, cached=False, bias=True,
                 gnn_mode="gcnt", homo_metapaths=None, heter_metapaths=None, fl_alpha=0.25, fl_gamma=2, **kwargs):
        super(SGALDA, self).__init__()
        self.size_u = size_u
        self.size_v = size_v
        self.size_w = size_w
        self.lr = lr
        if homo_metapaths is None:
            homo_metapaths = "uu,vv,uvu,vuv,uwu,vwv"
        if heter_metapaths is None:
            heter_metapaths = "uv,uwv,uvwv"
        self.homo_metapaths = homo_metapaths.split(",")
        self.heter_metapaths = heter_metapaths.split(",")
        self.metapaths = np.concatenate((self.homo_metapaths, self.heter_metapaths))
        self.mp_num = len(self.metapaths)
        self.num_nodes = size_u + size_v + size_w
        self.in_dim = self.num_nodes
        self.share = share
        self.use_sparse = use_sparse
        self.embedding_dim = embedding_dim
        self.layer_num = layer_num
        self.dropout = nn.Dropout(dropout)
        self.edge_dropout = EdgeDropout(keep_prob=1 - edge_dropout)
        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        # self.loss_fn = partial(self.bce_loss_fn, pos_weight=self.pos_weight)
        # self.loss_fn = partial(self.mse_loss_fn, pos_weight=self.pos_weight)
        self.loss_fn = partial(self.focal_loss_fn, alpha=fl_alpha, gamma=fl_gamma)
        # self.loss_fn = partial(self.focal_loss, alpha=fl_alpha, gamma=fl_gamma)
        # self.loss_fn = FocalLoss(gamma=fl_gamma, alpha=fl_alpha)
        # self.loss_fn = self.imc_loss
        # self.loss_fn = partial(self.imc_loss, weight_decay=1e-5)
        # self.loss_fn = self.ce_loss
        # self.loss_fn = partial(self.ce_loss, pos_weight=self.pos_weight)
        self.short_embedding = nn.Linear(in_features=self.in_dim, out_features=self.embedding_dim)

        intra_encoder = [ShareGCN(size_u=size_u, size_v=size_v, size_w=size_w, in_channels=self.in_dim,
                                  out_channels=embedding_dim, share=False,
                                  dropout=dropout, act=act, bias=bias,
                                  normalize=normalize, cached=cached, gnn_mode="gcn")]
        inter_encoder = [ShareGCN(size_u=size_u, size_v=size_v, size_w=size_w, in_channels=self.in_dim,
                                  out_channels=embedding_dim,
                                  dropout=dropout, act=act, bias=bias,
                                  normalize=normalize, cached=cached, gnn_mode=gnn_mode)]
        for layer in range(1, layer_num):
            intra_encoder.append(ShareGCN(size_u=size_u, size_v=size_v, size_w=size_w, in_channels=embedding_dim,
                                          out_channels=embedding_dim, share=False,
                                          dropout=dropout, act=act, bias=bias,
                                          normalize=normalize, cached=cached, gnn_mode="gcn"))

            inter_encoder.append(ShareGCN(size_u=size_u, size_v=size_v, size_w=size_w, in_channels=embedding_dim,
                                          out_channels=embedding_dim,
                                          dropout=dropout, act=act, bias=bias,
                                          normalize=normalize, cached=cached, gnn_mode=gnn_mode))
        self.intra_encoders = nn.ModuleList(intra_encoder)
        self.inter_encoders = nn.ModuleList(inter_encoder)

        self.homo_encoder = MetaGCN(size_u=size_u, size_v=size_v, size_w=size_w, in_channels=self.in_dim,
                                  out_channels=embedding_dim, metapaths=self.homo_metapaths,
                                  dropout=dropout, act=act, bias=bias,
                                  normalize=normalize, cached=cached, gnn_mode="gcn")
        self.heter_encoder = MetaGCN(size_u=size_u, size_v=size_v, size_w=size_w, in_channels=self.in_dim,
                                    out_channels=embedding_dim, metapaths=self.heter_metapaths,
                                    dropout=dropout, act=act, bias=bias,
                                    normalize=normalize, cached=cached, gnn_mode="gcn")
        self.attention = nn.Parameter(torch.ones(layer_num, 1, 1) / layer_num)
        self.mp_attention = AttentionLayer(in_dim=self.embedding_dim)
        self.global_attention = AttentionLayer(in_dim=self.embedding_dim)
        self.act = act(inplace=True) if act else nn.Identity()
        # self.decoder = InnerProductDecoder(size_u=size_u, size_v=size_v, input_dim=0, dropout=dropout)
        self.decoder = BiLinearDecoder(size_u=size_u, size_v=size_v, input_dim=embedding_dim, dropout=dropout, activation=lambda x: x)
        # self.decoder = IMCDecoder(size_u=size_u, size_v=size_v, input_dim=embedding_dim, dropout=dropout)
        self.save_hyperparameters()

    def step(self, batch: FullGraphData):
        x = batch.embedding
        u_edge_index, u_edge_weight = batch.u_edge[:2]
        v_edge_index, v_edge_weight = batch.v_edge[:2]
        w_edge_index, w_edge_weight = batch.w_edge[:2]
        uv_edge_index, uv_edge_weight = batch.uv_edge[:2]
        vu_edge_index, vu_edge_weight = batch.vu_edge[:2]
        uw_edge_index, uw_edge_weight = batch.uw_edge[:2]
        wu_edge_index, wu_edge_weight = batch.wu_edge[:2]
        vw_edge_index, vw_edge_weight = batch.vw_edge[:2]
        wv_edge_index, wv_edge_weight = batch.wv_edge[:2]

        label = batch.label.reshape(-1)
        graph_out = self.forward(x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight, w_edge_index,
                                     w_edge_weight,
                                     uv_edge_index, uv_edge_weight, vu_edge_index, vu_edge_weight, uw_edge_index,
                                     uw_edge_weight,
                                     wu_edge_index, wu_edge_weight, vw_edge_index, vw_edge_weight, wv_edge_index,
                                     wv_edge_weight)    # 邻域特征提取
        graph_outs = [graph_out]
        homo_edge_index_dict = {}
        homo_edge_weight_dict = {}
        heter_edge_index_dict = {}
        heter_edge_weight_dict = {}
        if "uu" in self.metapaths:
            homo_edge_index_dict['uu'] = u_edge_index
            homo_edge_weight_dict['uu'] = u_edge_weight
        if "vv" in self.metapaths:
            homo_edge_index_dict['vv'] = v_edge_index
            homo_edge_weight_dict['vv'] = v_edge_weight
        if "ww" in self.metapaths:
            homo_edge_index_dict['ww'] = w_edge_index
            homo_edge_weight_dict['ww'] = w_edge_weight
        if "uvu" in self.metapaths:
            uvu_edge_index, uvu_edge_weight = batch.uvu_edge[:2]
            homo_edge_index_dict['uvu'] = uvu_edge_index
            homo_edge_weight_dict['uvu'] = uvu_edge_weight
        if "vuv" in self.metapaths:
            vuv_edge_index, vuv_edge_weight = batch.vuv_edge[:2]
            homo_edge_index_dict['vuv'] = vuv_edge_index
            homo_edge_weight_dict['vuv'] = vuv_edge_weight
        if "uwu" in self.metapaths:
            uwu_edge_index, uwu_edge_weight = batch.uwu_edge[:2]
            homo_edge_index_dict['uwu'] = uwu_edge_index
            homo_edge_weight_dict['uwu'] = uwu_edge_weight
        if "vwv" in self.metapaths:
            vwv_edge_index, vwv_edge_weight = batch.vwv_edge[:2]
            homo_edge_index_dict['vwv'] = vwv_edge_index
            homo_edge_weight_dict['vwv'] = vwv_edge_weight

        if "uv" in self.metapaths:
            heter_edge_index_dict["uv"] = torch.cat([uv_edge_index, vu_edge_index], dim=1)
            heter_edge_weight_dict["uv"] = torch.cat([uv_edge_weight, vu_edge_weight], dim=0)
        if "uw" in self.metapaths:
            heter_edge_index_dict["uw"] = torch.cat([uw_edge_index, wu_edge_index], dim=1)
            heter_edge_weight_dict["uw"] = torch.cat([uw_edge_weight, wu_edge_weight], dim=0)
        if "vw" in self.metapaths:
            heter_edge_index_dict["vw"] = torch.cat([vw_edge_index, wv_edge_index], dim=1)
            heter_edge_weight_dict["vw"] = torch.cat([vw_edge_weight, wv_edge_weight], dim=0)
        if "uwv" in self.metapaths:
            uwv_edge_index, uwv_edge_weight = batch.uwv_edge[:2]
            vwu_edge_index, vwu_edge_weight = batch.vwu_edge[:2]
            heter_edge_index_dict["uwv"] = torch.cat([uwv_edge_index, vwu_edge_index], dim=1)
            heter_edge_weight_dict["uwv"] = torch.cat([uwv_edge_weight, vwu_edge_weight], dim=0)
        if "uvwv" in self.metapaths:
            uvwv_edge_index, uvwv_edge_weight = batch.uvwv_edge[:2]
            vwvu_edge_index, vwvu_edge_weight = batch.vwvu_edge[:2]
            heter_edge_index_dict["uvwv"] = torch.cat([uvwv_edge_index, vwvu_edge_index], dim=1)
            heter_edge_weight_dict["uvwv"] = torch.cat([uvwv_edge_weight, vwvu_edge_weight], dim=0)

        mp_outs = self.forward_mp(x, homo_edge_index_dict, homo_edge_weight_dict,
                                 heter_edge_index_dict, heter_edge_weight_dict) # 语义特征提取
        mp_attention = torch.stack([self.mp_attention(out) for out in mp_outs])
        mp_attention = torch.softmax(mp_attention, dim=0)
        mp_out = torch.sum(mp_outs * mp_attention, dim=0)   # 语义注意力机制融合语义特征

        # 全局注意力机制融合邻域特征和语义特征
        graph_outs.append(mp_out)
        graph_outs = torch.stack(graph_outs)
        global_attention = torch.stack([self.global_attention(out) for out in graph_outs])
        global_attention = torch.softmax(global_attention, dim=0)
        x = torch.sum(graph_outs * global_attention, dim=0)
        # x = torch.cat(graph_outs, dim=1)
        # predict, u, v, w = self.decoder(x)
        # x = mp_out
        logits, u, v, w = self.decoder(x)

        if not self.training:
            # predict = predict[batch.valid_mask.reshape(*predict.shape)]
            logits = logits[batch.valid_mask.reshape(-1)]
            label = label[batch.valid_mask.reshape(-1)]
        # ans = self.loss_fn(predict=predict, label=label)
        ans = self.loss_fn(logits=logits, label=label)
        # if self.training:
        #     self.log('my_loss', ans, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # ans["predict"] = predict.reshape(-1)
        ans["predict"] = F.softmax(logits, dim=1)[:, 1].reshape(-1)
        ans["label"] = label.reshape(-1)
        return ans

    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight, w_edge_index, w_edge_weight,
                uv_edge_index, uv_edge_weight, vu_edge_index, vu_edge_weight, uw_edge_index, uw_edge_weight,
                wu_edge_index, wu_edge_weight, vw_edge_index, vw_edge_weight, wv_edge_index, wv_edge_weight):
        inter_edge_indies = [[uv_edge_index, vu_edge_index], [uw_edge_index, wu_edge_index],
                             [vw_edge_index, wv_edge_index]]
        inter_edge_weights = [[uv_edge_weight, vu_edge_weight], [uw_edge_weight, wu_edge_weight],
                              [vw_edge_weight, wv_edge_weight]]

        # x = self.attr_attention(x)   # 1147 * embedding_dim
        short_embedding = self.short_embedding(x)
        layer_out = [short_embedding]
        for inter_encoder, intra_encoder in zip(self.inter_encoders, self.intra_encoders):
            intra_feature = intra_encoder(x, u_edge_index=u_edge_index, u_edge_weight=u_edge_weight,
                                          v_edge_index=v_edge_index, v_edge_weight=v_edge_weight,
                                          w_edge_index=w_edge_index, w_edge_weight=w_edge_weight)
            inter_feature = torch.zeros_like(intra_feature)
            for inter_edge_index, inter_edge_weight in zip(inter_edge_indies, inter_edge_weights):
                inter_edge_index[0], inter_edge_weight[0] = self.edge_dropout(inter_edge_index[0], inter_edge_weight[0])
                inter_edge_index[1], inter_edge_weight[1] = self.edge_dropout(inter_edge_index[1], inter_edge_weight[1])
                inter_feature += inter_encoder(x, u_edge_index=inter_edge_index[0], u_edge_weight=inter_edge_weight[0],
                                               v_edge_index=inter_edge_index[1], v_edge_weight=inter_edge_weight[1])
            x = intra_feature + inter_feature + layer_out[-1]
            layer_out.append(x)
        x = torch.stack(layer_out[1:])
        attention = torch.softmax(self.attention, dim=0)
        x = torch.sum(x * attention, dim=0)
        return x

    def forward_mp(self, x, homo_edge_index_dict, homo_edge_weight_dict, heter_edge_index_dict, heter_edge_weight_dict):
        # x = self.short_embedding(x)
        homo_features, _ = self.homo_encoder(x, homo_edge_index_dict, homo_edge_weight_dict) # def forward(self, x, edge_index_dict, edge_weight_dict):
        heter_features, _ = self.heter_encoder(x, heter_edge_index_dict, heter_edge_weight_dict)
        homo_outs = torch.stack(homo_features)
        heter_outs = torch.stack(heter_features)
        mp_outs = torch.concat((homo_outs, heter_outs))
        return mp_outs

    def training_step(self, batch, batch_idx=None):
        return self.step(batch)

    def validation_step(self, batch, batch_idx=None):
        return self.step(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=.1 * self.lr, weight_decay=1e-5)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1 * self.lr, max_lr=self.lr,
                                                   gamma=0.995, mode="exp_range", step_size_up=20,
                                                   cycle_momentum=False)
        return [optimizer], [lr_scheduler]

    # def RGD(X, omega, rank, A, B, verbose=True, reg_lambda=0.5, init_option=INIT_WITH_SVD, init_U=None, init_V=None,
    # stop_relRes=1e-14, stop_relDiff = -1, stop_relResDiff = -1):


@MODEL_REGISTRY.register()
class HGCNLDA(BaseModel):
    DATASET_TYPE = "FullGraphDataset"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("HGCNLDA model config")
        parser.add_argument("--embedding_dim", default=3, type=int)
        parser.add_argument("--layer_num", default=1, type=int)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument("--bias", type=bool, default=True)
        parser.add_argument("--add_self_loops", type=bool, default=True)
        # parser.add_argument("--fl_alpha", type=float, default=0.25)
        # parser.add_argument("--fl_gamma", type=float, default=2)
        return parent_parser

    def __init__(self, size_u, size_v, size_w, lr=0.05, layer_num=3, embedding_dim=64, share=False, normalize=True,
                 dropout=0.4, edge_dropout=0.2, pos_weight=1.0, use_sparse=True, act=nn.ReLU, cached=False, bias=True,
                 gnn_mode="gcnt", fl_alpha=0.25, fl_gamma=2, **kwargs):
        super(HGCNLDA, self).__init__()
        self.size_u = size_u
        self.size_v = size_v
        self.size_w = size_w
        self.lr = lr
        self.num_nodes = size_u + size_v + size_w
        self.in_dim = self.num_nodes
        self.share = share
        self.use_sparse = use_sparse
        self.embedding_dim = embedding_dim
        self.layer_num = layer_num
        self.dropout = nn.Dropout(dropout)
        self.edge_dropout = EdgeDropout(keep_prob=1 - edge_dropout)
        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        # self.loss_fn = partial(self.bce_loss_fn, pos_weight=self.pos_weight)
        # self.loss_fn = partial(self.mse_loss_fn, pos_weight=self.pos_weight)
        # self.loss_fn = partial(self.focal_loss_fn, alpha=fl_alpha, gamma=fl_gamma)
        # self.loss_fn = partial(self.focal_loss, alpha=fl_alpha, gamma=fl_gamma)
        # self.loss_fn = FocalLoss(gamma=fl_gamma, alpha=fl_alpha)
        # self.loss_fn = self.imc_loss
        # self.loss_fn = partial(self.imc_loss, weight_decay=1e-5)
        # self.loss_fn = self.ce_loss
        # self.loss_fn = partial(self.ce_loss, pos_weight=self.pos_weight)
        self.loss_fn = self.ce_loss
        self.norm = nn.LayerNorm(embedding_dim)
        self.short_embedding = nn.Linear(in_features=self.in_dim, out_features=self.embedding_dim)

        intra_encoder = [ShareGCN(size_u=size_u, size_v=size_v, size_w=size_w, in_channels=self.in_dim,
                                  out_channels=embedding_dim, share=False,
                                  dropout=dropout, act=act, bias=bias,
                                  normalize=normalize, cached=cached, gnn_mode="gcn")]
        inter_encoder = [ShareGCN(size_u=size_u, size_v=size_v, size_w=size_w, in_channels=self.in_dim,
                                  out_channels=embedding_dim,
                                  dropout=dropout, act=act, bias=bias,
                                  normalize=normalize, cached=cached, gnn_mode=gnn_mode)]
        attn_layer = [AttentionLayer(in_dim=self.embedding_dim)]
        for layer in range(1, layer_num):
            intra_encoder.append(ShareGCN(size_u=size_u, size_v=size_v, size_w=size_w, in_channels=embedding_dim,
                                          out_channels=embedding_dim, share=False,
                                          dropout=dropout, act=act, bias=bias,
                                          normalize=normalize, cached=cached, gnn_mode="gcn"))

            inter_encoder.append(ShareGCN(size_u=size_u, size_v=size_v, size_w=size_w, in_channels=embedding_dim,
                                          out_channels=embedding_dim,
                                          dropout=dropout, act=act, bias=bias,
                                          normalize=normalize, cached=cached, gnn_mode=gnn_mode))
            attn_layer.append(AttentionLayer(in_dim=self.embedding_dim))
        self.intra_encoders = nn.ModuleList(intra_encoder)
        self.inter_encoders = nn.ModuleList(inter_encoder)
        self.attn_layers = nn.ModuleList(attn_layer)
        # self.attention = nn.Parameter(torch.ones(2, 1, 1) / 2)
        # self.cat_attention = AttentionLayer(in_dim=self.embedding_dim)
        # self.mp_attention = AttentionLayer(in_dim=self.embedding_dim)
        # self.global_attention = AttentionLayer(in_dim=self.embedding_dim)
        self.inputLinear = nn.Linear(2*embedding_dim, embedding_dim)
        self.act = act(inplace=True) if act else nn.Identity()
        # self.decoder = InnerProductDecoder(size_u=size_u, size_v=size_v, input_dim=0, dropout=dropout)
        self.decoder = BiLinearDecoder(size_u=size_u, size_v=size_v, input_dim=embedding_dim, dropout=dropout, activation=lambda x: x, num_weights=2)
        # self.decoder = IMCDecoder(size_u=size_u, size_v=size_v, input_dim=embedding_dim, dropout=dropout)
        self.save_hyperparameters()

    def step(self, batch: FullGraphData):
        x = batch.embedding
        u_edge_index, u_edge_weight = batch.u_edge[:2]
        v_edge_index, v_edge_weight = batch.v_edge[:2]
        w_edge_index, w_edge_weight = batch.w_edge[:2]
        uv_edge_index, uv_edge_weight = batch.uv_edge[:2]
        vu_edge_index, vu_edge_weight = batch.vu_edge[:2]
        uw_edge_index, uw_edge_weight = batch.uw_edge[:2]
        wu_edge_index, wu_edge_weight = batch.wu_edge[:2]
        vw_edge_index, vw_edge_weight = batch.vw_edge[:2]
        wv_edge_index, wv_edge_weight = batch.wv_edge[:2]

        label = batch.label.reshape(-1)
        # label = batch.label
        graph_out = self.forward(x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight, w_edge_index,
                                     w_edge_weight,
                                     uv_edge_index, uv_edge_weight, vu_edge_index, vu_edge_weight, uw_edge_index,
                                     uw_edge_weight,
                                     wu_edge_index, wu_edge_weight, vw_edge_index, vw_edge_weight, wv_edge_index,
                                     wv_edge_weight)    # Neighborhood feature extraction

        logits, u, v, w = self.decoder(graph_out)
        # predict, u, v, w = self.decoder(graph_out)

        if not self.training:
            # predict = predict[batch.valid_mask.reshape(*predict.shape)]
            # label = label[batch.valid_mask]
            logits = logits[batch.valid_mask.reshape(-1)]
            label = label[batch.valid_mask.reshape(-1)]
        # ans = self.loss_fn(predict=predict, label=label.float())
        ans = self.loss_fn(logits=logits, label=label)
        if self.training:
            self.log('my_loss', ans, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # ans["predict"] = predict.reshape(-1)
        ans["predict"] = F.softmax(logits, dim=1)[:, 1].reshape(-1)
        ans["label"] = label.reshape(-1)
        ans['embedding'] = graph_out
        return ans

    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight, w_edge_index, w_edge_weight,
                uv_edge_index, uv_edge_weight, vu_edge_index, vu_edge_weight, uw_edge_index, uw_edge_weight,
                wu_edge_index, wu_edge_weight, vw_edge_index, vw_edge_weight, wv_edge_index, wv_edge_weight):
        inter_edge_indies = [[uv_edge_index, vu_edge_index], [uw_edge_index, wu_edge_index],
                             [vw_edge_index, wv_edge_index]]
        inter_edge_weights = [[uv_edge_weight, vu_edge_weight], [uw_edge_weight, wu_edge_weight],
                              [vw_edge_weight, wv_edge_weight]]

        # x = self.attr_attention(x)   # 1147 * embedding_dim
        short_embedding = self.short_embedding(x)
        # layer_out_a = [short_embedding]
        # layer_out_c = [short_embedding]
        layer_out = []
        # last_layer_out = torch.zeros_like(short_embedding)
        last_layer_out = short_embedding
        for inter_encoder, intra_encoder, attn_layer in zip(self.inter_encoders, self.intra_encoders, self.attn_layers):
            intra_feature = intra_encoder(x, u_edge_index=u_edge_index, u_edge_weight=u_edge_weight,
                                          v_edge_index=v_edge_index, v_edge_weight=v_edge_weight,
                                          w_edge_index=w_edge_index, w_edge_weight=w_edge_weight)
            inter_feature = torch.zeros_like(intra_feature)
            for inter_edge_index, inter_edge_weight in zip(inter_edge_indies, inter_edge_weights):
                inter_edge_index[0], inter_edge_weight[0] = self.edge_dropout(inter_edge_index[0], inter_edge_weight[0])
                inter_edge_index[1], inter_edge_weight[1] = self.edge_dropout(inter_edge_index[1], inter_edge_weight[1])
                inter_feature += inter_encoder(x, u_edge_index=inter_edge_index[0], u_edge_weight=inter_edge_weight[0],
                                               v_edge_index=inter_edge_index[1], v_edge_weight=inter_edge_weight[1])
            # x_a = intra_feature + inter_feature + layer_out_a[-1]
            x_a = intra_feature + inter_feature
            # x_c = torch.cat([intra_feature, inter_feature, layer_out_c[-1]], dim=1)
            x_c = torch.cat([intra_feature, inter_feature], dim=1)
            x_c = self.inputLinear(x_c)
            layer_out.append(x_a)
            layer_out.append(x_c)
            layer_outs = torch.stack(layer_out)
            c_attention = torch.stack([attn_layer(out) for out in layer_outs])
            c_attention = torch.softmax(c_attention, dim=0)
            x = torch.sum(layer_outs * c_attention, dim=0)
            x = self.norm(x + last_layer_out)
            last_layer_out = x
        # x = torch.stack(layer_out[:])
        # attention = torch.softmax(self.attention, dim=0)
        # x = torch.sum(x * attention, dim=0)

        # layer_out = torch.stack(layer_out)
        # c_attention = torch.stack([self.cat_attention(out) for out in layer_out])
        # c_attention = torch.softmax(c_attention, dim=0)
        # x = torch.sum(layer_out * c_attention, dim=0)
        return x

    def training_step(self, batch, batch_idx=None):
        return self.step(batch)

    def validation_step(self, batch, batch_idx=None):
        return self.step(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=.1 * self.lr, weight_decay=1e-5)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1 * self.lr, max_lr=self.lr,
                                                   gamma=0.995, mode="exp_range", step_size_up=20,
                                                   cycle_momentum=False)
        return [optimizer], [lr_scheduler]