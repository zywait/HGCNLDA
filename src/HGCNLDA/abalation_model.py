import numpy as np
from torch import optim
from functools import partial
from ..model_help import BaseModel
from .dataset import FullGraphData
from .. import MODEL_REGISTRY
import torch
import torch.nn as nn
from ..layers import AttentionLayer
from .model import EdgeDropout, ShareGCN, MetaGCN, InnerProductDecoder, BiLinearDecoder
import torch.nn.functional as F

# blinear + celoss
@MODEL_REGISTRY.register()
class HGCNLDA_ho(BaseModel):
    DATASET_TYPE = "FullGraphDataset"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("HGCNLDA_ho model config")
        parser.add_argument("--embedding_dim", default=64, type=int)
        parser.add_argument("--layer_num", default=1, type=int)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument("--bias", type=bool, default=True)
        parser.add_argument("--add_self_loops", type=bool, default=True)
        return parent_parser

    def __init__(self, size_u, size_v, size_w, lr=0.05, layer_num=3, embedding_dim=64, share=False, normalize=True,
                 dropout=0.4, edge_dropout=0.2, pos_weight=1.0, use_sparse=True, act=nn.ReLU, cached=False, bias=True,
                 gnn_mode="gcnt", fl_alpha=0.25, fl_gamma=2, **kwargs):
        super(HGCNLDA_ho, self).__init__()
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
        self.loss_fn = self.ce_loss
        self.short_embedding = nn.Linear(in_features=self.in_dim, out_features=self.embedding_dim)

        intra_encoder = [ShareGCN(size_u=size_u, size_v=size_v, size_w=size_w, in_channels=self.in_dim,
                                  out_channels=embedding_dim, share=False,
                                  dropout=dropout, act=act, bias=bias,
                                  normalize=normalize, cached=cached, gnn_mode="gcn")]

        for layer in range(1, layer_num):
            intra_encoder.append(ShareGCN(size_u=size_u, size_v=size_v, size_w=size_w, in_channels=embedding_dim,
                                          out_channels=embedding_dim, share=False,
                                          dropout=dropout, act=act, bias=bias,
                                          normalize=normalize, cached=cached, gnn_mode="gcn"))

        self.intra_encoders = nn.ModuleList(intra_encoder)

        self.inputLinear = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.act = act(inplace=True) if act else nn.Identity()
        # self.decoder = InnerProductDecoder(size_u=size_u, size_v=size_v, input_dim=0, dropout=dropout)
        self.decoder = BiLinearDecoder(size_u=size_u, size_v=size_v, input_dim=embedding_dim, dropout=dropout,
                                       activation=lambda x: x, num_weights=2)
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
                                     wv_edge_weight)    # Neighborhood feature extraction

        # predict, u, v, w = self.decoder(graph_out)
        logits, u, v, w = self.decoder(graph_out)

        if not self.training:
            # predict = predict[batch.valid_mask.reshape(*predict.shape)]
            # label = label[batch.valid_mask]
            logits = logits[batch.valid_mask.reshape(-1)]
            label = label[batch.valid_mask.reshape(-1)]
        # ans = self.loss_fn(predict=predict, label=label.float())
        # ans["predict"] = predict.reshape(-1)
        # ans["label"] = label.reshape(-1)
        ans = self.loss_fn(logits=logits, label=label)
        ans["predict"] = F.softmax(logits, dim=1)[:, 1].reshape(-1)
        ans["label"] = label.reshape(-1)
        return ans

    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight, w_edge_index, w_edge_weight,
                uv_edge_index, uv_edge_weight, vu_edge_index, vu_edge_weight, uw_edge_index, uw_edge_weight,
                wu_edge_index, wu_edge_weight, vw_edge_index, vw_edge_weight, wv_edge_index, wv_edge_weight):
        inter_edge_indies = [[uv_edge_index, vu_edge_index], [uw_edge_index, wu_edge_index],
                             [vw_edge_index, wv_edge_index]]


        short_embedding = self.short_embedding(x)
        layer_out = [torch.zeros_like(short_embedding)]
        for intra_encoder in self.intra_encoders:
            intra_feature = intra_encoder(x, u_edge_index=u_edge_index, u_edge_weight=u_edge_weight,
                                          v_edge_index=v_edge_index, v_edge_weight=v_edge_weight,
                                          w_edge_index=w_edge_index, w_edge_weight=w_edge_weight)
            output = self.inputLinear(intra_feature)
            output = self.norm(layer_out[-1] + output)
            layer_out.append(output)
        return layer_out[-1]

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


# blinear + celoss
@MODEL_REGISTRY.register()
class HGCNLDA_he(BaseModel):
    DATASET_TYPE = "FullGraphDataset"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("HGCNLDA_he model config")
        parser.add_argument("--embedding_dim", default=256, type=int)
        parser.add_argument("--layer_num", default=1, type=int)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument("--bias", type=bool, default=True)
        parser.add_argument("--add_self_loops", type=bool, default=True)
        return parent_parser

    def __init__(self, size_u, size_v, size_w, lr=0.05, layer_num=3, embedding_dim=64, share=False, normalize=True,
                 dropout=0.4, edge_dropout=0.2, pos_weight=1.0, use_sparse=True, act=nn.ReLU, cached=False, bias=True,
                 gnn_mode="gcnt", fl_alpha=0.25, fl_gamma=2, **kwargs):
        super(HGCNLDA_he, self).__init__()
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
        self.loss_fn = self.ce_loss
        self.short_embedding = nn.Linear(in_features=self.in_dim, out_features=self.embedding_dim)

        inter_encoder = [ShareGCN(size_u=size_u, size_v=size_v, size_w=size_w, in_channels=self.in_dim,
                                  out_channels=embedding_dim,
                                  dropout=dropout, act=act, bias=bias,
                                  normalize=normalize, cached=cached, gnn_mode=gnn_mode)]
        attn_layer = [AttentionLayer(in_dim=self.embedding_dim)]
        for layer in range(1, layer_num):
            inter_encoder.append(ShareGCN(size_u=size_u, size_v=size_v, size_w=size_w, in_channels=embedding_dim,
                                          out_channels=embedding_dim,
                                          dropout=dropout, act=act, bias=bias,
                                          normalize=normalize, cached=cached, gnn_mode=gnn_mode))
            attn_layer.append(AttentionLayer(in_dim=self.embedding_dim))
        self.inter_encoders = nn.ModuleList(inter_encoder)
        self.attn_layers = nn.ModuleList(attn_layer)

        self.inputLinear = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.act = act(inplace=True) if act else nn.Identity()
        # self.decoder = InnerProductDecoder(size_u=size_u, size_v=size_v, input_dim=0, dropout=dropout)
        self.decoder = BiLinearDecoder(size_u=size_u, size_v=size_v, input_dim=embedding_dim, dropout=dropout,
                                       activation=lambda x: x, num_weights=2)
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

        # predict, u, v, w = self.decoder(graph_out)
        logits, u, v, w = self.decoder(graph_out)

        if not self.training:
            # predict = predict[batch.valid_mask.reshape(*predict.shape)]
            # label = label[batch.valid_mask]
            logits = logits[batch.valid_mask.reshape(-1)]
            label = label[batch.valid_mask.reshape(-1)]
        # ans = self.loss_fn(predict=predict, label=label.float())
        # ans["predict"] = predict.reshape(-1)
        # ans["label"] = label.reshape(-1)
        ans = self.loss_fn(logits=logits, label=label)
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

        short_embedding = self.short_embedding(x)
        layer_out = [short_embedding]
        for inter_encoder, attn_layer in zip(self.inter_encoders, self.attn_layers):
            inter_feature = torch.zeros_like(short_embedding)
            for inter_edge_index, inter_edge_weight in zip(inter_edge_indies, inter_edge_weights):
                inter_edge_index[0], inter_edge_weight[0] = self.edge_dropout(inter_edge_index[0], inter_edge_weight[0])
                inter_edge_index[1], inter_edge_weight[1] = self.edge_dropout(inter_edge_index[1], inter_edge_weight[1])
                inter_feature += inter_encoder(x, u_edge_index=inter_edge_index[0], u_edge_weight=inter_edge_weight[0],
                                               v_edge_index=inter_edge_index[1], v_edge_weight=inter_edge_weight[1])
            output = self.inputLinear(inter_feature)
            output = self.norm(layer_out[-1] + output)
            layer_out.append(output)
        return layer_out[-1]

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


@MODEL_REGISTRY.register()
class HGCNLDA_add(BaseModel):
    DATASET_TYPE = "FullGraphDataset"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("HGCNLDA_add model config")
        parser.add_argument("--embedding_dim", default=256, type=int)
        parser.add_argument("--layer_num", default=1, type=int)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument("--bias", type=bool, default=True)
        parser.add_argument("--add_self_loops", type=bool, default=True)
        return parent_parser

    def __init__(self, size_u, size_v, size_w, lr=0.05, layer_num=3, embedding_dim=64, share=False, normalize=True,
                 dropout=0.4, edge_dropout=0.2, pos_weight=1.0, use_sparse=True, act=nn.ReLU, cached=False, bias=True,
                 gnn_mode="gcnt", fl_alpha=0.25, fl_gamma=2, **kwargs):
        super(HGCNLDA_add, self).__init__()
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
        self.inputLinear = nn.Linear(embedding_dim, embedding_dim)
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
                                     wv_edge_weight)    # 邻域特征提取

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
        return ans

    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight, w_edge_index, w_edge_weight,
                uv_edge_index, uv_edge_weight, vu_edge_index, vu_edge_weight, uw_edge_index, uw_edge_weight,
                wu_edge_index, wu_edge_weight, vw_edge_index, vw_edge_weight, wv_edge_index, wv_edge_weight):
        inter_edge_indies = [[uv_edge_index, vu_edge_index], [uw_edge_index, wu_edge_index],
                             [vw_edge_index, wv_edge_index]]
        inter_edge_weights = [[uv_edge_weight, vu_edge_weight], [uw_edge_weight, wu_edge_weight],
                              [vw_edge_weight, wv_edge_weight]]

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
            feature_add = intra_feature + inter_feature
            output = self.inputLinear(feature_add)
            output = self.norm(layer_out[-1] + output)
            layer_out.append(output)
        return layer_out[-1]

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



@MODEL_REGISTRY.register()
class HGCNLDA_cat(BaseModel):
    DATASET_TYPE = "FullGraphDataset"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("HGCNLDA_cat model config")
        parser.add_argument("--embedding_dim", default=256, type=int)
        parser.add_argument("--layer_num", default=1, type=int)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument("--bias", type=bool, default=True)
        parser.add_argument("--add_self_loops", type=bool, default=True)
        return parent_parser

    def __init__(self, size_u, size_v, size_w, lr=0.05, layer_num=3, embedding_dim=64, share=False, normalize=True,
                 dropout=0.4, edge_dropout=0.2, pos_weight=1.0, use_sparse=True, act=nn.ReLU, cached=False, bias=True,
                 gnn_mode="gcnt", fl_alpha=0.25, fl_gamma=2, **kwargs):
        super(HGCNLDA_cat, self).__init__()
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
        return ans

    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight, w_edge_index, w_edge_weight,
                uv_edge_index, uv_edge_weight, vu_edge_index, vu_edge_weight, uw_edge_index, uw_edge_weight,
                wu_edge_index, wu_edge_weight, vw_edge_index, vw_edge_weight, wv_edge_index, wv_edge_weight):
        inter_edge_indies = [[uv_edge_index, vu_edge_index], [uw_edge_index, wu_edge_index],
                             [vw_edge_index, wv_edge_index]]
        inter_edge_weights = [[uv_edge_weight, vu_edge_weight], [uw_edge_weight, wu_edge_weight],
                              [vw_edge_weight, wv_edge_weight]]

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
            feature_cat = torch.cat([intra_feature, inter_feature], dim=1)
            output = self.inputLinear(feature_cat)
            output = self.norm(layer_out[-1] + output)
            layer_out.append(output)
        return layer_out[-1]

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



@MODEL_REGISTRY.register()
class HGCNLDA_ac(BaseModel):
    DATASET_TYPE = "FullGraphDataset"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("HGCNLDA_ac model config")
        parser.add_argument("--embedding_dim", default=256, type=int)
        parser.add_argument("--layer_num", default=1, type=int)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument("--bias", type=bool, default=True)
        parser.add_argument("--add_self_loops", type=bool, default=True)
        return parent_parser

    def __init__(self, size_u, size_v, size_w, lr=0.05, layer_num=3, embedding_dim=64, share=False, normalize=True,
                 dropout=0.4, edge_dropout=0.2, pos_weight=1.0, use_sparse=True, act=nn.ReLU, cached=False, bias=True,
                 gnn_mode="gcnt", fl_alpha=0.25, fl_gamma=2, **kwargs):
        super(HGCNLDA_ac, self).__init__()
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
        self.inputLinear = nn.Linear(2*embedding_dim, embedding_dim)
        self.outLinear = nn.Linear(embedding_dim, embedding_dim)
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
                                     wv_edge_weight)    # 邻域特征提取

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
        return ans

    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight, w_edge_index, w_edge_weight,
                uv_edge_index, uv_edge_weight, vu_edge_index, vu_edge_weight, uw_edge_index, uw_edge_weight,
                wu_edge_index, wu_edge_weight, vw_edge_index, vw_edge_weight, wv_edge_index, wv_edge_weight):
        inter_edge_indies = [[uv_edge_index, vu_edge_index], [uw_edge_index, wu_edge_index],
                             [vw_edge_index, wv_edge_index]]
        inter_edge_weights = [[uv_edge_weight, vu_edge_weight], [uw_edge_weight, wu_edge_weight],
                              [vw_edge_weight, wv_edge_weight]]

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
            x_a = intra_feature + inter_feature
            x_c = torch.cat([intra_feature, inter_feature], dim=1)
            x_c = self.inputLinear(x_c)
            output = self.outLinear(x_a + x_c)
            output = self.norm(layer_out[-1] + output)
            layer_out.append(output)
        return layer_out[-1]

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

# (weight)cep_loss, productDecoder
@MODEL_REGISTRY.register()
class HGCNLDA_wcep(BaseModel):
    DATASET_TYPE = "FullGraphDataset"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("HGCNLDA_wcep model config")
        parser.add_argument("--embedding_dim", default=128, type=int)
        parser.add_argument("--layer_num", default=1, type=int)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument("--bias", type=bool, default=True)
        parser.add_argument("--add_self_loops", type=bool, default=True)
        return parent_parser

    def __init__(self, size_u, size_v, size_w, lr=0.05, layer_num=3, embedding_dim=64, share=False, normalize=True,
                 dropout=0.4, edge_dropout=0.2, pos_weight=1.0, use_sparse=True, act=nn.ReLU, cached=False, bias=True,
                 gnn_mode="gcnt", fl_alpha=0.25, fl_gamma=2, **kwargs):
        super(HGCNLDA_wcep, self).__init__()
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
        self.loss_fn = partial(self.cep_loss, pos_weight=self.pos_weight)
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
        self.norm = nn.LayerNorm(embedding_dim)
        self.act = act(inplace=True) if act else nn.Identity()
        self.decoder = InnerProductDecoder(size_u=size_u, size_v=size_v, input_dim=0, dropout=dropout, act=lambda x: x)
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

        label = batch.label.float().reshape(-1)
        graph_out = self.forward(x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight, w_edge_index,
                                     w_edge_weight,
                                     uv_edge_index, uv_edge_weight, vu_edge_index, vu_edge_weight, uw_edge_index,
                                     uw_edge_weight,
                                     wu_edge_index, wu_edge_weight, vw_edge_index, vw_edge_weight, wv_edge_index,
                                     wv_edge_weight)    # 邻域特征提取

        logits, u, v, w = self.decoder(graph_out)
        logits = logits.reshape(-1)

        if not self.training:
            logits = logits[batch.valid_mask.reshape(-1)]
            label = label[batch.valid_mask.reshape(-1)]
        ans = self.loss_fn(logits=logits, label=label)
        ans["predict"] = F.softmax(logits, dim=0).reshape(-1)
        ans["label"] = label.reshape(-1)
        return ans

    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight, w_edge_index, w_edge_weight,
                uv_edge_index, uv_edge_weight, vu_edge_index, vu_edge_weight, uw_edge_index, uw_edge_weight,
                wu_edge_index, wu_edge_weight, vw_edge_index, vw_edge_weight, wv_edge_index, wv_edge_weight):
        inter_edge_indies = [[uv_edge_index, vu_edge_index], [uw_edge_index, wu_edge_index],
                             [vw_edge_index, wv_edge_index]]
        inter_edge_weights = [[uv_edge_weight, vu_edge_weight], [uw_edge_weight, wu_edge_weight],
                              [vw_edge_weight, wv_edge_weight]]

        short_embedding = self.short_embedding(x)
        layer_out = []
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
            x_a = intra_feature + inter_feature
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

# focal_loss, productDecoder
@MODEL_REGISTRY.register()
class HGCNLDA_fc(BaseModel):
    DATASET_TYPE = "FullGraphDataset"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("HGCNLDA_bd model config")
        parser.add_argument("--embedding_dim", default=64, type=int)
        parser.add_argument("--layer_num", default=1, type=int)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument("--bias", type=bool, default=True)
        parser.add_argument("--add_self_loops", type=bool, default=True)
        parser.add_argument("--fl_alpha", type=float, default=0.25)
        parser.add_argument("--fl_gamma", type=float, default=5)
        return parent_parser

    def __init__(self, size_u, size_v, size_w, lr=0.05, layer_num=3, embedding_dim=64, share=False, normalize=True,
                 dropout=0.4, edge_dropout=0.2, pos_weight=1.0, use_sparse=True, act=nn.ReLU, cached=False, bias=True,
                 gnn_mode="gcnt", fl_alpha=0.25, fl_gamma=2, **kwargs):
        super(HGCNLDA_fc, self).__init__()
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
        self.loss_fn = partial(self.focal_loss, alpha=fl_alpha, gamma=fl_gamma)
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
        self.norm = nn.LayerNorm(embedding_dim)
        self.act = act(inplace=True) if act else nn.Identity()
        self.decoder = InnerProductDecoder(size_u=size_u, size_v=size_v, input_dim=0, dropout=dropout)
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

        label = batch.label
        graph_out = self.forward(x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight, w_edge_index,
                                     w_edge_weight,
                                     uv_edge_index, uv_edge_weight, vu_edge_index, vu_edge_weight, uw_edge_index,
                                     uw_edge_weight,
                                     wu_edge_index, wu_edge_weight, vw_edge_index, vw_edge_weight, wv_edge_index,
                                     wv_edge_weight)    # 邻域特征提取

        predict, u, v, w = self.decoder(graph_out)

        if not self.training:
            predict = predict[batch.valid_mask.reshape(*predict.shape)]
            label = label[batch.valid_mask]
        ans = self.loss_fn(predict=predict, label=label.float())
        ans["predict"] = predict.reshape(-1)
        ans["label"] = label.reshape(-1)
        return ans

    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight, w_edge_index, w_edge_weight,
                uv_edge_index, uv_edge_weight, vu_edge_index, vu_edge_weight, uw_edge_index, uw_edge_weight,
                wu_edge_index, wu_edge_weight, vw_edge_index, vw_edge_weight, wv_edge_index, wv_edge_weight):
        inter_edge_indies = [[uv_edge_index, vu_edge_index], [uw_edge_index, wu_edge_index],
                             [vw_edge_index, wv_edge_index]]
        inter_edge_weights = [[uv_edge_weight, vu_edge_weight], [uw_edge_weight, wu_edge_weight],
                              [vw_edge_weight, wv_edge_weight]]

        short_embedding = self.short_embedding(x)
        layer_out = []
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


# mse_loss, productDecoder
@MODEL_REGISTRY.register()
class HGCNLDA_mse(BaseModel):
    DATASET_TYPE = "FullGraphDataset"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("HGCNLDA_bd model config")
        parser.add_argument("--embedding_dim", default=64, type=int)
        parser.add_argument("--layer_num", default=1, type=int)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument("--bias", type=bool, default=True)
        parser.add_argument("--add_self_loops", type=bool, default=True)
        return parent_parser

    def __init__(self, size_u, size_v, size_w, lr=0.05, layer_num=3, embedding_dim=64, share=False, normalize=True,
                 dropout=0.4, edge_dropout=0.2, pos_weight=1.0, use_sparse=True, act=nn.ReLU, cached=False, bias=True,
                 gnn_mode="gcnt", fl_alpha=0.25, fl_gamma=2, **kwargs):
        super(HGCNLDA_mse, self).__init__()
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
        self.loss_fn = partial(self.mse_loss_fn, pos_weight=self.pos_weight)
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
        self.norm = nn.LayerNorm(embedding_dim)
        self.act = act(inplace=True) if act else nn.Identity()
        self.decoder = InnerProductDecoder(size_u=size_u, size_v=size_v, input_dim=0, dropout=dropout)
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

        label = batch.label
        graph_out = self.forward(x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight, w_edge_index,
                                     w_edge_weight,
                                     uv_edge_index, uv_edge_weight, vu_edge_index, vu_edge_weight, uw_edge_index,
                                     uw_edge_weight,
                                     wu_edge_index, wu_edge_weight, vw_edge_index, vw_edge_weight, wv_edge_index,
                                     wv_edge_weight)    # 邻域特征提取

        predict, u, v, w = self.decoder(graph_out)

        if not self.training:
            predict = predict[batch.valid_mask.reshape(*predict.shape)]
            label = label[batch.valid_mask]
        ans = self.loss_fn(predict=predict, label=label.float())
        ans["predict"] = predict.reshape(-1)
        ans["label"] = label.reshape(-1)
        return ans

    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight, w_edge_index, w_edge_weight,
                uv_edge_index, uv_edge_weight, vu_edge_index, vu_edge_weight, uw_edge_index, uw_edge_weight,
                wu_edge_index, wu_edge_weight, vw_edge_index, vw_edge_weight, wv_edge_index, wv_edge_weight):
        inter_edge_indies = [[uv_edge_index, vu_edge_index], [uw_edge_index, wu_edge_index],
                             [vw_edge_index, wv_edge_index]]
        inter_edge_weights = [[uv_edge_weight, vu_edge_weight], [uw_edge_weight, wu_edge_weight],
                              [vw_edge_weight, wv_edge_weight]]

        short_embedding = self.short_embedding(x)
        layer_out = []
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


# cep_loss（without weight）, productDecoder
@MODEL_REGISTRY.register()
class HGCNLDA_cep(BaseModel):
    DATASET_TYPE = "FullGraphDataset"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("HGCNLDA_cep model config")
        parser.add_argument("--embedding_dim", default=64, type=int)
        parser.add_argument("--layer_num", default=1, type=int)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument("--bias", type=bool, default=True)
        parser.add_argument("--add_self_loops", type=bool, default=True)
        return parent_parser

    def __init__(self, size_u, size_v, size_w, lr=0.05, layer_num=3, embedding_dim=64, share=False, normalize=True,
                 dropout=0.4, edge_dropout=0.2, pos_weight=1.0, use_sparse=True, act=nn.ReLU, cached=False, bias=True,
                 gnn_mode="gcnt", fl_alpha=0.25, fl_gamma=2, **kwargs):
        super(HGCNLDA_cep, self).__init__()
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
        self.loss_fn = self.cep_loss
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
        self.norm = nn.LayerNorm(embedding_dim)
        self.act = act(inplace=True) if act else nn.Identity()
        self.decoder = InnerProductDecoder(size_u=size_u, size_v=size_v, input_dim=0, dropout=dropout, act=lambda x: x)
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

        label = batch.label.float().reshape(-1)
        graph_out = self.forward(x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight, w_edge_index,
                                     w_edge_weight,
                                     uv_edge_index, uv_edge_weight, vu_edge_index, vu_edge_weight, uw_edge_index,
                                     uw_edge_weight,
                                     wu_edge_index, wu_edge_weight, vw_edge_index, vw_edge_weight, wv_edge_index,
                                     wv_edge_weight)    # Neighborhood feature extraction

        logits, u, v, w = self.decoder(graph_out)
        logits = logits.reshape(-1)

        if not self.training:
            logits = logits[batch.valid_mask.reshape(-1)]
            label = label[batch.valid_mask.reshape(-1)]
        ans = self.loss_fn(logits=logits, label=label)
        ans["predict"] = F.softmax(logits, dim=0).reshape(-1)
        ans["label"] = label.reshape(-1)
        return ans

    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight, w_edge_index, w_edge_weight,
                uv_edge_index, uv_edge_weight, vu_edge_index, vu_edge_weight, uw_edge_index, uw_edge_weight,
                wu_edge_index, wu_edge_weight, vw_edge_index, vw_edge_weight, wv_edge_index, wv_edge_weight):
        inter_edge_indies = [[uv_edge_index, vu_edge_index], [uw_edge_index, wu_edge_index],
                             [vw_edge_index, wv_edge_index]]
        inter_edge_weights = [[uv_edge_weight, vu_edge_weight], [uw_edge_weight, wu_edge_weight],
                              [vw_edge_weight, wv_edge_weight]]

        short_embedding = self.short_embedding(x)
        layer_out = []
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
            x_a = intra_feature + inter_feature
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

# (weight)ce_loss, biDecoder
@MODEL_REGISTRY.register()
class HGCNLDA_wceb(BaseModel):
    DATASET_TYPE = "FullGraphDataset"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("HGCNLDA_wceb model config")
        parser.add_argument("--embedding_dim", default=128, type=int)
        parser.add_argument("--layer_num", default=1, type=int)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument("--bias", type=bool, default=True)
        parser.add_argument("--add_self_loops", type=bool, default=True)
        return parent_parser

    def __init__(self, size_u, size_v, size_w, lr=0.05, layer_num=3, embedding_dim=64, share=False, normalize=True,
                 dropout=0.4, edge_dropout=0.2, pos_weight=1.0, use_sparse=True, act=nn.ReLU, cached=False, bias=True,
                 gnn_mode="gcnt", fl_alpha=0.25, fl_gamma=2, **kwargs):
        super(HGCNLDA_wceb, self).__init__()
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
        self.loss_fn = partial(self.ce_loss, pos_weight=self.pos_weight)
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
        self.norm = nn.LayerNorm(embedding_dim)
        self.act = act(inplace=True) if act else nn.Identity()
        self.decoder = BiLinearDecoder(size_u=size_u, size_v=size_v, input_dim=embedding_dim, dropout=dropout,
                                       activation=lambda x: x, num_weights=2)
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
                                     wv_edge_weight)    # Neighborhood feature extraction

        logits, u, v, w = self.decoder(graph_out)

        if not self.training:
            logits = logits[batch.valid_mask.reshape(-1)]
            label = label[batch.valid_mask.reshape(-1)]
        ans = self.loss_fn(logits=logits, label=label)
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

        short_embedding = self.short_embedding(x)
        layer_out = []
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