import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, List

from mmdet.models.utils.transformer import inverse_sigmoid
from ..structures import Instances

# 查询交互模块 20240505
# 核心类是 QueryInteractionModule，继承自 QueryInteractionBase
# QueryInteractionModule 依次执行 _select_active_tracks 和 _update_track_embedding
# _select_active_tracks 用于选择激活的track，_update_track_embedding 用于更新track的embedding
# _select_active_tracks 会根据训练状态，随机丢弃一部分track，然后添加一部分False Positive track



def random_drop_tracks(track_instances: Instances, drop_probability: float) -> Instances:
    if drop_probability > 0 and len(track_instances) > 0:
        keep_idxes = torch.rand_like(track_instances.scores) > drop_probability
        track_instances = track_instances[keep_idxes]
    return track_instances


class QueryInteractionBase(nn.Module):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.args = args
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        self._reset_parameters()

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        raise NotImplementedError()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _select_active_tracks(self, data: dict) -> Instances:
        raise NotImplementedError()

    def _update_track_embedding(self, track_instances):
        raise NotImplementedError()


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = F.relu
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt


class QueryInteractionModule(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args['random_drop']
        self.fp_ratio = args['fp_ratio']
        self.update_query_pos = args['update_query_pos']

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args['merger_dropout']

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args['update_query_pos']:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        # Instances: query, output_embedding, pred_boxes, obj_idxes, iou, scores
        # query: [num_tracks, 512], output_embedding: [num_tracks, 256]
        if len(track_instances) == 0:
            return track_instances
        dim = track_instances.query.shape[1]  # 512
        out_embed = track_instances.output_embedding
        query_pos = track_instances.query[:, :dim // 2]  # 256
        query_feat = track_instances.query[:, dim // 2:]  # 256
        q = k = query_pos + out_embed

        # attention
        tgt = out_embed
        # [:,None] is used to add a new dimension, equivalent to unsqueeze(-1)
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            # ffn: linear_pos2
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query[:, :dim // 2] = query_pos

        # add and norm
        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query[:, dim // 2:] = query_feat
        # track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :2].detach().clone())
        # update ref_pts using track_instances.pred_boxes
        return track_instances

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances,
                       active_track_instances: Instances) -> Instances:
        '''
        self.fp_ratio is used to control num(add_fp) / num(active)

        active_track_instances: Instances, track_instances[obj_idxes >= 0] + dropout
        '''
        inactive_instances = track_instances[track_instances.obj_idxes < 0]

        # add fp for each active track in a specific probability.
        fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[
            torch.bernoulli(fp_prob).bool()]  # 每个active track有fp_prob的概率被选中
        num_fp = len(selected_active_track_instances)

        if len(inactive_instances) > 0 and num_fp > 0:
            # inactive 的数量必须大于等于 selected_active 的数量
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                # randomly select num_fp from inactive_instances
                # fp_indexes = np.random.permutation(len(inactive_instances))
                # fp_indexes = fp_indexes[:num_fp]
                # fp_track_instances = inactive_instances[fp_indexes]

                # v2: select the fps with top scores rather than random selection
                fp_indexes = torch.argsort(inactive_instances.scores)[-num_fp:]
                fp_track_instances = inactive_instances[fp_indexes]

            merged_track_instances = Instances.cat([active_track_instances,
                                                    fp_track_instances])
            return merged_track_instances

        return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            # active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.iou > 0.5)
            # obj取值范围 ？
            active_idxes = (track_instances.obj_idxes >= 0)
            active_track_instances = track_instances[active_idxes]
            # set -2 instead of -1 to ensure that these tracks will not be selected in matching.

            # 训练时，随机丢弃一部分track，然后添加一部分fp track
            active_track_instances = self._random_drop_tracks(active_track_instances)
            if self.fp_ratio > 0:
                active_track_instances = self._add_fp_tracks(track_instances, active_track_instances)
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def forward(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances)
        return active_track_instances

        # init_track_instances: Instances = data['init_track_instances']
        # merged_track_instances = Instances.cat([init_track_instances, active_track_instances])
        # return merged_track_instances


def build_qim(args, dim_in, hidden_dim, dim_out):
    qim_type = args['qim_type']
    interaction_layers = {
        'QIMBase': QueryInteractionModule,
    }
    assert qim_type in interaction_layers, 'invalid query interaction layer: {}'.format(qim_type)

    # 调用 QIM constructor
    return interaction_layers[qim_type](args, dim_in, hidden_dim, dim_out)
