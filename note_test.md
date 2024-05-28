# Test 


`tools.test.py` (1)
```python
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
# ... 
outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
```

`mmdet3d/apis/test.py` (2)
```python
with torch.no_grad():
    result = model(return_loss=False, rescale=True, **data)
```

`vip3d.py`  (3)
```python
@force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        else:
            if self.do_pred:
                self.predictor.decoder.do_eval = True
            return self.forward_test(**kwargs)
```

`vip3d.py`  (4)
```python
def forward_test(self, 
                 points = [[tensor(34720, 5)]], 
                 img = tensor(1,1,6,3,928,1600), 
                 radar= [tensor(1,100,14)],
                 img_metas= [dict(dict_keys(['filename', 'ori_shape', 'img_shape', 'lidar2img', 
                                            'pad_shape', 'scale_factor', 'box_mode_3d', 'box_type_3d', 
                                            'img_norm_cfg', 'sample_idx', 'pts_filename']))], 
                 timestamp= [tensor(1)],
                 l2g_r_mat=tensor(3,3), 
                 l2g_t=tensor(1,3),
                 instance_idx_2_labels=None, 
                 mapping=[dict( dict_keys(['cur_l2e_r': list[4], 'cur_l2e_t': list[0.94, 0.0, 1.84], 
                                           'cur_e2g_r': list[4], 'cur_e2g_t': list[732, 949, 0.0], 
                                           'r_index_2_rotation_and_transform', 
                                           'valid_pred' = False, 'instance_inds', 
                                           'lanes': list[50, np.array(num_pts, 2)], 
                                           'map_name':str = 'singapore-onenorth', 
                                           'timestamp': float = 1531281439.800013, 
                                           'timestamp_origin': int= 1531281439800013, 
                                            'sample_token': str = '30e55a3ec6184d8cb1944b39ba19d622', 
                                           'scene_id':str = 'scene-0268', 
                                            'same_scene'=False, 'index':int = 0]))],
                 gt_bboxes_3d=None list[list[:obj:`BaseInstance3DBoxes`]], 
                 gt_labels_3d=None list[list[torch.Tensor]]):
```

### `vip3d.py` :: `forward_test`  (5) 初始化 test_track_instances

```python
if self.test_track_instances is None:  # track agent 初始化
    track_instances = self._generate_empty_tracks()
    self.test_track_instances = track_instances
    self.timestamp = timestamp[0]
    # avoid repeated generation of empty tracks
```
- 一个 track(跟踪) 是 instances 的集合 (300个)，
- 每个 instance 是一个实例， 对应一个 query
- 每个 query 为 512 维 ： 256维的位置向量 + 256维的特征向量 TODO: check


`vip3d.py` :: _generate_empty_tracks  (6)
- 初始化一个 Instances 类
```python
track_instances = Instances((1, 1))
```
- 用query前半部分计算 bbox 尺寸, 添加到track_instances.pred_boxes
- 用query前半部分计算 3d 参考点， 添加到track_instances.ref_pts
- 添加 query 字段
```python
box_sizes = self.bbox_size_fc(query[..., :dim // 2])  # (Nq 256) -- FC --> (Nq 3)
pred_boxes_init = torch.zeros((len(query), 10), ...)
pred_boxes_init[..., 2:4] = box_sizes[..., 0:2]  # w l
pred_boxes_init[..., 5:6] = box_sizes[..., 2:3]  # h

# xy, wl, z, h, sin, cos, vx, vy, vz
track_instances.pred_boxes = pred_boxes_init
# 向instances中添加query 、 ref_points 字段
track_instances.ref_pts = self.reference_points(query[..., :dim // 2])  # (Nq 256) -- FC --> (Nq 3)
track_instances.query = query
```

初始化其他空字段
```python
track_instances.output_embedding = torch.zeros(Nq 256)
track_instances.obj_idxes = torch.full([Nq], -1, long)
track_instances.matched_gt_idxes = torch.full([Nq], -1, long)
track_instances.disappear_time = torch.zeros([Nq], long)
track_instances.scores = torch.zeros([Nq], float)
track_instances.track_scores = torch.zeros([Nq], float)
track_instances.pred_logits = torch.zeros(torch.zeros([Nq, num_classes], float)

track_instances.mem_bank = torch.zeros([Nq, mem_bank_len=4, 256])
track_instances.mem_padding_mask = torch.ones([Nq, mem_bank_len=4], bool)
track_instances.save_period = torch.zeros([Nq], float)
```

`vip3d.py` :: `forward_test`  (5->7) 
```python
track_instances = self._inference_single(
                  points_single, # B=1, N=34752, C=5
                  img_single,  # B=1 V=6 C=3 H=928 W=1600
                  radar_single,  # B 100 14
                  img_metas_single,
                  track_instances,
                  l2g_r1, l2g_t1, l2g_r2, l2g_t2,  # None, None, [3 3], [1 3]
                  time_delta,  # 0.0
                  gt_bboxes_3d=predictor_utils.tensors_tracking_to_detection(gt_bboxes_3d, i),  # None for test
                  gt_labels_3d=predictor_utils.tensors_tracking_to_detection(gt_labels_3d, i))
```

`ViP3D/plugin/vip3d/utils.py` (8)
```python
def tensors_tracking_to_detection(tensors, cur_frame):
    if tensors is None:
        return None
    return [each[cur_frame] for each in tensors]
```

### `vip3d.py` :: `_inference_single`  (9)
```python
def inference_single(...):
    active_inst = track_instances[track_instances.obj_idxes >= 0]  # 0 when init
    other_inst = track_instances[track_instances.obj_idxes < 0]  # 300 when init

    if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
        ref_pts = active_inst.ref_pts  # 参考位置 [300, 3]
        velo = active_inst.pred_boxes[:, -2:]  # 速度 [300, 2]
        ref_pts = self.velo_update(  # 更新参考位置 # TODO
            ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2,
            time_delta=time_delta)
        active_inst.ref_pts = ref_pts
    track_instances = Instances.cat([other_inst, active_inst])


```

vip3d.py :: velo_update  (10)
通过速度更新参考位置
```python
def velo_update(self, ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2,time_delta):
    reference_points = ref_pts.sigmoid().clone() # inv_sigmoid 空间 -> 3D 空间
    # unormalize
    reference_points = reference_points + velo_pad * time_delta
    ref_pts = reference_points @ l2g_r1.T + l2g_t1  # lidar_t1 -> global 
    ref_pts = (ref_pts - l2g_t2) @ torch.linalg.inv(l2g_r2).T.type(torch.float)  # global -> lidar_curr
    return inverse_sigmoid(ref_pts) # 3D 空间 -> inv_sigmoid 空间
```

### `vip3d.py` :: `_inference_single`  (9)
```python
# 更新ref_pts位置
# 提取传感器特征
img_feats, radar_feats, pts_feats = self.extract_feat(
                points, img=img, radar=radar, img_metas=img_metas)
# multi scale
img_feats = [a.clone() for a in img_feats] # 1 6 256 116*200 58*100 29*50 15*25

```

### `vip3d.py` :: `extract_feat`  (12)
提取radar 和 img 特征
```python
if radar is not None:
    radar_feats = self.radar_encoder(radar)
else:
    radar_feats = None
if self.fix_feats:  # False
    with torch.no_grad():
        img_feats = self.extract_img_feat(img, img_metas)
    else:
        img_feats = self.extract_img_feat(img, img_metas)
return (img_feats, radar_feats, None)  # 没用激光雷达信息？
```

### `vip3d.py` :: `_inference_single`  (9)
```python
# 更新ref_pts位置
# 提取传感器特征
# output_classes: [num_dec, B, num_query, num_classes]
# query_feats: [B, num_query, embed_dim]
ref_box_sizes = torch.cat( 
                [track_instances.pred_boxes[:, 2:4],  #  w l 
                 track_instances.pred_boxes[:, 5:6]], dim=1)  # h

output_classes, output_coords, query_feats, last_ref_pts = self.pts_bbox_head(
                img_feats, radar_feats, track_instances.query,
                track_instances.ref_pts, ref_box_sizes, img_metas)  # output 6 1 300 7=class 10=coords
                # query_feats  1 300 256 last_ref_pts 1 300 3
```

### `vip3d.py` :: `_inference_single`  (9)
```python
if self.add_branch:
    self.update_history_img_list(img_metas, img, img_feats)  # img_feats queue of size 4

out = {'pred_logits': output_classes[-1],  # not used, 1 300 7
        'pred_boxes': output_coords[-1],  # 6 是 transformer 层数还是 视角？
        'ref_pts': last_ref_pts}
```

### 'vip3d/models/head_plus_raw.py' :: `DeformableDETR3DCamHeadTrackPlusRaw`::forward()  (13)
- 给特征图添加 size padding 并 叠加位置编码
- 交叉注意力
```python
    def forward(self, mlvl_feats, radar_feats,
                query_embeds, ref_points, ref_size, img_metas, petr_feature=False):
        """
        Args:
            输入图像特征、毫米波、agent queries、参考点和参考大小
            multi-level multi-view (tuple[Tensor]): List of Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W). # 4 B=1 V=6 256 116x200 / 58x100 / 29 50 / 15 25
            radar_feats (Tensor) : radar features of shape (B, N, C)
            query_embeds (Tensor):  pos_embed and feature for querys of shape
                (num_query, embed_dim*2)  # B=1 100 35
            ref_points (Tensor):  3d reference points associated with each query
                shape (num_query, 3) in inverse sigmoid space
            ref_size (Tensor): the size(bbox size) associated with each query
                shape (num_query, 3) in log space. 
        Returns:
            输出 num_dec_layers 个分类和回归分支的预测、最后一层的query特征和参考点
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, sine, cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 10].
            last_query_feats (Tensor): shape [bs, num_query, feat_dim]
            last_ref_points (Tensor): shape [bs, num_query, 3]
        """
        # 给特征图添加 size padding 并 叠加位置编码

        # 交叉注意力
        hs, inter_references, inter_box_sizes = self.transformer(  # Detr3DCamTrackTransformer
            mlvl_feats,
            query_embeds,
            ref_points,
            ref_size,
            reg_branches=self.reg_branches,
            img_metas=img_metas,
            radar_feats=radar_feats,
        )
```

`vip3d/models/transformer.py`::`Detr3DCamTrackTransformer`::forward()  (14)
- 维度调整
- 增强ref_points
- 调用Detr3DCamTrackTransformerDecoder
```python
    def forward(self,
                mlvl_feats,
                query_embed,
                reference_points,
                ref_size,
                reg_branches=None,
                **kwargs):
        """Forward function for `Transformer`.
        Args:
            mlvl_feats (list(Tensor)): lvl=4 * [bs, num_cams, embed_dims, h, w].
            query_embed (Tensor): [num_query, 2*embed_dim] = feat : pos
            reference_points (Tensor): 3d ref points with shape (num_query, 3) in inverse sigmoid space
            ref_size (Tensor): the wlh(bbox size) associated  (num_query, 3) in log space. 
            reg_branches (obj:`nn.ModuleList`): Regression heads for feature maps from each decoder layer. 
                
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder, has shape \
                      (num_dec_layers, num_query, bs, embed_dims)
                - init_reference_out: The initial value of reference \
                    points, (bs, num_queries, 3).
                - inter_references_out: The internal value of reference \
                    points in decoder, (num_dec_layers, bs, num_query, 3)
                
        """
        # ref_pts augmentation
        # ... 
        inter_states, inter_references, inter_box_sizes = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            ref_size=ref_size,
            **kwargs)
        return inter_states, inter_references, inter_box_sizes
```


`vip3d/models/transformer.py`::`Detr3DCamTrackPlusTransformerDecoder`::forward()
- 逐层调用transformer decoder做交叉注意力解码
```python
    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                ref_size=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): `(num_query, bs, embed_dims)`.
            # value=mlvl_feats,
            reference_points (Tensor):  (num_query, 3) inv_sigm
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
            ref_size (Tensor): (bs, num_query, 3) log
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query  # torch.Size([300, 1, 256])
        intermediate = []
        intermediate_reference_points = []
        intermediate_box_sizes = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                ref_size=ref_size,
                **kwargs)
            output = output.permute(1, 0, 2)
```

`vip3d/models/attention_detr3d.py` Detr3DCrossAtten::forward()  (15)

```python
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape (num_query, bs, embed_dims). 
            key (Tensor): The key tensor with shape (num_key, bs, embed_dims).  None
            value (Tensor): The value tensor with shape (num_key, bs, embed_dims)`. List[(B, N, C, H, W)]
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`. Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default None.
            reference_points (Tensor):  The normalized reference points with shape (bs, num_query, 3) #!4,
                all elements is range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add additional two dimensions is (w, h) to form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with shape [bs, num_key].  None
            spatial_shapes (Tensor): Spatial shape of features in different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level. A tensor has shape (num_levels) and 
                can be represented as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)
        bs, num_query, _ = query.size()
        # query -- FC(256, 24) --> (bs, num_query, cams*levels*points) 6 4 1
        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        # 1 1 300 6 1 4
        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        
        
```

`vip3d/models/attention_detr3d.py` feature_sampling()  (16)
- 通过参考点采样特征: 把3d参考点通过img_metas::lidar2img变换投影到6个图像上，在对应位置采集特征
```python
def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points (B, num_queries, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0][0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 1:2] > -1.0)
            & (reference_points_cam[..., 1:2] < 1.0))
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B * N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B * N, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam, 1, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask
```

`vip3d/models/attention_detr3d.py` Detr3DCrossAtten::forward()  (15)
```python
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat
```

`vip3d/models/transformer.py`::`Detr3DCamTrackPlusTransformerDecoder`::forward()  (14)
- 逐层调用transformer decoder做交叉注意力解码
- 过reg_branches
```python
if reg_branches is not None:
    tmp = reg_branches[lid](output)
    ref_pts_update = torch.cat([tmp[..., :2],tmp[..., 4:5],], dim=-1)
    ref_size_update = torch.cat([tmp[..., 2:4], tmp[..., 5:6]], dim=-1)
    assert reference_points.shape[-1] == 3
    new_reference_points = ref_pts_update + inverse_sigmoid(reference_points)
    new_reference_points = new_reference_points.sigmoid()
    reference_points = new_reference_points.detach()

    # ref_size = (ref_size.exp() + ref_size_update.exp()).log()
    ref_size = ref_size + ref_size_update
    if lid > 0:
        ref_size = ref_size.detach()
    output = output.permute(1, 0, 2)
    if self.return_intermediate:
        intermediate.append(output)
        intermediate_reference_points.append(reference_points)
        intermediate_box_sizes.append(ref_size)

    if self.return_intermediate:
        return torch.stack(intermediate), torch.stack(
                intermediate_reference_points), \
                torch.stack(intermediate_box_sizes)

    return output, reference_points, ref_size
```