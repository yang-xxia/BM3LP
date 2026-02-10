import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 获取项目的根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from layers.layer import *


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.device = args.device

    @staticmethod
    def format_metrics(metrics, split):
        return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])

    @staticmethod
    def has_improved(m1, m2):
        return (m1["Mean Rank"] > m2["Mean Rank"]) or (m1["Mean Reciprocal Rank"] < m2["Mean Reciprocal Rank"])

    @staticmethod
    def init_metric_dict():
        return {"Hits@100": -1, "Hits@10": -1, "Hits@3": -1, "Hits@1": -1,
                "Mean Rank": 100000, "Mean Reciprocal Rank": -1}


class SpGAT_Relational(nn.Module):
    """
    Relational GAT layer for KG embeddings (multi-head)
    """
    def __init__(self, num_nodes, in_dim, out_dim, nheads=2, dropout=0.3, alpha=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nheads = nheads
        self.dropout = dropout
        self.alpha = alpha

        # 每个头的线性映射
        self.W = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(nheads)])
        # 注意力参数
        self.att = nn.ParameterList([nn.Parameter(torch.empty(2*out_dim,1)) for _ in range(nheads)])
        for a in self.att:
            nn.init.xavier_uniform_(a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, entity_emb, relation_emb, edge_index, edge_type=None):
        """
        这里先做占位，返回多头扩展后的 entity 和 relation embedding
        """
        out_entity = entity_emb.repeat(1, self.nheads)  # [num_nodes, in_dim * nheads]
        out_relation = relation_emb.repeat(1, self.nheads)  # [num_relations, in_dim * nheads]
        alpha = None
        return out_entity, out_relation, alpha


class DFS_RGAT(nn.Module):
    """
    DFS + RGAT Encoder for KG embeddings (支持反向关系)
    """
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.num_nodes = len(args.entity2id)

        # ===== 核心修复2：关系数量翻倍 =====
        self.num_relations = len(args.relation2id) * 2   # 支持反向关系
        self.in_dim = args.dim
        self.out_dim = args.dim
        self.nheads = args.n_heads
        self.neg_num = getattr(args, "neg_num_gat", 2)
        self.dropout = getattr(args, "dropout_gat", 0.3)
        self.alpha = getattr(args, "alpha_gat", 0.2)

        # Embeddings
        self.entity_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.in_dim))
        self.relation_embeddings = nn.Parameter(torch.randn(self.num_relations, self.in_dim))

        # final embedding (multi-head)
        self.final_entity_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.out_dim*self.nheads))
        self.final_relation_embeddings = nn.Parameter(torch.randn(self.num_relations, self.out_dim*self.nheads))

        # RGAT layer
        self.rgat = SpGAT_Relational(self.num_nodes, self.in_dim, self.out_dim, self.nheads, self.dropout, self.alpha)

        # DFS batch mask & attention cache
        self.capture_attention = True
        self.last_edge_index = None
        self.last_alpha = None
        self.last_out_entity = None
        self.last_out_relation = None

        # W 投影矩阵
        self.W_entities = nn.Parameter(torch.zeros(self.in_dim, self.out_dim*self.nheads))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

    def forward(self, adj, train_indices):
        edge_index, edge_type = adj[0].to(self.device), adj[1].to(self.device)

        # normalize embeddings
        self.entity_embeddings.data = F.normalize(self.entity_embeddings.data, p=2, dim=1).detach()
        self.relation_embeddings.data = F.normalize(self.relation_embeddings.data, p=2, dim=1).detach()

        # DFS batch mask
        batch_nodes = torch.unique(train_indices.flatten()).to(self.device)
        mask = torch.zeros(self.num_nodes, device=self.device)
        mask[batch_nodes] = 1.0

        # RGAT forward
        out_entity, out_relation, alpha = self.rgat(self.entity_embeddings, self.relation_embeddings, edge_index, edge_type)

        # 线性投影 + batch mask
        mask_exp = mask.unsqueeze(-1).expand(-1, out_entity.size(1))
        out_entity = F.normalize(self.entity_embeddings.mm(self.W_entities) + mask_exp * out_entity, p=2, dim=1)

        # attention cache
        if self.capture_attention:
            self.last_edge_index = edge_index.detach().clone()
            self.last_alpha = alpha.detach().clone() if alpha is not None else None
            self.last_out_entity = out_entity.detach().clone()
            self.last_out_relation = out_relation.detach().clone()

        # update final embedding
        self.final_entity_embeddings.data = out_entity.data
        self.final_relation_embeddings.data = out_relation.data

        return out_entity, out_relation

    def loss_func(self, train_indices, entity_embeddings, relation_embeddings):
        len_pos_triples = int(train_indices.shape[0] / (int(self.neg_num)+1))
        pos_triples = train_indices[:len_pos_triples]
        neg_triples = train_indices[len_pos_triples:]

        # 对齐正负样本数量
        min_len = min(len(pos_triples)*self.neg_num, len(neg_triples))
        pos_triples = pos_triples.repeat(self.neg_num, 1)[:min_len]
        neg_triples = neg_triples[:min_len]

        # 正样本
        src = entity_embeddings[pos_triples[:,0]]
        rel = relation_embeddings[pos_triples[:,1]]
        tgt = entity_embeddings[pos_triples[:,2]]
        pos_norm = torch.norm(src + rel - tgt, p=1, dim=1)

        # 负样本
        src = entity_embeddings[neg_triples[:,0]]
        rel = relation_embeddings[neg_triples[:,1]]
        tgt = entity_embeddings[neg_triples[:,2]]
        neg_norm = torch.norm(src + rel - tgt, p=1, dim=1)

        y = -torch.ones(min_len, device=self.device)
        return F.margin_ranking_loss(pos_norm, neg_norm, y, margin=1.0)

    @torch.no_grad()
    def export_attention(self):
        return self.last_edge_index, self.last_alpha

class Mutan(BaseModel):
    def __init__(self, args):
        super(Mutan, self).__init__(args)
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        self.dim = args.dim
        self.Mutan = MutanLayer(args.dim, 5)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()

    def forward(self, batch_inputs):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        pred = self.Mutan(e_embed, r_embed)
        pred = torch.mm(pred, self.entity_embeddings.weight.transpose(1, 0))
        pred = torch.sigmoid(pred)
        return pred

    def loss_func(self, output, target):
        return self.bceloss(output, target)


class TuckER(BaseModel):
    def __init__(self, args):
        super(TuckER, self).__init__(args)
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        if args.pre_trained:
            self.entity_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_entity_vec.pkl', 'rb'))).float(), freeze=False)
            self.relation_embeddings = nn.Embedding.from_pretrained(torch.cat((
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float(),
                -1 * torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float()), dim=0), freeze=False)
        self.dim = args.dim
        self.TuckER = TuckERLayer(args.dim, args.r_dim)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()

    def forward(self, batch_inputs, lookup=None):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        pred = self.TuckER(e_embed, r_embed)
        if lookup is None:
            pred = torch.mm(pred, self.entity_embeddings.weight.transpose(1, 0))
        else:
            pred = torch.bmm(pred.unsqueeze(1), self.entity_embeddings.weight[lookup].transpose(1, 2)).squeeze(1)
        pred = torch.sigmoid(pred)
        return pred

    def loss_func(self, output, target):
        return self.bceloss(output, target)


class ConvE(BaseModel):
    def __init__(self, args):
        super(ConvE, self).__init__(args)
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        self.dim = args.dim
        self.k_w = args.k_w
        self.k_h = args.k_h
        self.ConvE = ConvELayer(args.dim, args.out_channels, args.kernel_size, args.k_h, args.k_w)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()

    def forward(self, batch_inputs):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        e_embed = e_embed.view(-1, 1, self.dim)
        r_embed = r_embed.view(-1, 1, self.dim)
        embed = torch.cat([e_embed, r_embed], dim=1)
        embed = torch.transpose(embed, 2, 1).reshape((-1, 1, 2 * self.k_w, self.k_h))

        pred = self.ConvE(embed)
        pred = torch.mm(pred, self.entity_embeddings.weight.transpose(1, 0))
        pred += self.bias.expand_as(pred)
        pred = torch.sigmoid(pred)
        return pred

    def loss_func(self, output, target):
        return self.bceloss(output, target)


class BM3LP(BaseModel):
    """
    改进点：
    - return_intermediates: 在 forward 中可选返回中间嵌入，便于外部可视化/记录
    - use_image / use_text: 在前向中屏蔽某个模态（置零），用于模态贡献度估计（留一/只保留）
    - image_heatmap: 基于 head 的图像嵌入做 Grad-CAM 风格热图（16x16）
    - modality_contribution: 返回 structure/image/text 对 pred_mm(h,r,t) 的相对贡献
    - text_similarity: 文本向量与关系/尾实体文本向量的余弦相似度
    """
    def __init__(self, args):
        super(BM3LP, self).__init__(args)

        # === 基础嵌入 ===
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        # === 可解释/可视化开关 ===
        self.return_intermediates = False   # True 时 forward 额外返回中间嵌入
        self.use_image = True               # 控制是否使用图像模态
        self.use_text  = True               # 控制是否使用文本模态

        # === 预训练（可选） ===
        if getattr(args, "pre_trained", 0):
            self.entity_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open(f'datasets/{args.dataset}/gat_entity_vec.pkl', 'rb'))).float(),
                freeze=False
            )
            gat_rel = torch.from_numpy(pickle.load(open(f'datasets/{args.dataset}/gat_relation_vec.pkl', 'rb'))).float()
            self.relation_embeddings = nn.Embedding.from_pretrained(
                torch.cat((gat_rel, -1 * gat_rel), dim=0),
                freeze=False
            )

        # === 图像与文本实体嵌入的构造 ===
        # 图像：输入假设为 [N, 64*64] 可 reshape 为 (N,64,64) -> AvgPool2d(4) -> (N,16,16) -> flatten 256
        img_pool = torch.nn.AvgPool2d(4, stride=4)
        img = img_pool(args.img.to(self.device).view(-1, 64, 64))
        img = img.view(img.size(0), -1)  # 16*16 = 256
        self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=False)
        self.img_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.img_relation_embeddings.weight)

        # 文本：输入假设为 [N, 8*64] 可 reshape 为 (N,8,64) -> AdaptiveAvgPool2d(4,64) -> flatten 256
        txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
        txt = txt_pool(args.desp.to(self.device).view(-1, 8, 64))
        txt = txt.view(txt.size(0), -1)   # 4*64 = 256
        self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=False)
        self.txt_relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.txt_relation_embeddings.weight)

        # === 模型主体 ===
        self.dim = args.dim
        self.TuckER_S  = TuckERLayer(args.dim, args.r_dim)
        self.TuckER_I  = TuckERLayer(args.dim, args.r_dim)
        self.TuckER_D  = TuckERLayer(args.dim, args.r_dim)
        self.TuckER_MM = TuckERLayer(args.dim, args.r_dim)
        self.Mutan_MM_E = MutanLayer(args.dim, 2)
        self.Mutan_MM_R = MutanLayer(args.dim, 2)

        self.bias      = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss   = nn.BCELoss()
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)

        # 输出层（对所有尾实体得分）
        self.output_layer_s  = nn.Linear(args.dim, len(args.entity2id))
        self.output_layer_i  = nn.Linear(args.dim, len(args.entity2id))
        self.output_layer_d  = nn.Linear(args.dim, len(args.entity2id))
        self.output_layer_mm = nn.Linear(args.dim, len(args.entity2id))

        # === 记录图像网格尺寸用于热图（与上面的池化对应） ===
        self.img_grid_h = 16
        self.img_grid_w = 16

    # ==========================
    #         损失们
    # ==========================
    def cross_modal_contrastive_loss(self, head, tail, e_embed, img_embed, txt_embed):
        e_embed  = F.normalize(e_embed,  dim=-1)
        img_embed= F.normalize(img_embed,dim=-1)
        txt_embed= F.normalize(txt_embed,dim=-1)

        pos_e_img = torch.sum(e_embed * img_embed, dim=1, keepdim=True)
        pos_e_txt = torch.sum(e_embed * txt_embed, dim=1, keepdim=True)

        neg_e_img = torch.matmul(e_embed, img_embed.t())
        neg_e_txt = torch.matmul(e_embed, txt_embed.t())
        neg_e_img = neg_e_img - torch.diag(torch.diag(neg_e_img))
        neg_e_txt = neg_e_txt - torch.diag(torch.diag(neg_e_txt))

        pos = torch.mean(torch.cat([pos_e_img, pos_e_txt], dim=1), dim=1)
        neg = torch.mean(torch.cat([neg_e_img, neg_e_txt], dim=1), dim=1)

        loss = torch.mean(F.softplus(neg / self.temperature - pos / self.temperature))
        return loss

    def entity_interaction_contrastive_loss(self, head_entities, e_embed, img_embed, txt_embed):
        e_embed  = F.normalize(e_embed,  dim=-1)
        img_embed= F.normalize(img_embed,dim=-1)
        txt_embed= F.normalize(txt_embed,dim=-1)

        pos_e_img = torch.sum(e_embed * img_embed, dim=1, keepdim=True)
        pos_e_txt = torch.sum(e_embed * txt_embed, dim=1, keepdim=True)

        neg_e_img = torch.matmul(e_embed, img_embed.t())
        neg_e_txt = torch.matmul(e_embed, txt_embed.t())
        neg_e_img = neg_e_img - torch.diag(torch.diag(neg_e_img))
        neg_e_txt = neg_e_txt - torch.diag(torch.diag(neg_e_txt))

        pos = torch.mean(torch.cat([pos_e_img, pos_e_txt], dim=1), dim=1)
        neg = torch.mean(torch.cat([neg_e_img, neg_e_txt], dim=1), dim=1)

        loss = torch.mean(F.softplus(neg / self.temperature - pos / self.temperature))
        return loss

    def contrastive_loss(self, s_embed, v_embed, t_embed):
        s_embed = F.normalize(s_embed, dim=-1)
        v_embed = F.normalize(v_embed, dim=-1)
        t_embed = F.normalize(t_embed, dim=-1)

        pos_sv = torch.sum(s_embed * v_embed, dim=1, keepdim=True)
        pos_st = torch.sum(s_embed * t_embed, dim=1, keepdim=True)
        pos_vt = torch.sum(v_embed * t_embed, dim=1, keepdim=True)

        neg_s = torch.matmul(s_embed, s_embed.t()); neg_s = neg_s - torch.diag(torch.diag(neg_s))
        neg_v = torch.matmul(v_embed, v_embed.t()); neg_v = neg_v - torch.diag(torch.diag(neg_v))
        neg_t = torch.matmul(t_embed, t_embed.t()); neg_t = neg_t - torch.diag(torch.diag(neg_t))

        pos = torch.mean(torch.cat([pos_sv, pos_st, pos_vt], dim=1), dim=1)
        neg = torch.mean(torch.cat([neg_s, neg_v, neg_t], dim=1), dim=1)

        loss = torch.mean(F.softplus(neg / self.temperature - pos / self.temperature))
        return loss

    # ==========================
    #         前向
    # ==========================
    def forward(self, batch_inputs):
        """
        输入: batch_inputs [B, 2]，列为 (head_id, relation_id)
        输出: [pred_s, pred_i, pred_d, pred_mm]，每个都是 [B, |E|] 的 sigmoid 分数
        当 self.return_intermediates 为 True 时，返回 (pred_list, intermediates_dict)
        """
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]

        # 基础嵌入
        e_embed   = self.entity_embeddings(head)            # [B, dim]
        r_embed   = self.relation_embeddings(relation)      # [B, r_dim]
        e_img_embed = self.img_entity_embeddings(head)      # [B, dim]
        r_img_embed = self.img_relation_embeddings(relation)# [B, r_dim]
        e_txt_embed = self.txt_entity_embeddings(head)      # [B, dim]
        r_txt_embed = self.txt_relation_embeddings(relation)# [B, r_dim]

        # 屏蔽模态（零化张量，不影响其他分支梯度）
        if not self.use_image:
            e_img_embed = torch.zeros_like(e_img_embed)
            r_img_embed = torch.zeros_like(r_img_embed)
        if not self.use_text:
            e_txt_embed = torch.zeros_like(e_txt_embed)
            r_txt_embed = torch.zeros_like(r_txt_embed)

        # 融合（Mutan）
        e_mm_embed = self.Mutan_MM_E(e_embed, e_img_embed, e_txt_embed)
        r_mm_embed = self.Mutan_MM_R(r_embed, r_img_embed, r_txt_embed)

        # 各分支 TuckER -> 全尾实体打分 -> Sigmoid
        pred_s  = torch.sigmoid(self.output_layer_s (self.TuckER_S (e_embed,   r_embed)))
        pred_i  = torch.sigmoid(self.output_layer_i (self.TuckER_I (e_img_embed, r_img_embed)))
        pred_d  = torch.sigmoid(self.output_layer_d (self.TuckER_D (e_txt_embed, r_txt_embed)))
        pred_mm = torch.sigmoid(self.output_layer_mm(self.TuckER_MM(e_mm_embed, r_mm_embed)))

        preds = [pred_s, pred_i, pred_d, pred_mm]

        if self.return_intermediates:
            inter = {
                "e_embed": e_embed,       "r_embed": r_embed,
                "e_img_embed": e_img_embed, "r_img_embed": r_img_embed,
                "e_txt_embed": e_txt_embed, "r_txt_embed": r_txt_embed,
                "e_mm_embed": e_mm_embed,   "r_mm_embed": r_mm_embed
            }
            return preds, inter
        return preds

    def loss_func(self, output, target, head):
        """
        output: [pred_s, pred_i, pred_d, pred_mm]，每个 [B, |E|]
        target: [B, |E|] 多标签
        head:   [B] 头实体 id
        """
        loss_s  = self.bceloss(output[0], target)
        loss_i  = self.bceloss(output[1], target)
        loss_d  = self.bceloss(output[2], target)
        loss_mm = self.bceloss(output[3], target)

        # 对比学习损失
        e_embed    = self.entity_embeddings(head)
        e_img_embed= self.img_entity_embeddings(head)
        e_txt_embed= self.txt_entity_embeddings(head)

        contrastive_loss_value = self.contrastive_loss(e_embed, e_img_embed, e_txt_embed)
        entity_interaction_loss = self.entity_interaction_contrastive_loss(head, e_embed, e_img_embed, e_txt_embed)
        cross_modal_loss        = self.cross_modal_contrastive_loss(head, target, e_embed, e_img_embed, e_txt_embed)

        total_loss = loss_s + loss_i + loss_d + loss_mm \
                   + contrastive_loss_value + entity_interaction_loss + cross_modal_loss
        return total_loss

    # ==========================
    #     便捷可视化/解释接口
    # ==========================
    @torch.no_grad()
    def score_triple_mm(self, h_id:int, r_id:int, t_id:int, device=None) -> float:
        """
        返回融合分支 pred_mm 对 (h,r,t) 的分数（标量）
        """
        device = device or getattr(self, "device", "cpu")
        self.eval()
        x = torch.tensor([[h_id, r_id]], dtype=torch.long, device=device)
        preds = self.forward(x)[3]  # pred_mm: [1, |E|]
        return float(preds[0, t_id].item())

    def modality_contribution(self, h_id:int, r_id:int, t_id:int, device=None):
        """
        通过“只保留单一模态”的方式估计三模态相对贡献，返回 softmax 归一化权重。
        """
        device = device or getattr(self, "device", "cpu")
        self.eval()

        # baseline （不用于权重，仅返回参考）
        self.use_image, self.use_text = True, True
        base = self.score_triple_mm(h_id, r_id, t_id, device)

        # 只结构
        self.use_image, self.use_text = False, False
        s_struct = self.score_triple_mm(h_id, r_id, t_id, device)

        # 只图像
        self.use_image, self.use_text = True, False
        s_img = self.score_triple_mm(h_id, r_id, t_id, device)

        # 只文本
        self.use_image, self.use_text = False, True
        s_txt = self.score_triple_mm(h_id, r_id, t_id, device)

        # 还原
        self.use_image, self.use_text = True, True

        raw = torch.tensor([s_struct, s_img, s_txt], dtype=torch.float32)
        w = F.softmax(raw, dim=0).tolist()
        return {
            "structure": w[0], "image": w[1], "text": w[2],
            "base_score": base,
            "scores": {"struct_only": float(s_struct), "img_only": float(s_img), "txt_only": float(s_txt)}
        }

    def image_heatmap(self, h_id:int, r_id:int, t_id:int, device=None):
        """
        Grad-CAM 风格热图：对 pred_mm(h,r,·) 中 t_id 的标量，对 head 的图像嵌入求梯度，
        将 256 维还原为 (16,16)，返回 numpy 数组；若维度不为 256，返回一维重要性向量。
        """
        device = device or getattr(self, "device", "cpu")
        self.eval()

        # 暂时冻结所有参数
        for p in self.parameters():
            p.requires_grad_(False)

        # 仅对整张图像实体嵌入表启用梯度
        table = self.img_entity_embeddings.weight  # Parameter [|E|, 256]
        table.requires_grad_(True)

        # 前向到目标分数
        x = torch.tensor([[h_id, r_id]], dtype=torch.long, device=device)
        preds = self.forward(x)[3]  # pred_mm
        score = preds[0, t_id]

        grads_full = torch.autograd.grad(score, table, retain_graph=False, allow_unused=True)[0]
        if grads_full is None:
            return None
        g = grads_full[h_id]  # [256] (理想情况下)

        kh, kw = self.img_grid_h, self.img_grid_w
        try:
            heat = g.reshape(kh, kw)
            heat = torch.relu(heat)
            heat = heat / (heat.max() + 1e-6)
            return heat.detach().cpu().numpy()
        except Exception:
            # 若不是 256 维，直接返回一维重要性向量
            return g.detach().cpu().numpy()

    @torch.no_grad()
    def text_similarity(self, h_id:int, r_id:int=None, t_id:int=None):
        """
        计算 head 文本向量与关系/尾实体文本向量的余弦相似度
        """
        self.eval()
        vec_h = self.txt_entity_embeddings.weight[h_id]
        sims = {}
        if r_id is not None:
            sims['cos_to_relation'] = F.cosine_similarity(
                vec_h, self.txt_relation_embeddings.weight[r_id], dim=0).item()
        if t_id is not None:
            sims['cos_to_tail_text'] = F.cosine_similarity(
                vec_h, self.txt_entity_embeddings.weight[t_id], dim=0).item()
        return sims
    
    