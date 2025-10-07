import dgl
import torch.nn as nn
from transformers.models.esm.openfold_utils import feats
import torch.nn.functional as F
from mpnn import MPNNGNN
from Feature_Fusion import Fusion
import torch


class MultiAttention(nn.Module):
    def __init__(self, embed_dim=128, nhead=4, ff_dim=256):
        super().__init__()
        self.attn_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim,
                                                     batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # self.pool = nn.AdaptiveAvgPool1d(1)  # 也可以换成自定义AttentionPool
        self.pool = AttentionPool(embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):  # x shape: [B, L, C]
        # Multi-head self-attention + Residual
        attn_out = self.attn_layer(x)  # [B, L, C]
        x = self.norm1(x + attn_out)

        # Feed-forward + Residual
        ffn_out = self.ffn(x)  # [B, L, C]
        x = self.norm2(x + ffn_out)

        # 池化，变成 [B, C]
        # x = x.transpose(1, 2)  # [B, C, L]
        x = self.pool(x)  # [B, C]

        # 预测输出
        out = self.regressor(x).squeeze(-1)  # [B]
        return out

class AttentionPool(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))  # [1, 1, C]
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):  # x: [B, L, C]
        # 复制 query 到每个 batch：[B, 1, C]
        query = self.query.expand(x.size(0), -1, -1)
        # 计算注意力分数：[B, 1, L]
        attn_scores = torch.bmm(query, x.transpose(1, 2)) / (x.size(-1) ** 0.5)
        attn_weights = self.softmax(attn_scores)  # [B, 1, L]
        # 加权求和：[B, 1, C] → squeeze → [B, C]
        context = torch.bmm(attn_weights, x).squeeze(1)
        return context

class AttentionRegressor(nn.Module):
    def __init__(self, embed_dim=256, nhead=4, ff_dim=512):
        super().__init__()
        self.attn = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim, batch_first=True)
        # self.pool = nn.AdaptiveAvgPool1d(1)  # 或自定义 attention pool
        self.pool = AttentionPool(embed_dim)  # 使用自定义 Attention Pool
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出 kcat
        )

    def forward(self, x):  # x: [B, L, C]
        x = self.attn(x)  # 保持 [B, L, C]
        # x = x.mean(dim=1)  # 简单平均池化（也可换成 attention pool）
        x = self.pool(x)
        out= self.regressor(x).squeeze(-1)
        return out  # 输出 [B]

class CrossAttention(nn.Module):
    def __init__(self, node_dim, protein_dim, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(node_dim, hidden_dim)
        self.key_proj = nn.Linear(protein_dim, hidden_dim)
        self.value_proj = nn.Linear(protein_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, node_dim)  # 输出维度回到节点特征

    def forward(self, X_c, X_p):
        # X_c: [B, N, node_dim], X_p: [B, L, protein_dim]
        Q = self.query_proj(X_c)  # [B, N, H]
        K = self.key_proj(X_p)    # [B, L, H]
        V = self.value_proj(X_p)  # [B, L, H]

        # 注意力计算
        attn_scores = torch.bmm(Q, K.transpose(1,2)) / (Q.size(-1)**0.5)  # [B, N, L]
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights, V)  # [B, N, H]

        # 输出更新后的节点特征
        out = self.out_proj(context) + X_c  # 残差连接
        return out

class MPNNPredictorWithProtein(nn.Module):
    def __init__(self,
                 node_in_feats=74,  # 节点特征维度
                 edge_in_feats=12,  # 边特征维度
                 protein_feats=1320,  #蛋白特征维度
                 node_out_feats=64,     #default=64
                 protein_out_feats=64,     #蛋白和化合物输出维度需一样
                 edge_hidden_feats=128,
                 n_tasks=1,     #default=1
                 num_step_message_passing=8,        #default=6
                 num_step_set2set=6,        #default=6
                 num_layer_set2set=3,        #default=3
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        super().__init__()
        self.fusion = Fusion(hidden=64 ,
                             node_feats_in=node_out_feats,
                             pro_feats_in=protein_out_feats).to(device)

        # MPNN部分：处理化合物的图结构
        '''输出维度是输出维度是node_out_feats'''
        self.gnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing,
                           )

        # 图级别的Set2Set聚合
        # '''输出维度是(2 * node_out_feats)'''
        # self.readout = Set2Set(input_dim=node_out_feats,
        #                        n_iters=num_step_set2set,
        #                        n_layers=num_layer_set2set)

        #蛋白质特征处理
        self.protein_projector = nn.Sequential(
            nn.Linear(protein_feats, 1024),  # 降维到256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),  # 降维到256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, protein_out_feats)  # 进一步对齐分子图特征维度
        )

        # 预测层
        self.predict = nn.Sequential(
            nn.Linear(64, 32),      #分子图特征128 + 蛋白质特征128
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, n_tasks),
        )









    def unbatch_node_feats(self,node_feats, batched_graph, max_nodes):

        batch_size = batched_graph.batch_size
        output_feats = []
        num_nodes = batched_graph.batch_num_nodes(ntype='_N')       #tensor（L1，L2,L3....）
        num_node_list = num_nodes.tolist()  # 将 tensor 转为列表后遍历

        # 获取每个图的节点数
        start_idx = 0
        for i in range(batch_size):
            num_node=num_node_list[i]
            # 获取该图的节点特征
            graph_node_feats = node_feats[start_idx:start_idx +num_node ]

            # 填充：如果该图的节点数少于Lmax，则用0填充
            padded_feats = torch.zeros((max_nodes, graph_node_feats.shape[1]), dtype=torch.float32)
            padded_feats[:num_node] = graph_node_feats  # 将图的节点特征填充到矩阵的前部分

            output_feats.append(padded_feats)

            # 更新当前节点的索引
            start_idx += num_node

        # 将所有图的节点特征堆叠到一起，得到 (batch_size, Lmax, C)
        return torch.stack(output_feats)



    def forward(self, graph_feats, node_feats, edge_feats, protein_feats, Ad):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        """联合化合物和蛋白质特征进行预测"""
        # 处理化合物图的节点和边特征
        node_feats = self.gnn(graph_feats, node_feats, edge_feats)        #(sigL,64)

        num_nodes = graph_feats.batch_num_nodes(ntype='_N')
        max_nodes = num_nodes.max().item()
        node_feats = self.unbatch_node_feats(node_feats, graph_feats, max_nodes)        #（batchsize，Lmax，C）（64）
        node_feats = node_feats.to(device)


        # 聚合图节点特征
        # graph_feats = self.readout(graph_feats, node_feats)       # 输出形状: (batch_size, 2*64=128)
        # print('00000000000000000000000',graph_feats.shape)
        # graph_feats_unbatch = dgl.unbatch(graph_feats)      #解合并图
        # for i, graph in enumerate(graph_feats_unbatch):

        # 蛋白质分支
        protein_feats = self.protein_projector(protein_feats)  # 输出形状: (batch_size, 64)


        # 蛋白-化合物特征融合
        cross_attn = CrossAttention(node_dim=64, protein_dim=128, hidden_dim=128)
        enhanced_node_feats = cross_attn(node_feats, protein_feats)


        combined_feats = self.fusion(enhanced_node_feats,Ad, protein_feats)
        # print('conbine:',combined_feats.shape)


        # 预测活性
        predict = self.predict(combined_feats)
        predict = torch.mean(predict, dim=1)
        predict = predict.unsqueeze(-1)  # shape [32,1]
        predict = predict.float()
        # print('predict shape:', predict.shape)
        return predict

