import torch
import torch.nn as nn

class Fusion(nn.Module):
    def __init__(self, layer_gnn=6,  hidden=256 , node_feats_in=256, pro_feats_in=256):
        super(Fusion, self).__init__()
        self.W_gnn = nn.ModuleList([nn.Linear(node_feats_in, hidden)
                                    for _ in range(layer_gnn)])
        self.W_pnn = nn.ModuleList([nn.Linear(pro_feats_in, hidden)
                                    for _ in range(layer_gnn)])

        self.gnn_act = nn.GELU()
        self.G_A = nn.ModuleList(
            [nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1, groups=hidden, bias=False)
             for _ in range(layer_gnn)])
        # self.bn_1d = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layer_gnn)])
        self.ln_1d = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layer_gnn)])

        self.soft_1 = nn.Softmax(-1)
        self.soft_2 = nn.Softmax(-1)

        self.dropout = nn.ModuleList([nn.Dropout(p=0.3) for _ in range(layer_gnn)])


    def Style_Exract(self, feats_d, Ad, feats_p, layer):          #(batch_size, L, hidden2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(layer):
            feats_d_gnn = self.gnn_act(self.W_gnn[i](feats_d))
            feats_p_gnn = self.gnn_act(self.W_pnn[i](feats_p))
            # feats_p_gnnDWC = self.bn_1d[i](self.G_A[i](feats_p_gnn.permute(0, 2, 1))).permute(0, 2, 1)
            feats_p_gnnDWC = self.G_A[i](feats_p_gnn.permute(0, 2, 1)).permute(0, 2, 1)
            feats_p_gnnDWC = self.ln_1d[i](feats_p_gnnDWC)


            C = torch.matmul(feats_p, feats_d.permute(0, 2, 1))

            # CwFd = self.bn_1d[i](torch.matmul(self.soft_1(C), feats_d_gnn).permute(0, 2, 1)).permute(0, 2, 1)
            CwFd = torch.matmul(self.soft_1(C), feats_d_gnn)
            CwFd = self.ln_1d[i](CwFd)


            # AdwFd = self.bn_1d[i](torch.matmul(Ad, feats_d_gnn).permute(0, 2, 1)).permute(0, 2, 1)
            AdwFd = torch.matmul(Ad, feats_d_gnn)
            AdwFd = self.ln_1d[i](AdwFd)


            CT = torch.matmul(feats_d, feats_p.permute(0, 2, 1))
            # CTwFp = self.bn_1d[i](torch.matmul(self.soft_2(CT), feats_p_gnn).permute(0, 2, 1)).permute(0, 2, 1)
            CTwFp = torch.matmul(self.soft_2(CT), feats_p_gnn)
            CTwFp = self.ln_1d[i](CTwFp)


            feats_p = feats_p_gnnDWC + CwFd + feats_p
            feats_d = AdwFd + CTwFp + feats_d
        Feats_fin = torch.cat((feats_p, feats_d), dim=1)

        return Feats_fin

    def forward(self, feats_d, Ad, feats_p, layer=4):
        Feats_fin=self.Style_Exract(feats_d,Ad, feats_p, layer)
        return Feats_fin
