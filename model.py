import torch
import torch.nn as nn
from torch.nn import Conv2d, Parameter
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm
from torch_geometric.nn import SAGEConv, GATConv, JumpingKnowledge
import math


def nconv(x, A):
    return torch.einsum('bcnt,nm->bcmt', (x, A)).contiguous()


class Creator(nn.Module):
    def __init__(self, num_of_x, node_num=170, seq_len=12, graph_dim=16,
                 tcn_dim=[10], choice=[1,1],pred_len=6):
        super(Creator, self).__init__()
        self.node_num = node_num
        self.seq = seq_len
        self.num_of_x = num_of_x
        self.seq_len = seq_len
        self.graph_dim = graph_dim
        self.tcn_dim = tcn_dim
        self.output_dim = (np.sum(choice)) * graph_dim  # TODO： 要改
        self.choice = choice
        self.pred_len = pred_len
        self.seq_linear = nn.Linear(in_features=self.seq_len * self.num_of_x, out_features=self.seq_len * self.num_of_x)
        self.output_linear = nn.Linear(in_features=32, out_features=self.seq_len*self.num_of_x)

        if choice[0] == 1:
            self.sp_origin = nn.Linear(in_features=seq_len*num_of_x, out_features=graph_dim)
            self.sp_gconv1 = GATConv(seq_len*num_of_x, graph_dim, heads=3, concat=False)
            self.sp_gconv2 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
            self.sp_gconv3 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
            self.sp_gconv4 = GATConv(graph_dim, graph_dim, heads=1, concat=False)
            self.sp_source_embed = nn.Parameter(torch.Tensor(self.node_num, 12))
            self.sp_target_embed = nn.Parameter(torch.Tensor(12, self.node_num))
            self.sp_linear_1 = nn.Linear(self.seq_len*num_of_x, self.graph_dim)
            self.sp_linear_2 = nn.Linear(self.graph_dim, self.graph_dim)
            self.sp_linear_3 = nn.Linear(self.graph_dim, self.graph_dim)
            self.sp_linear_4 = nn.Linear(self.graph_dim, self.graph_dim)

            nn.init.xavier_uniform_(self.sp_source_embed)
            nn.init.xavier_uniform_(self.sp_target_embed)

            if choice[1] == 1:
                self.dtw_origin = nn.Linear(in_features=seq_len*num_of_x, out_features=graph_dim)
                self.dtw_gconv1 = GATConv(seq_len*num_of_x, graph_dim, heads=3, concat=False)
                self.dtw_gconv2 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
                self.dtw_gconv3 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
                self.dtw_gconv4 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
                self.dtw_source_embed = nn.Parameter(torch.Tensor(self.node_num, 12))
                self.dtw_target_embed = nn.Parameter(torch.Tensor(12, self.node_num))
                self.dtw_linear_1 = nn.Linear(self.seq_len*num_of_x, self.graph_dim)
                self.dtw_linear_2 = nn.Linear(self.graph_dim, self.graph_dim)
                self.dtw_linear_3 = nn.Linear(self.graph_dim, self.graph_dim)
                self.dtw_linear_4 = nn.Linear(self.graph_dim, self.graph_dim)

                nn.init.xavier_uniform_(self.dtw_source_embed)
                nn.init.xavier_uniform_(self.dtw_target_embed)

    def forward(self, x, edge_index, dtw_edge_index):
        output_list = [0, 0]

        if self.choice[0] == 1:
            x = self.seq_linear(x) + x

            sp_learned_matrix =F.softmax(F.relu(torch.mm(self.sp_source_embed, self.sp_target_embed)), dim=1)

            sp_gout_1 = self.sp_gconv1(x, edge_index)  # GAT
            adp_input_1 = torch.reshape(x, (-1, self.node_num, self.seq_len*self.num_of_x))
            sp_adp_1 = self.sp_linear_1(sp_learned_matrix.matmul(F.dropout(adp_input_1, p=0.1)))  # 门控值
            sp_adp_1 = torch.reshape(sp_adp_1, (-1, self.graph_dim))
            sp_origin = self.sp_origin(x)  # 10880,16
            sp_output_1 = torch.tanh(sp_gout_1) * torch.sigmoid(sp_adp_1) + sp_origin * (1 - torch.sigmoid(sp_adp_1))

            sp_gout_2 = self.sp_gconv2(torch.tanh(sp_output_1), edge_index)
            adp_input_2 = torch.reshape(torch.tanh(sp_output_1), (-1, self.node_num, self.graph_dim))
            sp_adp_2 = self.sp_linear_2(sp_learned_matrix.matmul(F.dropout(adp_input_2, p=0.1)))
            sp_adp_2 = torch.reshape(sp_adp_2, (-1, self.graph_dim))
            sp_output_2 = F.leaky_relu(sp_gout_2) * torch.sigmoid(sp_adp_2) + sp_output_1 * (
                        1 - torch.sigmoid(sp_adp_2))

            sp_gout_3 = self.sp_gconv3(F.relu(sp_output_2), edge_index)
            adp_input_3 = torch.reshape(F.relu(sp_output_2), (-1, self.node_num, self.graph_dim))
            sp_adp_3 = self.sp_linear_3(sp_learned_matrix.matmul(F.dropout(adp_input_3, p=0.1)))
            sp_adp_3 = torch.reshape(sp_adp_3, (-1, self.graph_dim))
            sp_output_3 = F.relu(sp_gout_3) * torch.sigmoid(sp_adp_3) + sp_output_2 * (1 - torch.sigmoid(sp_adp_3))

            sp_gout_4 = self.sp_gconv4(F.relu(sp_output_3), edge_index)
            adp_input_4 = torch.reshape(F.relu(sp_output_3), (-1, self.node_num, self.graph_dim))
            sp_adp_4 = self.sp_linear_4(sp_learned_matrix.matmul(F.dropout(adp_input_4, p=0.1)))
            sp_adp_4 = torch.reshape(sp_adp_4, (-1, self.graph_dim))
            sp_output_4 = F.relu(sp_gout_4) * torch.sigmoid(sp_adp_4) + sp_output_3 * (1 - torch.sigmoid(sp_adp_4))

            sp_output = torch.reshape(sp_output_4, (-1, self.node_num, self.graph_dim))

            output_list[0] = sp_output

            if self.choice[1] == 1:
                x = self.seq_linear(x) + x

                dtw_learned_matrix = F.softmax(F.relu(torch.mm(self.dtw_source_embed, self.dtw_target_embed)), dim=1)

                dtw_gout_1 = self.dtw_gconv1(x, dtw_edge_index)
                adp_input_1 = torch.reshape(x, (-1, self.node_num, self.seq_len*self.num_of_x))  # 64,170,12
                dtw_adp_1 = self.dtw_linear_1(dtw_learned_matrix.matmul(F.dropout(adp_input_1, p=0.1)))  # 64,170,16
                dtw_adp_1 = torch.reshape(dtw_adp_1, (-1, self.graph_dim))
                dtw_origin = self.dtw_origin(x)
                dtw_output_1 = torch.tanh(dtw_gout_1) * torch.sigmoid(dtw_adp_1) + dtw_origin * (
                            1 - torch.sigmoid(dtw_adp_1))

                dtw_gout_2 = self.dtw_gconv2(torch.tanh(dtw_output_1), dtw_edge_index)
                adp_input_2 = torch.reshape(torch.tanh(dtw_output_1), (-1, self.node_num, self.graph_dim))
                dtw_adp_2 = self.dtw_linear_2(dtw_learned_matrix.matmul(F.dropout(adp_input_2, p=0.1)))
                dtw_adp_2 = torch.reshape(dtw_adp_2, (-1, self.graph_dim))
                dtw_output_2 = F.leaky_relu(dtw_gout_2) * torch.sigmoid(dtw_adp_2) + dtw_output_1 * (
                            1 - torch.sigmoid(dtw_adp_2))

                dtw_gout_3 = self.dtw_gconv3(F.relu(dtw_output_2), dtw_edge_index)
                adp_input_3 = torch.reshape(F.relu(dtw_output_2), (-1, self.node_num, self.graph_dim))
                dtw_adp_3 = self.dtw_linear_3(dtw_learned_matrix.matmul(F.dropout(adp_input_3, p=0.1)))
                dtw_adp_3 = torch.reshape(dtw_adp_3, (-1, self.graph_dim))
                dtw_output_3 = F.relu(dtw_gout_3) * torch.sigmoid(dtw_adp_3) + dtw_output_2 * (
                            1 - torch.sigmoid(dtw_adp_3))

                dtw_gout_4 = self.dtw_gconv4(F.relu(dtw_output_3), dtw_edge_index)
                adp_input_4 = torch.reshape(F.relu(dtw_output_3), (-1, self.node_num, self.graph_dim))
                dtw_adp_4 = self.dtw_linear_4(dtw_learned_matrix.matmul(F.dropout(adp_input_4, p=0.1)))
                dtw_adp_4 = torch.reshape(dtw_adp_4, (-1, self.graph_dim))
                dtw_output_4 = F.relu(dtw_gout_4) * torch.sigmoid(dtw_adp_4) + dtw_output_3 * (
                            1 - torch.sigmoid(dtw_adp_4))

                dtw_output = torch.reshape(dtw_output_4, (-1, self.node_num, self.graph_dim))  # 64,170,16
                output_list[1] = dtw_output

            step = 0
            for i in range(0, len(self.choice)):
                if self.choice[i] == 1 and step == 0:
                    cell_output = output_list[i]
                    step += 1
                elif self.choice[i] == 1:
                    cell_output = torch.cat((cell_output, output_list[i]), dim=2)  # 32, 170, 48

            cell_output = torch.reshape(cell_output, (-1, self.output_dim))  # (32*170,48)
            output = self.output_linear(cell_output).reshape(-1,self.node_num,self.seq_len*self.num_of_x).unsqueeze(1)  # 10800,12

            return output


class Diffusion_GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=1):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.conv = Conv2d(c_in, c_out, (1, 1), padding=(
            0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list):  # (32, 64, 170, 6)
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.conv(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
    
    
def sample_gumbel(device, shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(device, logits, temperature,  eps=1e-10):
    sample = sample_gumbel(device, logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(device, logits, temperature, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(
        device, logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


class Encoder(nn.Module):
    def __init__(self, device, channels, num_nodes, seq_len, num_of_x, adj, dtw, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.node = num_nodes
        self.device = device
        self.adj = adj
        self.dtw = dtw
        self.fc0 = nn.Linear(channels, num_nodes)
        self.fc1 = nn.Linear(num_nodes, 2*num_nodes)
        self.fc2 = nn.Linear(2*num_nodes, num_nodes)
        self.diffusion_conv = Diffusion_GCN(channels, channels, dropout, support_len=1)
        self.Creator = Creator(num_of_x=num_of_x, node_num=num_nodes, seq_len=int(seq_len/2), graph_dim=16, choice=[1, 1]
                            , pred_len=12)
        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=channels,
                                    kernel_size=(1, 1))


    def forward(self, x1):  # x: (B, F, N, T)   x1: (B, 1, N, T)   adj: (N, N)
        input = x1
        input = input.reshape(input.size(0) * input.size(2), -1)
        output = self.Creator(input, edge_index=self.adj.to('cuda'), dtw_edge_index=self.dtw.to('cuda'))  # 32,1,170,6
        output1 = self.start_conv(output)
        x = output1[-1,:,:,:].sum(2).permute(1, 0)
        x = self.fc0(x)
        x = torch.tanh(x)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.log(F.softmax(x, dim=-1))
        x = gumbel_softmax(self.device, x, temperature=0.5, hard=True)  # (170, 170)
        mask = torch.eye(x.shape[0], x.shape[0]).bool().to(device=self.device)
        x.masked_fill_(mask, 0)
        return x, output


class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, :, :, ::2]

    def odd(self, x):
        return x[:, :, :, 1::2]

    def forward(self, x):
        return (self.even(x), self.odd(x))


class IS_BLOCK(nn.Module):
    def __init__(self, device, channels, seq_len, num_of_x, adj, dtw, splitting=True,
                 num_nodes=170, dropout=0.25, pre_adj=None, pre_adj_len=1
                 ):
        super(IS_BLOCK, self).__init__()
        device = device
        self.dropout = dropout
        self.pre_adj_len = pre_adj_len
        self.num_nodes = num_nodes
        self.splitting = splitting
        self.pre_graph = pre_adj or []
        self.split = Splitting()
        self.seq_len = seq_len
        self.num_of_x = num_of_x

        Conv1 = []
        Conv2 = []
        Conv3 = []
        Conv4 = []
        pad_l = 3
        pad_r = 3
        self.pre_adj_len = 1

        Conv1 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh(),
        ]
        Conv2 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh()
        ]

        Conv4 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh()
        ]
        Conv3 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh()
        ]

        self.conv1 = nn.Sequential(*Conv1)
        self.conv2 = nn.Sequential(*Conv2)
        self.conv3 = nn.Sequential(*Conv3)
        self.conv4 = nn.Sequential(*Conv4)

        self.a = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)

        self.Encoder = Encoder(
            device, channels, num_nodes, seq_len, num_of_x, adj, dtw)

        self.diffusion_conv = Diffusion_GCN(
            channels, channels, dropout, support_len=self.pre_adj_len)



    def forward(self, x, x_1):
        if self.splitting:
            (x_even, x_odd) = self.split(x)  # (32, 64, 170, 6)
            (x_1even, x_1odd) = self.split(x_1)  # (32,1,170,6)
        else:
            (x_even, x_odd) = x
            (x_1even, x_1odd) = x_1

        x1 = self.conv1(x_even)
        learn_adj, output = self.Encoder(x_1even)
        x1 = x1+self.diffusion_conv(x1, [learn_adj])
        d = x_odd.mul(torch.tanh(x1))

        x2 = self.conv2(x_odd)
        learn_adj, output= self.Encoder(output)
        x2 = x2+self.diffusion_conv(x2, [learn_adj])
        c = x_even.mul(torch.tanh(x2))

        x3 = self.conv3(c)
        learn_adj, output = self.Encoder(output)
        x3 = x3+self.diffusion_conv(x3, [learn_adj])
        x_odd_update = d - x3

        x4 = self.conv4(d)
        learn_adj, output1 = self.Encoder(output)
        x4 = x4+self.diffusion_conv(x4, [learn_adj])
        x_even_update = c + x4
        output = torch.cat((output, output1),dim=-1)

        return (x_even_update, x_odd_update, learn_adj, output)


class IS_Tree(nn.Module):
    def __init__(self, device, num_nodes, channels, adj, dtw, dropout, seq_len, num_of_x, pre_adj=None, pre_adj_len=1):
        super().__init__()
        self.pre_graph = pre_adj or []

        self.IS1 = IS_BLOCK(splitting=True, channels=channels, seq_len=seq_len, num_of_x=num_of_x,device=device,
                            pre_adj=pre_adj, num_nodes=num_nodes, dropout=dropout, pre_adj_len=pre_adj_len,
                            adj=adj,dtw=dtw)
        self.IS2 = IS_BLOCK(splitting=True, channels=channels, seq_len=seq_len, num_of_x=num_of_x,device=device,
                            pre_adj=pre_adj, num_nodes=num_nodes, dropout=dropout, pre_adj_len=pre_adj_len,
                            adj=adj,dtw=dtw)
        self.IS3 = IS_BLOCK(splitting=True, channels=channels, seq_len=seq_len, num_of_x=num_of_x,device=device,
                            pre_adj=pre_adj, num_nodes=num_nodes, dropout=dropout, pre_adj_len=pre_adj_len,
                            adj=adj,dtw=dtw)

        self.a = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)
        self.b = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)
        self.c = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)

    def concat(self, even, odd):
        even = even.permute(3, 1, 2, 0)
        odd = odd.permute(3, 1, 2, 0)
        len = even.shape[0]
        _ = []
        for i in range(len):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        return torch.cat(_, 0).permute(3, 1, 2, 0)

    def forward(self, x, x1):
        x_even_update1, x_odd_update1, dadj1, x2 = self.IS1(x, x1)
        x_even_update2, x_odd_update2, dadj2, x3 = self.IS2(x_even_update1, x2)
        x_even_update3, x_odd_update3, dadj3, x4 = self.IS3(x_odd_update1, x3)
        concat1 = self.concat(x_even_update2, x_odd_update2)
        concat2 = self.concat(x_even_update3, x_odd_update3)
        concat0 = self.concat(concat1, concat2)
        adj = dadj1*self.a+dadj2*self.b+dadj3*self.c
        return concat0, adj


class ISGCN_WEEK(nn.Module):
    def __init__(self, device, num_nodes, seq_len, num_of_weeks, channels, adj, dtw,dropout=0.25, pre_adj=None):
        super().__init__()

        self.dropout = dropout
        self.num_nodes = num_nodes
        self.input_len = 12
        self.output_len = 12
        input_channel = 1
        apt_size = 10
        self.seq_len = seq_len
        self.dtw = dtw
        self.adj = adj
        self.pre_graph = pre_adj or []
        self.pre_adj_len = len(self.pre_graph)+1
        self.num_of_weeks = num_of_weeks

        aptinit = pre_adj[0]
        nodevecs = self.svd_init(apt_size, aptinit)
        self.nodevec1, self.nodevec2 = [
            Parameter(n.to(device), requires_grad=True) for n in nodevecs]

        self.a = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)

        self.start_conv = nn.Conv2d(in_channels=input_channel,
                                    out_channels=channels,
                                    kernel_size=(1, 1))

        self.tree = IS_Tree(
            device=device,
            channels=channels,
            num_nodes=self.num_nodes,
            dropout=self.dropout,
            pre_adj_len=self.pre_adj_len,
            pre_adj=pre_adj,
            seq_len=seq_len,
            num_of_x=num_of_weeks,
            adj=adj,
            dtw=dtw
        )

        self.diffusion_conv = Diffusion_GCN(
            channels, channels, dropout, support_len=self.pre_adj_len)

        self.Conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=256,
                               kernel_size=(1, 1), stride=(1, 1))
        self.Conv2 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=(1, 12*num_of_weeks), stride=(1, 1))
        self.Conv3 = nn.Conv2d(in_channels=512,
                               out_channels=12,
                               kernel_size=(1, 1), stride=(1, 1))
        self.W = nn.Parameter(torch.empty(12, self.num_nodes))
        nn.init.xavier_uniform(self.W)

    @staticmethod
    def svd_init(apt_size, aptinit):
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(
            p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    def forward(self, input):
        x1 = input
        x = self.start_conv(x1)
        skip = x
        x, dadj = self.tree(x, x1)
        x = skip + x

        a = torch.mm(self.nodevec1, self.nodevec2)
        adaptive_adj = F.softmax(F.relu(a), dim=1)
        adj = self.a*adaptive_adj+(1-self.a)*dadj
        adj = self.pre_graph + [adj]

        gcn = self.diffusion_conv(x, adj)
        x = gcn + x
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = self.Conv3(x).squeeze()
        # x = x*self.W
        return x.unsqueeze(-1)



class ISGCN_DAY(nn.Module):
    def __init__(self, device, num_nodes, seq_len, num_of_days, channels, adj,dtw, dropout=0.25, pre_adj=None):
        super().__init__()

        self.dropout = dropout
        self.num_nodes = num_nodes
        self.input_len = 12
        self.output_len = 12
        input_channel = 1
        apt_size = 10
        self.seq_len = seq_len
        self.dtw = dtw
        self.adj = adj
        self.pre_graph = pre_adj or []
        self.pre_adj_len = len(self.pre_graph)+1
        self.num_of_days = num_of_days

        aptinit = pre_adj[0]
        nodevecs = self.svd_init(apt_size, aptinit)
        self.nodevec1, self.nodevec2 = [
            Parameter(n.to(device), requires_grad=True) for n in nodevecs]

        self.a = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)

        self.start_conv = nn.Conv2d(in_channels=input_channel,
                                    out_channels=channels,
                                    kernel_size=(1, 1))

        self.tree = IS_Tree(
            device=device,
            channels=channels,
            num_nodes=self.num_nodes,
            dropout=self.dropout,
            pre_adj_len=self.pre_adj_len,
            pre_adj=pre_adj,
            seq_len=seq_len,
            num_of_x=num_of_days,
            adj=adj,
            dtw=dtw
        )

        self.diffusion_conv = Diffusion_GCN(
            channels, channels, dropout, support_len=self.pre_adj_len)

        self.Conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=256,
                               kernel_size=(1, 1), stride=(1, 1))
        self.Conv2 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=(1, 12*self.num_of_days), stride=(1, 1))
        self.Conv3 = nn.Conv2d(in_channels=512,
                               out_channels=12,
                               kernel_size=(1, 1), stride=(1, 1))
        self.W = nn.Parameter(torch.empty(12, self.num_nodes))
        nn.init.xavier_uniform(self.W)

    @staticmethod
    def svd_init(apt_size, aptinit):
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(
            p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    def forward(self, input):
        x1 = input
        x = self.start_conv(x1)
        skip = x
        x, dadj = self.tree(x, x1)
        x = skip + x

        adaptive_adj = F.softmax(
            F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        adj = self.a*adaptive_adj+(1-self.a)*dadj
        adj = self.pre_graph + [adj]

        gcn = self.diffusion_conv(x, adj)
        x = gcn + x
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = self.Conv3(x).squeeze()
        # x = x*self.W
        return x.unsqueeze(-1)


class ISGCN_HOUR(nn.Module):
    def __init__(self, device, num_nodes, seq_len, num_of_hours, channels, adj,dtw, dropout=0.25, pre_adj=None):
        super().__init__()

        self.dropout = dropout
        self.num_nodes = num_nodes
        self.input_len = 12
        self.output_len = 12
        input_channel = 1
        apt_size = 10
        self.seq_len = seq_len*num_of_hours
        self.dtw = dtw
        self.adj = adj
        self.pre_graph = pre_adj or []
        self.pre_adj_len = len(self.pre_graph)+1

        aptinit = pre_adj[0]
        nodevecs = self.svd_init(apt_size, aptinit)
        self.nodevec1, self.nodevec2 = [
            Parameter(n.to(device), requires_grad=True) for n in nodevecs]

        self.a = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)

        self.start_conv = nn.Conv2d(in_channels=input_channel,
                                    out_channels=channels,
                                    kernel_size=(1, 1))

        self.tree = IS_Tree(
            device=device,
            channels=channels,
            num_nodes=self.num_nodes,
            dropout=self.dropout,
            pre_adj_len=self.pre_adj_len,
            pre_adj=pre_adj,
            seq_len=seq_len,
            num_of_x=num_of_hours,
            adj=adj,
            dtw=dtw
        )

        self.diffusion_conv = Diffusion_GCN(
            channels, channels, dropout, support_len=self.pre_adj_len)

        self.Conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=256,
                               kernel_size=(1, 1), stride=(1, 1))
        self.Conv2 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=(1, 12*num_of_hours), stride=(1, 1))
        self.Conv3 = nn.Conv2d(in_channels=512,
                               out_channels=12,
                               kernel_size=(1, 1), stride=(1, 1))
        self.W = nn.Parameter(torch.empty(12, self.num_nodes))
        nn.init.xavier_uniform(self.W)

    @staticmethod
    def svd_init(apt_size, aptinit):
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(
            p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    def forward(self, input):
        x1 = input
        x = self.start_conv(x1)
        skip = x
        x, dadj = self.tree(x, x1)
        x = skip + x

        adaptive_adj = F.softmax(
            F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        adj = self.a*adaptive_adj+(1-self.a)*dadj
        adj = self.pre_graph + [adj]

        gcn = self.diffusion_conv(x, adj)
        x = gcn + x
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = self.Conv3(x).squeeze()  # (64,12,170,25)
        # x = x*self.W
        return x.unsqueeze(-1)


class ISGCN(nn.Module):
    def __init__(self, device, num_nodes, seq_len,channels, dropout, adj,dtw, num_of_weeks, num_of_days,num_of_hours, pre_adj=None):
        super().__init__()

        self.dropout = dropout
        self.num_nodes = num_nodes
        self.input_len = 12
        self.output_len = 12
        self.num_levels = 2
        self.groups = 1
        self.channels = channels


        self.pre_graph = pre_adj or []
        self.pre_adj_len = len(self.pre_graph)+1

        self.ISGCN_WEEK = ISGCN_WEEK(device=device, num_nodes=num_nodes, seq_len=seq_len, num_of_weeks=num_of_weeks,
                               channels=channels, dropout=dropout, pre_adj=pre_adj, adj=adj,dtw=dtw)
        self.ISGCN_DAY = ISGCN_DAY(device=device, num_nodes=num_nodes, seq_len=seq_len, num_of_days=num_of_days,
                                channels=channels, dropout=dropout, pre_adj=pre_adj,adj=adj,dtw=dtw)
        self.ISGCN_HOUR = ISGCN_HOUR(device=device, num_nodes=num_nodes, seq_len=seq_len, num_of_hours=num_of_hours,
                                 channels=channels, dropout=dropout, pre_adj=pre_adj,adj=adj,dtw=dtw)

    def forward(self, input):
        x1 = self.ISGCN_WEEK(input['input1'])
        x2 = self.ISGCN_DAY(input['input2'])
        x3 = self.ISGCN_HOUR(input['input3'])
        x = (x3+x1+x2)/3
        return x