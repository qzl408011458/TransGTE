import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from backbones.GCN_Modules import GATNet, GCN, ChebNet

class SpatialGate(nn.Module):
    def __init__(self, emb_size):
        super(SpatialGate, self).__init__()
        self.linear_gate_sp1 = nn.Linear(emb_size * 2, emb_size)
        self.linear_c_stat = nn.Linear(emb_size, emb_size)
        self.linear_c_sp = nn.Linear(emb_size, emb_size)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Tanh()

    def forward(self, stat_embed, sp_embed):
        # Global spatial information embed
        gate_sp1 = self.act1(self.linear_gate_sp1(torch.cat([sp_embed, stat_embed], dim=-1)))
        gate_sp2 = 1 - gate_sp1
        c_stat = self.act2(self.linear_c_stat(stat_embed))
        c_sp = self.act2(self.linear_c_sp(sp_embed))
        c = gate_sp2 * c_stat + gate_sp1 * c_sp

        return c, gate_sp1

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(0)].clone().detach().transpose(0, 1)


class TransGTE(nn.Module):
    def __init__(self, model_dim=256, enc_nhead=8, dec_nhead=8, enc_layers=1, dec_layers=1,
                 gcn_hid_dim=4, matrix_adjacent=0, g=50, n_head=1, K=2, GCNtype='GCN', device='cuda'):
        super(TransGTE, self).__init__()
        self.embed_size = model_dim
        self.g = g
        self.device = device
        self.__init_embeddingLayer(model_dim, g)
        self.__init_encoder(model_dim, enc_nhead, enc_layers)
        self.__init_decoder(model_dim, dec_nhead, dec_layers)
        self.pos_encoder = PositionalEncoding(model_dim, n_position=10000)

        # model_dim + 16 + 16
        self.ffc = nn.Sequential(
            nn.Linear(model_dim + 16 * 2, g * g),
            nn.Softmax(-1)
        )

        self.matrix_adj = torch.tensor(matrix_adjacent, dtype=torch.float, device=device)
        self.spatialGate_grid = SpatialGate(emb_size=model_dim)

        if GCNtype == 'GCN':
            self.__init_GCN(gcn_hid_dim, K)
        if GCNtype == 'GAT':
            self.__init_GAT(n_head, gcn_hid_dim)
        if GCNtype == 'Cheb':
            self.__init_Cheb(K, gcn_hid_dim)
        self.GCNtype = GCNtype

    def __init_embeddingLayer(self, mdim, g):
        self.emb_hour = nn.Embedding(24, 16)
        self.emb_weekday = nn.Embedding(7, 16)
        # self.emb_grid = nn.Embedding(2500 + 1, mdim)

        self.emb_params_grid = nn.Parameter(torch.randn(g * g + 1, mdim))


    def __init_encoder(self, mdim, nhead, layers):
        encoder_layer = nn.TransformerEncoderLayer(d_model=mdim, nhead=nhead, dim_feedforward=1024)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers,)
                                             # norm=nn.LayerNorm(mdim, eps=1e-6))


    def __init_decoder(self, mdim, nhead, layers):
        decoder_layer = nn.TransformerDecoderLayer(d_model=mdim, nhead=nhead, dim_feedforward=1024)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers,)
                                             # norm=nn.LayerNorm(mdim, eps=1e-6))

    def __init_GCN(self, gcn_hid_dim, layers):
        self.gcnNet = GCN(in_c=1, hid_c=gcn_hid_dim, out_c=1, layers=layers)
    def __forward_GCN(self, input):
        return self.gcnNet(input, self.device)

    def __init_GAT(self, n_head, gcn_hid_dim):
        self.gatNet = GATNet(in_c=1, hid_c=gcn_hid_dim, out_c=1, n_heads=n_head)
    def __forward_GAT(self, input):
        return self.gatNet(input, self.device)

    def __init_Cheb(self, K, gcn_hid_dim):
        self.chebNet = ChebNet(in_c=1, hid_c=gcn_hid_dim, out_c=1, K=K)
    def __forward_Cheb(self, input):
        return self.chebNet(input, self.device)

    def __GCN_Module_forward(self, input_od):
        input_od = F.softmax(input_od.masked_fill(input_od == 0, -1e9), dim=1)
        input_g = {
            'flow_x': input_od,
            'graph': self.matrix_adj
        }
        # Capture spatial features
        if self.GCNtype == 'GCN':
            output_g = self.__forward_GCN(input_g)
            output_g = F.softmax(output_g.masked_fill(output_g == 0, -1e9), dim=1)
        if self.GCNtype == 'GAT':
            output_g = self.__forward_GAT(input_g)
            output_g = F.softmax(output_g.masked_fill(output_g == 0, -1e9), dim=1)
        if self.GCNtype == 'Cheb':
            output_g = self.__forward_Cheb(input_g)
            output_g = F.softmax(output_g.masked_fill(output_g == 0, -1e9), dim=1)
        return output_g

    def forward(self, x):
        bat, seq = x[0].shape
        grid_seq, gps_seq, h1_grid, in_gcn, w, h, mask = x

        out_emb_h = self.emb_hour(h)
        out_emb_w = self.emb_weekday(w)
        # out_emb_grid_seq = self.emb_grid(grid_seq)

        out_gcn = self.__GCN_Module_forward(in_gcn)
        out_gcn = torch.cat([torch.zeros((bat, 1, 1), device=self.device, dtype=torch.float),
                             out_gcn], dim=1)
        output_gemb_grid = torch.matmul(out_gcn[:, :, 0], self.emb_params_grid)


        out_emb_cell = torch.matmul(h1_grid, self.emb_params_grid)
        output_gemb_grid = output_gemb_grid.unsqueeze(1).repeat(1, seq, 1)
        out_gatedEmb_grid, gate_sp = self.spatialGate_grid(out_emb_cell, output_gemb_grid)
        anfg_input = torch.cat([out_emb_cell, output_gemb_grid], dim=-1)

        in_enc = out_gatedEmb_grid[:, :-1].transpose(0, 1)
        in_dec = out_gatedEmb_grid[:, -1:].transpose(0, 1)

        # B, S, 64
        out_enc = self.encoder(self.pos_encoder(in_enc))
        out_dec = self.decoder(self.pos_encoder(in_dec), out_enc)

        out = self.ffc(torch.cat([out_dec[-1], out_emb_h, out_emb_w], dim=-1))
        return out




