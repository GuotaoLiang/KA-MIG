import torch

import pickle

import torch.nn as nn


from torch_geometric.nn import GATConv, GCNConv


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
def conv_nd(*args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """

    return nn.Conv1d(*args, **kwargs)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=heads)
        self.gat3 = GATConv(hidden_channels * heads, out_channels, heads=heads)

    def forward(self, inputs_idx, x, edge_index, edge_weight):
        edge_index = edge_index.to(x.device)
        # res_embs = [self.get_inputs_idx_emb(inputs_idx, x)]
        res_embs = []
        x = self.gat1(x, edge_index).relu()
        # x = torch.relu(x)
        res_embs.append(self.get_inputs_idx_emb(inputs_idx, x))
        x = self.gat2(x, edge_index).relu()
        res_embs.append(self.get_inputs_idx_emb(inputs_idx, x))
        x = self.gat3(x, edge_index)
        res_embs.append(self.get_inputs_idx_emb(inputs_idx, x))        
        res_embs = torch.cat(res_embs, dim=0).mean(0)
        return res_embs

    def get_inputs_idx_emb(self, inputs_idx, x):
        x_ = torch.cat((x, torch.zeros((1, x.shape[-1])).to(x.device)), dim=0)
        return x_[inputs_idx].unsqueeze(0)

class GCN(torch.nn.Module):
    def __init__(self, in_channels,
                 hidden_channels,
                 out_channels,):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
    def forward(self, inputs_idx, x, edge_index, edge_weight):
        edge_index = edge_index.to(x.device)
        edge_weight = edge_weight.to(x.device).type_as(x)

        # res_embs = [self.get_inputs_idx_emb(inputs_idx, x)]
        res_embs = []

        x = self.conv1(x, edge_index, edge_weight).relu()
        res_embs.append(self.get_inputs_idx_emb(inputs_idx, x))

        x = self.conv2(x, edge_index, edge_weight).relu()
        res_embs.append(self.get_inputs_idx_emb(inputs_idx, x))

        x = self.conv3(x, edge_index, edge_weight).relu()
        res_embs.append(self.get_inputs_idx_emb(inputs_idx, x))

        res_embs = torch.cat(res_embs, dim=0).mean(0)
        return res_embs

    def get_inputs_idx_emb(self, inputs_idx, x):
        x_ = torch.cat((x, torch.zeros((1, x.shape[-1])).to(x.device)), dim=0)
        return x_[inputs_idx].unsqueeze(0)


class PosNegCodebook(nn.Module):
    def __init__(self, graph, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, padding=0)
        self.graph = graph
    def forward(self, class_ids, codebook_embs):
        # 
        num, dim = codebook_embs.shape
        class_graph = self.graph.to(class_ids.device)[class_ids]
        codebook_embs_conv = self.conv1(codebook_embs.unsqueeze(0).permute(0, 2, 1)).permute(0, 2, 1).reshape(num, dim)
        # 
        pos_neg_emb = class_graph.matmul(codebook_embs_conv)  # (B, 256, 1024) *  1024 * 256
        # 
        return pos_neg_emb


class GraphAdapter(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 transformer_layers,
                 co_occurrence_graph_path,
                 replace_graph_path,
                 position_negative_codebook_graph_path,
                 gnn='gcn',
                 ):
        super().__init__()
        self.vocab_size = vocab_size,
        self.hidden_size = hidden_size
        self.transformer_layers = transformer_layers

        self.co_edge_index, self.co_edge_weight = self.load_graph(co_occurrence_graph_path)
        self.re_edge_index, self.re_edge_weight = self.load_graph(replace_graph_path)
        pos_neg_codebook_graph = self.load_graph(position_negative_codebook_graph_path)
        padding_graph = torch.zeros_like(pos_neg_codebook_graph[0]).unsqueeze(0)
        pos_neg_codebook_graph = torch.cat([pos_neg_codebook_graph, padding_graph], dim=0)
        if gnn.lower() == 'gcn':
            self.co_graph_layer = GCN(hidden_size, hidden_size, hidden_size)
            self.re_graph_layer = GCN(hidden_size, hidden_size, hidden_size)
        elif gnn.lower() == 'gat':
            print("======================================using gat====================================")
            self.co_graph_layer = GAT(hidden_size, hidden_size, hidden_size)
            self.re_graph_layer = GAT(hidden_size, hidden_size, hidden_size)

        self.pos_neg_codebook_graph_layer = PosNegCodebook(pos_neg_codebook_graph, hidden_size, hidden_size)

        self.zero_convs = nn.ModuleList(
            [
                zero_module(conv_nd(hidden_size, hidden_size, 1, padding=0)) for _ in range(transformer_layers)
            ]
        )
        self.zero_convs_pos = nn.ModuleList(
            [
                zero_module(conv_nd(hidden_size, hidden_size, 1, padding=0)) for _ in range(transformer_layers)
            ]
        )


    def load_graph(self, graph_path):
        with open(graph_path, 'rb') as f:
            return pickle.load(f)


    def forward(self, word_embs, input_ids, class_ids, mask):
        
        co_re_mask = ~mask
        # word_embs = self.word_embeddings.weight
        zero_padding = torch.zeros(input_ids.shape[0], 1, word_embs.shape[-1]).to(word_embs.device)
        ones_padding = torch.ones(input_ids.shape[0], 1, word_embs.shape[-1]).to(input_ids.device)
        co_graph_word_embs = self.co_graph_layer(input_ids, word_embs, self.co_edge_index, self.co_edge_weight)
        re_graph_word_embs = self.re_graph_layer(input_ids, word_embs, self.re_edge_index, self.re_edge_weight)
        co_graph_word_embs = co_graph_word_embs.permute(0, 2, 1)
        re_graph_word_embs = re_graph_word_embs.permute(0, 2, 1)

        co_graph_word_embs_layers = [self.zero_convs[i](co_graph_word_embs).permute(0, 2, 1) for i in range(self.transformer_layers)]
        re_graph_word_embs_layers = [self.zero_convs[i](re_graph_word_embs).permute(0, 2, 1) for i in range(self.transformer_layers)]

        co_graph_word_embs_layers = [torch.cat((zero_padding, i * co_re_mask), dim=1) for i in co_graph_word_embs_layers]
        re_graph_word_embs_layers = [torch.cat((zero_padding, i * co_re_mask), dim=1) for i in re_graph_word_embs_layers]


        pos_neg_embs = self.pos_neg_codebook_graph_layer(class_ids, word_embs)
        pos_neg_embs = pos_neg_embs.permute(0, 2, 1)
        
        pos_neg_embs_layers = [self.zero_convs_pos[i](pos_neg_embs).permute(0, 2, 1) for i in range(self.transformer_layers)]
        pos_neg_embs_layers = [torch.cat((zero_padding, i), dim=1) for i in pos_neg_embs_layers]
        return co_graph_word_embs_layers, re_graph_word_embs_layers, pos_neg_embs_layers