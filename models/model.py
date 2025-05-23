import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import knn, knn_graph
import torch_geometric.nn as pyg_nn


class VirtualNode(nn.Module):
    # یک گره مجازی برای انتقال اطلاعات جهانی میان همه گره‌ها
    def __init__(self, hidden_dim):
        super().__init__()
        self.aggregate = nn.Linear(hidden_dim, hidden_dim)
        self.distribute = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        global_context = x.sum(dim=0, keepdim=True)
        global_context = self.aggregate(global_context)
        global_context = self.norm(global_context)
        return x + self.distribute(global_context)



class GraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, dropout_param=0.1):
        super(GraphAttentionLayer, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.position_embedding = nn.Linear(3, hidden_dim)
        
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(p=dropout_param)
        self.hidden_dim = hidden_dim

    def forward(self, x, edge_index, pos):
        if edge_index.shape[1] == 0:
            raise ValueError("edge_index is empty!")
        if x.shape[-1] != self.hidden_dim:
            raise ValueError(f"Input feature dim {x.shape[-1]} doesn't match hidden_dim {self.hidden_dim}")

        pos_emb = self.position_embedding(pos)
        pos_emb = F.normalize(pos_emb, p=2, dim=-1)

        Q = self.norm_q(self.query(x))
        K = self.norm_k(self.key(x))
        V = self.norm_v(self.value(x))

        scale = math.sqrt(self.hidden_dim)
        attention_scores = (Q @ K.T) / scale
        pos_sim = pos_emb @ pos_emb.T
        attention_scores = attention_scores + pos_sim

        mask = torch.zeros_like(attention_scores)
        rows, cols = edge_index
        mask[rows, cols] = 1

        attention_scores = attention_scores.masked_fill(mask == 0, -1e4)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        updated_features = attention_weights @ V
        return updated_features, attention_weights



class AGTBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_param=0.1):
        super(AGTBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_param),
            nn.Linear(output_dim, output_dim)
        )
        self.graph_attention = GraphAttentionLayer(output_dim, dropout_param)
        self.residual = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(p=dropout_param)

    def forward(self, x, edge_index, pos):
        h = self.mlp(x)
        h, attention_weights = self.graph_attention(h, edge_index, pos)
        h = self.dropout(h)
        
        residual = self.residual(x)
        output = self.norm(h + residual)
        return output, attention_weights



class Stage(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_param=0.1):
        super(Stage, self).__init__()
        
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if dropout_param < 0 or dropout_param >= 1:
            raise ValueError("dropout_param must be in [0, 1)")
        
        layers = []
        for i in range(num_layers):
            current_input_dim = input_dim if i == 0 else hidden_dim
            layers.append(AGTBlock(current_input_dim, hidden_dim, dropout_param))
        self.layers = nn.ModuleList(layers)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_param = dropout_param

    def forward(self, x, edge_index, pos):
        self._validate_inputs(x, edge_index, pos)
        attention_weights_all = []
        
        for layer in self.layers:
            x, attention_weights = layer(x, edge_index, pos)
            attention_weights_all.append(attention_weights)
        
        return x, attention_weights_all

    def _validate_inputs(self, x, edge_index, pos):
        if x.dim() != 2:
            raise ValueError(f"Features must be 2D tensor, got {x.dim()}D")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must be shape [2, E]")
        if pos.dim() != 2 or pos.size(-1) != 3:
            raise ValueError("pos must be shape [N, 3]")
        if x.size(0) != pos.size(0):
            raise ValueError("Feature and position count mismatch")
        if x.size(-1) != self.input_dim and x.size(-1) != self.hidden_dim:
            raise ValueError(f"Feature dimension mismatch. Expected {self.input_dim} or {self.hidden_dim}, got {x.size(-1)}")

    def extra_repr(self):
        return f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, num_layers={self.num_layers}, dropout={self.dropout_param}"



class InterpolationStage(nn.Module):
    def __init__(self, decoder_dim, encoder_dim, out_dim, knn_param, dropout_param=0.1):
        super(InterpolationStage, self).__init__()
        
        if knn_param <= 0:
            raise ValueError("knn_param must be positive")
        if dropout_param < 0 or dropout_param >= 1:
            raise ValueError("dropout_param must be in [0, 1)")
        
        self.knn_param = knn_param
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.out_dim = out_dim

        self.query_layer = nn.Linear(encoder_dim, decoder_dim)
        self.key_layer = nn.Linear(decoder_dim, decoder_dim)
        self.value_layer = nn.Linear(decoder_dim, decoder_dim)

        self.combination_mlp = nn.Sequential(
            nn.Linear(decoder_dim + encoder_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_param),
            nn.Linear(out_dim, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(p=dropout_param)

    def forward(self, decoder_features, decoder_pos, encoder_features, encoder_pos, encoder_labels):
        self._validate_inputs(decoder_features, decoder_pos, encoder_features, encoder_pos)

        knn_indices = knn(x=decoder_pos, y=encoder_pos, k=self.knn_param)
        neighbor_indices = knn_indices[1].view(encoder_pos.size(0), self.knn_param)

        neighbor_decoder_features = decoder_features[neighbor_indices]

        query = self.query_layer(encoder_features).unsqueeze(1)
        keys = self.key_layer(neighbor_decoder_features)
        values = self.value_layer(neighbor_decoder_features)

        scores = torch.matmul(query, keys.transpose(1, 2)) / math.sqrt(self.decoder_dim)
        weights = F.softmax(scores, dim=-1)

        aggregated_decoder_features = torch.matmul(weights, values).squeeze(1)

        combined_features = torch.cat([aggregated_decoder_features, encoder_features], dim=-1)
        upsampled_features = self.combination_mlp(combined_features)
        
        output = self.norm(upsampled_features)
        output = self.dropout(output)
        return output, encoder_pos, encoder_labels

    def _validate_inputs(self, decoder_features, decoder_pos, encoder_features, encoder_pos):
        if decoder_features.dim() != 2 or encoder_features.dim() != 2:
            raise ValueError("Features must be 2D tensors")
        if decoder_pos.size(-1) != 3 or encoder_pos.size(-1) != 3:
            raise ValueError("Positions must have 3 coordinates")
        if decoder_features.size(0) != decoder_pos.size(0):
            raise ValueError("Decoder features and positions count mismatch")



class Encoder(nn.Module):
    def __init__(self, input_dim, stages_config, knn_param, dropout_param=0.1):
        super(Encoder, self).__init__()
        
        if not isinstance(stages_config, list) or len(stages_config) == 0:
            raise ValueError("stages_config must be a non-empty list")
        if knn_param <= 0:
            raise ValueError("knn_param must be positive")

        self.knn_param = knn_param
        self.stages = nn.ModuleList()
        self.virtual_nodes = nn.ModuleList()
        self.downsampling_ratios = []

        current_dim = input_dim
        for idx, stage_cfg in enumerate(stages_config):
            if 'hidden_dim' not in stage_cfg or 'num_layers' not in stage_cfg:
                raise ValueError("Stage config must contain hidden_dim and num_layers")
            if idx == 0:
                self.stages.append(
                    nn.Sequential(
                        nn.Linear(current_dim, stage_cfg['hidden_dim']),
                        nn.ReLU(),
                        nn.Dropout(p=dropout_param),
                        nn.Linear(stage_cfg['hidden_dim'], stage_cfg['hidden_dim'])
                    )
                )
            else:
                self.stages.append(
                    Stage(input_dim=current_dim,
                          hidden_dim=stage_cfg['hidden_dim'],
                          num_layers=stage_cfg['num_layers'],
                          dropout_param=dropout_param)
                )

            self.virtual_nodes.append(VirtualNode(stage_cfg['hidden_dim']))
            self.downsampling_ratios.append(stage_cfg.get('downsample_ratio', None))
            current_dim = stage_cfg['hidden_dim']

    def forward(self, x, pos, labels):
        self._validate_inputs(x, pos, labels)
        features = []
        positions = []
        sampled_labels = []
        attention_maps = []

        for stage, virtual_node, ratio in zip(self.stages, self.virtual_nodes, self.downsampling_ratios):
            if ratio is not None:
                x, pos, labels, edge_index = self._downsample(x, pos, labels, ratio)
            else:
                edge_index = knn_graph(pos, k=self.knn_param, loop=False)

            if isinstance(stage, nn.Sequential):
                x = stage(x)
            else:
                x, attention_weights = stage(x, edge_index, pos)
                attention_maps.append(attention_weights)

            x = virtual_node(x)
            features.append(x)
            positions.append(pos)
            sampled_labels.append(labels)
        return features, positions, sampled_labels, attention_maps

    def _downsample(self, x, pos, labels, ratio):
        
        ratio_val = ratio.item() if isinstance(ratio, torch.Tensor) else float(ratio)
        if ratio_val <= 0 or ratio_val > 1:
            raise ValueError(f"Downsample ratio must be in (0, 1], got {ratio_val}")
        
        mask = pyg_nn.fps(pos, ratio=ratio_val)
        x_sampled = x[mask]
        pos_sampled = pos[mask]
        labels_sampled = labels[mask]
        
        edge_index = knn_graph(pos_sampled, k=self.knn_param, loop=False)
        return x_sampled, pos_sampled, labels_sampled, edge_index

    def _validate_inputs(self, x, pos, labels):
        if x.dim() != 2:
            raise ValueError(f"Features must be 2D tensor, got {x.dim()}D")
        if pos.dim() != 2 or pos.size(-1) != 3:
            raise ValueError("Positions must be shape [N, 3]")
        if labels.dim() != 1:
            raise ValueError("Labels must be 1D tensor")
        if x.size(0) != pos.size(0) or x.size(0) != labels.size(0):
            raise ValueError("Inputs must have same number of points")



class Decoder(nn.Module):
    def __init__(self, main_output_dim, stages_config, knn_param, dropout_param=0.1):
        super(Decoder, self).__init__()
        
        if not isinstance(stages_config, list) or len(stages_config) < 2:
            raise ValueError("stages_config must be a list with at least 2 stages")
        if knn_param <= 0:
            raise ValueError("knn_param must be positive")

        self.knn_param = knn_param
        self.stages = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        for i in range(len(stages_config)-1):
            encoder_stage = stages_config[-(i+1)]
            prev_stage = stages_config[-(i+2)]
            output_dim = prev_stage['hidden_dim']
            self.stages.append(
                InterpolationStage(
                    decoder_dim=encoder_stage['hidden_dim'],
                    encoder_dim=output_dim,
                    out_dim=output_dim,
                    knn_param=knn_param,
                    dropout_param=dropout_param
                )
            )
            self.skip_connections.append(
                nn.Sequential(
                    nn.Linear(output_dim, output_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_param)
                )
            )
        self.final_mlp = nn.Sequential(
            nn.Linear(stages_config[0]['hidden_dim'], main_output_dim),
            nn.Dropout(p=dropout_param)
        )

    def forward(self, encoder_features, positions, sampled_labels):
        self._validate_inputs(encoder_features, positions, sampled_labels)
        
        x = encoder_features[-1]
        pos = positions[-1]
        labels = sampled_labels[-1]

        for i, (stage, skip_conn) in enumerate(zip(self.stages, self.skip_connections)):
            skip_features = encoder_features[-(i+2)]
            skip_pos = positions[-(i+2)]
            skip_lbls = sampled_labels[-(i+2)]
            
            x, pos, labels = stage(
                decoder_features=x,
                decoder_pos=pos,
                encoder_features=skip_features,
                encoder_pos=skip_pos,
                encoder_labels=skip_lbls
            )
            x = x + skip_conn(skip_features)

        return self.final_mlp(x), labels

    def _validate_inputs(self, encoder_features, positions, sampled_labels):
        if not (len(encoder_features) == len(positions) == len(sampled_labels)):
            raise ValueError("Input lists must have same length")
        for i, (feat, pos, lbl) in enumerate(zip(encoder_features, positions, sampled_labels)):
            if feat.dim() != 2 or pos.dim() != 2 or pos.size(-1) != 3 or lbl.dim() != 1:
                raise ValueError(f"Inputs at stage {i} have invalid shape")
            if feat.size(0) != pos.size(0) or feat.size(0) != lbl.size(0):
                raise ValueError(f"Inputs at stage {i} have mismatched sizes")



class ASGFormer(nn.Module):
    def __init__(self, feature_dim, main_input_dim, main_output_dim, stages_config, knn_param, dropout_param=0.1):
        super(ASGFormer, self).__init__()
       
        if not isinstance(stages_config, list) or len(stages_config) < 2:
            raise ValueError("stages_config must be a list with at least 2 stages")
        if knn_param <= 0:
            raise ValueError("knn_param must be positive")

        self.x_mlp = nn.Sequential(
            nn.Linear(feature_dim, main_input_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_param),
            nn.Linear(main_input_dim, main_input_dim)
        )
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, main_input_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_param),
            nn.Linear(main_input_dim, main_input_dim)
        )
        self.encoder = Encoder(
            input_dim=main_input_dim,
            stages_config=stages_config,
            knn_param=knn_param,
            dropout_param=dropout_param
        )
        self.decoder = Decoder(
            main_output_dim=main_output_dim,
            stages_config=stages_config,
            knn_param=knn_param,
            dropout_param=dropout_param
        )
        self._initialize_weights()

    def forward(self, x, pos, labels):
        self._validate_inputs(x, pos, labels)
        
        x_emb = self.x_mlp(x)
        pos_emb = self.pos_mlp(pos)
        combined_features = x_emb + pos_emb
        
        encoder_features, positions, sampled_labels, _ = self.encoder(combined_features, pos, labels)
        logits, final_labels = self.decoder(encoder_features, positions, sampled_labels)
        return logits, final_labels

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _validate_inputs(self, x, pos, labels):
        if x.dim() != 2:
            raise ValueError(f"Features must be 2D tensor, got {x.dim()}D")
        if pos.dim() != 2 or pos.size(-1) != 3:
            raise ValueError("Positions must be shape [N, 3]")
        if labels.dim() != 1:
            raise ValueError("Labels must be 1D tensor")
        if x.size(0) != pos.size(0) or x.size(0) != labels.size(0):
            raise ValueError("Input sizes must match along dimension 0")
