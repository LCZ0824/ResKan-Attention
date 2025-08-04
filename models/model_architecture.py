import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SuperKANLinear(nn.Module):
    def __init__(self, in_features, out_features, num_grids=8):
        super(SuperKANLinear, self).__init__()
        self.in_features, self.out_features, self.num_grids = in_features, out_features, num_grids
        self.base_linear = nn.Linear(in_features, out_features)
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, num_grids))
        self.register_buffer('grid', torch.linspace(-2, 2, num_grids))
        self.adaptive_gate = nn.Parameter(torch.ones(out_features))
        self.feature_importance = nn.Parameter(torch.ones(in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_linear.weight, a=math.sqrt(5))
        if self.base_linear.bias is not None: 
            nn.init.zeros_(self.base_linear.bias)
        nn.init.normal_(self.spline_weight, 0, 0.1)

    def forward(self, x):
        x_weighted = x * torch.sigmoid(self.feature_importance).unsqueeze(0)
        base_output = self.base_linear(x_weighted)
        x_norm = torch.tanh(x_weighted)
        x_expanded = x_norm.unsqueeze(-1)
        grid_expanded = self.grid.view(1, 1, -1)
        basis = torch.exp(-((x_expanded - grid_expanded) ** 2) * 2.0)
        basis = F.normalize(basis, p=1, dim=-1)
        spline_output = torch.einsum('big,oig->bo', basis, self.spline_weight)
        gate = torch.sigmoid(self.adaptive_gate).unsqueeze(0)
        return gate * spline_output + (1 - gate) * base_output

class UltraAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, use_layer_scale=False):
        super(UltraAttentionBlock, self).__init__()
        self.use_layer_scale = use_layer_scale
        self.self_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(nn.Linear(dim, dim * 3), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 3, dim))
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        if use_layer_scale:
            self.scale_attn = nn.Parameter(torch.ones(1) * 1e-2)
            self.scale_ff = nn.Parameter(torch.ones(1) * 1e-2)

    def forward(self, x):
        if len(x.shape) == 2: 
            x = x.unsqueeze(1)
        attn_out, _ = self.self_attention(x, x, x)
        if self.use_layer_scale: 
            attn_out = attn_out * self.scale_attn
        x = self.norm1(x + attn_out)
        ff_out = self.feed_forward(x)
        gate_weight = self.gate(x)
        if self.use_layer_scale: 
            ff_out = ff_out * self.scale_ff
        x = self.norm2(x + gate_weight * ff_out)
        return x.squeeze(1)

class ResidualKANBlock(nn.Module):
    def __init__(self, dim, num_grids=8, dropout=0.1, use_layer_scale=False):
        super(ResidualKANBlock, self).__init__()
        self.use_layer_scale = use_layer_scale
        self.kan_layer = SuperKANLinear(dim, dim, num_grids)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        if use_layer_scale: 
            self.scale_res = nn.Parameter(torch.ones(1) * 1e-2)

    def forward(self, x):
        residual = x
        out = self.dropout(self.kan_layer(x))
        gate = self.residual_gate(residual)
        if self.use_layer_scale: 
            out = out * self.scale_res
        return self.norm(residual + gate * out)

class CrossFusionModule(nn.Module):
    def __init__(self, kan_dim, traditional_dim, fusion_dim):
        super(CrossFusionModule, self).__init__()
        self.kan_proj = nn.Linear(kan_dim, fusion_dim)
        self.traditional_proj = nn.Linear(traditional_dim, fusion_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=4, batch_first=True)
        self.fusion_gate = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.Sigmoid())

    def forward(self, kan_features, traditional_features):
        kan_proj = self.kan_proj(kan_features).unsqueeze(1)
        trad_proj = self.traditional_proj(traditional_features).unsqueeze(1)
        cross_attn_out, _ = self.cross_attention(kan_proj, trad_proj, trad_proj)
        cross_attn_out = cross_attn_out.squeeze(1)
        concat_features = torch.cat([kan_proj.squeeze(1), cross_attn_out], dim=-1)
        fusion_weight = self.fusion_gate(concat_features)
        return fusion_weight * kan_proj.squeeze(1) + (1 - fusion_weight) * cross_attn_out

class ResKANUltraAttention(nn.Module):
    def __init__(self, input_features, config):
        super(ResKANUltraAttention, self).__init__()
        base_dim = config['base_dim']
        num_grids = config['num_grids']
        dropout = config['dropout']
        residual_depth = config['residual_depth']
        attention_layers = config['attention_layers']
        cross_fusion_layers = config['cross_fusion_layers']
        use_layer_scale = config.get('use_layer_scale', False)
        
        self.kan_input = SuperKANLinear(input_features, base_dim, num_grids)
        self.kan_residual_blocks = nn.ModuleList([
            ResidualKANBlock(base_dim, num_grids, dropout, use_layer_scale) 
            for _ in range(residual_depth)
        ])
        self.kan_attention_layers = nn.ModuleList([
            UltraAttentionBlock(base_dim, num_heads=8, dropout=dropout, use_layer_scale=use_layer_scale) 
            for _ in range(attention_layers)
        ])
        
        self.traditional_path = nn.Sequential(
            nn.Linear(input_features, base_dim * 2), nn.BatchNorm1d(base_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(base_dim * 2, base_dim), nn.BatchNorm1d(base_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.traditional_attention_layers = nn.ModuleList([
            UltraAttentionBlock(base_dim, num_heads=4, dropout=dropout, use_layer_scale=use_layer_scale) 
            for _ in range(attention_layers)
        ])
        
        self.cross_fusion_layers = nn.ModuleList([
            CrossFusionModule(kan_dim=base_dim, traditional_dim=base_dim, fusion_dim=base_dim) 
            for _ in range(cross_fusion_layers)
        ])
        
        self.final_classifier = nn.Sequential(
            SuperKANLinear(base_dim, base_dim // 2, num_grids // 2),
            nn.LayerNorm(base_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(base_dim // 2, 32), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout * 0.3),
            nn.Linear(16, 1), nn.Sigmoid()
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None: 
                nn.init.constant_(module.bias, 0.01)

    def forward(self, x):
        kan_out = F.gelu(self.kan_input(x))
        for block in self.kan_residual_blocks: 
            kan_out = block(kan_out)
        for block in self.kan_attention_layers: 
            kan_out = block(kan_out)
        
        traditional_out = self.traditional_path(x)
        for block in self.traditional_attention_layers: 
            traditional_out = block(traditional_out)
        
        fused_features = kan_out
        for block in self.cross_fusion_layers: 
            fused_features = block(fused_features, traditional_out)
        
        return self.final_classifier(fused_features)

def get_default_config():
    return {
        'base_dim': 128,
        'num_grids': 8,
        'dropout': 0.2,
        'lr': 0.0005,
        'weight_decay': 0.01,
        'batch_size': 32,
        'max_epochs': 200,
        'patience': 30,
        'residual_depth': 3,
        'attention_layers': 1,
        'cross_fusion_layers': 1,
        'use_layer_scale': False,
    }