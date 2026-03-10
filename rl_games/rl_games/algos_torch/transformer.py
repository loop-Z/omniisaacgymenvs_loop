import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import ModuleList

def make_mlp_default(dim_list, dense_func = nn.Linear, norm_func_name = None, norm_only_first_layer=False, final_nonlinearity=True, nonlinearity=nn.ReLU):
    in_size = dim_list[0]
    layers = []
    need_norm = True
    for unit in dim_list[1:]:
        layers.append(dense_func(in_size, unit))
        layers.append(nonlinearity)

        if not need_norm:
            continue
        if norm_only_first_layer and norm_func_name is not None:
           need_norm = False 
        if norm_func_name == 'layer_norm':
            layers.append(torch.nn.LayerNorm(unit))
        elif norm_func_name == 'batch_norm':
            layers.append(torch.nn.BatchNorm1d(unit))
        in_size = unit
    return nn.Sequential(*layers)

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src

        for l in self.layers:
            output = l(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def get_attention_maps(self, src, mask=None, src_key_padding_mask=None):
        attention_maps = []
        output = src

        for l in self.layers:
            # NOTE: Shape of attention map: Batch Size x MAX_JOINTS x MAX_JOINTS
            # pytorch avgs the attention map over different heads; in case of
            # nheads > 1 code needs to change.
            output, attention_map = l(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                return_attention=True
            )
            attention_maps.append(attention_map)

        if self.norm is not None:
            output = self.norm(output)

        return output, attention_maps
    
class TransformerEncoderLayerResidual(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super(TransformerEncoderLayerResidual, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayerResidual, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, return_attention=False):
        src2 = self.norm1(src)
        src2, attn_weights = self.self_attn(
            src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        if return_attention:
            return src, attn_weights
        else:
            return src

class TransformerLayerResidual(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super(TransformerLayerResidual, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_k = nn.LayerNorm(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerLayerResidual, self).__setstate__(state)

    def forward(self, v, k, q, src_mask=None, src_key_padding_mask=None, return_attention=False):
        k = self.norm_k(k)
        q = self.norm_q(q)
        v = self.norm_v(v)
        out, attn_weights = self.self_attn(
            k, q, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        out = out + self.dropout1(v)

        out = self.norm2(out)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = v + self.dropout2(out)

        if return_attention:
            return out, attn_weights
        else:
            return out
        
class TransformerEncoderDecoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, decoder_layer, fuse_layer, num_layers, norm=None):
        super(TransformerEncoderDecoder, self).__init__()
        self.encode_layers = _get_clones(encoder_layer, num_layers)
        self.decode_layer = _get_clones(decoder_layer, 2)
        self.fuse_layer = fuse_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, encode, decode, mask=None, src_key_padding_mask=None):
        input = encode
        # Encode
        for l in self.encode_layers:
            input = l(input, input, input, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        # Decode
        output = torch.cat([encode, decode], dim=-1)
        output = self.fuse_layer(output)
        output = self.decode_layer[0](output, input, input, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decode_layer[1](output, output, output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def get_attention_maps(self, src, mask=None, src_key_padding_mask=None):
        attention_maps = []
        output = src

        for l in self.encode_layers + [self.decode_layer]:
            # NOTE: Shape of attention map: Batch Size x MAX_JOINTS x MAX_JOINTS
            # pytorch avgs the attention map over different heads; in case of
            # nheads > 1 code needs to change.
            output, attention_map = l(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                return_attention=True
            )
            attention_maps.append(attention_map)

        if self.norm is not None:
            output = self.norm(output)

        return output, attention_maps

class TransformerModel(nn.Module):
    def __init__(self, cfg, obs_space, act_fn):
        super(TransformerModel, self).__init__()
        self.cfg = cfg
        self.seq_len = obs_space["transforms"][0]
        self.state_input = obs_space["state"][0]
        self.transforms_input = obs_space["transforms"][-1]
        self.d_model = self.cfg["d_model"]

        if self.cfg["pos_embedding"] == "learnt":
            seq_len = self.seq_len
            self.pos_embedding = PositionalEncoding(self.d_model, seq_len)
        elif self.cfg["pos_embedding"] == "abs":
            self.pos_embedding = PositionalEncoding1D(self.d_model, self.seq_len)

        self.state_embedding = MLPEncoder(
            self.state_input,
            act_fn,
            self.cfg["state_mlp_dim"]
        )
        self.transforms_embedding = MLPEncoder(
            self.transforms_input,
            act_fn,
            self.cfg["transforms_mlp_dim"]
        )
        encoder_layer = TransformerLayerResidual(
            self.d_model,
            self.cfg["num_head"],
            self.cfg["dim_feedforward"],
            self.cfg["dropout"],
        )
        decoder_layer = TransformerLayerResidual(
            self.d_model,
            self.cfg["num_head"],
            self.cfg["dim_feedforward"],
            self.cfg["dropout"],
        )
        self.fuse_layer = MLPEncoder(
            self.d_model + self.cfg["state_mlp_dim"][-1],
            act_fn,
            self.cfg["fuse_mlp_dim"]
        )
        self.transformer = TransformerEncoderDecoder(
            encoder_layer,
            decoder_layer,
            self.fuse_layer,
            self.cfg["num_layer"],
            norm=None,
        )
         
        self.decoder = make_mlp_default(
            [self.d_model] + list(self.cfg["decoder_mlp_dim"]),
            final_nonlinearity=True,
            nonlinearity=act_fn,
        )

    def init_weights(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.Identity(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)    
        for m in self.state_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.Identity(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)    
        for m in self.transforms_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.Identity(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)    
        for m in self.fuse_layer.modules():
            if isinstance(m, nn.Linear):
                nn.Identity(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs, obs_mask, return_attention=False):
        transforms_obs = obs["transforms"].permute(1,0,2)
        _, batch_size, _ = transforms_obs.shape
        state_obs = obs["state"]
        state_obs = state_obs.repeat(self.seq_len, 1)
        state_obs = state_obs.reshape(self.seq_len, batch_size, -1)

        state_e = self.state_embedding(state_obs)        
        transforms_e = self.transforms_embedding(transforms_obs)
        
        attention_maps = None
        if self.cfg["pos_embedding"] in ["learnt", "abs"]:
            transforms_ep = self.pos_embedding(transforms_e)
        if return_attention:
            obs_embed_t = self.transformer(
                transforms_ep,
                state_e,
                src_key_padding_mask=obs["masks"].bool()
            )
        else:
            obs_embed_t = self.transformer(
                transforms_ep,
                state_e,
                src_key_padding_mask=obs["masks"].bool()
            )

        output = self.decoder(obs_embed_t)
        output = output.permute(1, 0, 2)
        return output, attention_maps, batch_size


class MetamorphModel(nn.Module):
    def __init__(self, cfg, obs_space, act_fn):
        super(MetamorphModel, self).__init__()
        self.cfg = cfg
        self.seq_len = obs_space["transforms"][0]
        self.state_input = obs_space["state"][0]
        self.transforms_input = obs_space["transforms"][-1]
        self.d_model = self.cfg["state_embed_dim"][-1]

        if self.cfg["pos_embedding"] == "learnt":
            seq_len = self.seq_len
            self.pos_embedding = PositionalEncoding(self.d_model, seq_len)
        elif self.cfg["pos_embedding"] == "abs":
            self.pos_embedding = PositionalEncoding1D(self.d_model, self.seq_len)

        self.embed = MLPEncoder(
            obs_space["state"][0] + obs_space["transforms"][-1],
            act_fn,
            self.cfg)
        encoder_layers = TransformerEncoderLayerResidual(
            self.d_model,
            self.cfg["num_head"],
            self.cfg["dim_feedforward"],
            self.cfg["dropout"],
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, self.cfg["num_layer"], norm=None,
        )

        self.decoder = make_mlp_default(
            [self.d_model] + list(self.cfg["decoder_mlp_dim"]),
            final_nonlinearity=True,
            nonlinearity=act_fn,
        )
        self.init_weights()

    def init_weights(self):
        for m in self.embed.modules():
            if isinstance(m, nn.Linear):
                nn.Identity(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)    
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.Identity(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)    

    def forward(self, obs, obs_mask, return_attention=False):
        transforms_obs = obs["transforms"].permute(1,0,2)
        _, batch_size, _ = transforms_obs.shape
        state_obs = obs["state"]
        state_obs = state_obs.repeat(self.seq_len, 1)
        state_obs = state_obs.reshape(self.seq_len, batch_size, -1)
        obs_t = torch.cat([transforms_obs, state_obs], axis=-1)
        obs_embed = self.embed(obs_t) * math.sqrt(self.d_model)

        attention_maps = None
        if self.cfg["pos_embedding"] in ["learnt", "abs"]:
            obs_embed = self.pos_embedding(obs_embed)
        if return_attention:
            obs_embed_t, attention_maps = self.transformer_encoder.get_attention_maps(
                obs_embed, src_key_padding_mask=obs["masks"].bool()
            )
        else:
            obs_embed_t = self.transformer_encoder(
                obs_embed, src_key_padding_mask=obs["masks"].bool()
            )

        output = self.decoder(obs_embed_t)
        output = output.permute(1, 0, 2)
        return output, attention_maps, batch_size

class StatePositionalEncoder(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """
    def __init__(self, d_input: int, n_freqs: int, log_space: bool = False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
    
    def forward(self, x) -> torch.Tensor:
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, act, dim_list, norm_func_name="layer_norm"):
        super(MLPEncoder, self).__init__()
        layers = []
        in_size = input_dim
        for unit in dim_list[1:]:
            layers.append(nn.Linear(in_size, unit))
            layers.append(act)
            if norm_func_name == 'layer_norm':
                layers.append(nn.LayerNorm(unit))
            elif norm_func_name == 'batch_norm':
                layers.append(nn.BatchNorm1d(unit))
            in_size = unit
        self.encode = nn.Sequential(*layers)
    
    def forward(self, input):
        return self.encode(input)