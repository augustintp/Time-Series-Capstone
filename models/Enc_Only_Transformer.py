import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer, Transpose
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PositionalEmbedding
from torch.nn.modules.container import ModuleList


class Enc_Only_Transformer(nn.Module):
    """
    This is the direct forecasting method, where we use the encoder-only transformer to
    take in the context range, and then take the output embeddings from the encoder, flatten them,
    and finally pass them through a linear layer to get the forecasted values.

    In the vanilla transformer, each token represents the values at each timestep.
    At timestep t, the values are given_enc[:, t], x_enc[:, t], x_mark_enc[:, t], and meta_x.
    """

    def __init__(self, configs, device):
        super(Enc_Only_Transformer, self).__init__()
        self.is_train = True
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.window_size = self.seq_len + self.pred_len
        self.time_num = configs.time_num
        self.time_cat = configs.time_cat
        self.tgt_num = configs.tgt_num
        self.meta_num = configs.meta_num
        self.meta_cat = configs.meta_cat
        self.c_out = self.tgt_num
        self.d_model = configs.d_model
        self.use_norm = configs.use_norm
        self.output_attention = configs.output_attention

        freq_to_time_cov = {
            'daily': 2
        }
        self.time_cov_size = freq_to_time_cov[configs.freq]

        # Embedding
        self.time_embed_map = ModuleList([nn.Embedding(configs.time_num_class[i], configs.time_cat_embed[i])
                                          for i in range(self.time_cat)])
        self.meta_embed_map = ModuleList([nn.Embedding(configs.meta_num_class[i], configs.meta_cat_embed[i])
                                          for i in range(self.meta_cat)])
        self.input_proj = nn.Linear(self.time_num + sum(configs.time_cat_embed) + self.tgt_num
                                    + self.meta_num + sum(configs.meta_cat_embed) + self.time_cov_size, configs.d_model)
        self.position_embedding = PositionalEmbedding(configs.d_model)
        self.embed_dropout = nn.Dropout(configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=self.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        )
        # Decoder
        if self.task_name in ['forecast']:
            self.projection = nn.Linear(self.d_model * self.seq_len, configs.pred_len * self.tgt_num, bias=True)
        else:
            raise ValueError(f'Unknown task_name: {self.task_name}')

    def forecast(self, given_enc, x_enc, x_mark_enc, meta_x, output_attention=False):
        b_sz = given_enc.size(0)

        # window-wise normalization
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Embedding
        given_enc_cat_list = [self.time_embed_map[i](given_enc[:, :self.seq_len, -self.time_cat + i].long())
                              for i in range(self.time_cat)]
        given_enc_embed = torch.cat([given_enc[:, :self.seq_len, :-self.time_cat]] + given_enc_cat_list, dim=-1)
        meta_x_cat_list = [self.meta_embed_map[i](meta_x[:, -self.meta_cat + i].long())
                           for i in range(self.meta_cat)]
        meta_x_embed = torch.cat([meta_x[:, :-self.meta_cat]] + meta_x_cat_list, dim=-1)
        enc_in = torch.cat([given_enc_embed, x_enc, x_mark_enc[:, :self.seq_len],
                            meta_x_embed.unsqueeze(1).repeat(1, self.seq_len, 1)], dim=-1)

        # Encoder
        enc_in = self.input_proj(enc_in) + self.position_embedding(enc_in)
        enc_out, attns = self.encoder(enc_in, attn_mask=None)

        # Decoder
        dec_out = self.projection(enc_out.reshape(b_sz, -1)).reshape(b_sz, self.pred_len, self.tgt_num)

        if self.use_norm:
            dec_out = dec_out * stdev
            dec_out = dec_out + means

        if output_attention:
            return dec_out, attns
        else:
            return dec_out

    def set_train_mode(self):
        self.is_train = True

    def set_eval_mode(self):
        self.is_train = False

    def forward(self, given_enc, x_enc, x_mark_enc, meta_x=None, output_attention=False):
        """
        Shape:
        given_enc: (batch_size, window_size, time_num + time_cat)
        x_enc: (batch_size, seq_len, tgt_num)
        x_mark_enc: (batch_size, window_size, time_cov_size)
        meta_x: (batch_size, meta_num + meta_cat)
        """
        assert not (not self.output_attention and output_attention), \
            'model is not configured to output attention'

        if self.task_name == 'forecast':
            if output_attention:
                dec_out, attns = self.forecast(given_enc, x_enc, x_mark_enc, meta_x, output_attention)
                return dec_out[:, -self.pred_len:, :], attns
            else:
                dec_out = self.forecast(given_enc, x_enc, x_mark_enc, meta_x, output_attention)
                return dec_out[:, -self.pred_len:, :]
        else:
            raise ValueError(f'Unknown task_name: {self.task_name}')
