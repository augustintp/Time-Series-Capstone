import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer, Transpose
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PositionalEmbedding
from torch.nn.modules.container import ModuleList


class Enc_Dec_Transformer(nn.Module):
    """
    Example: An encoder-decoder Transformer for time-series forecasting,
    inspired by your Enc_Only_Transformer code. Each time step is treated as a token.
    """

    def __init__(self, configs, device):
        super(Enc_Dec_Transformer, self).__init__()
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

        self.device = device

        # 1) Embeddings (similar to Enc_Only_Transformer)
        self.time_embed_map = ModuleList([
            nn.Embedding(configs.time_num_class[i], configs.time_cat_embed[i])
            for i in range(self.time_cat)
        ])
        self.meta_embed_map = ModuleList([
            nn.Embedding(configs.meta_num_class[i], configs.meta_cat_embed[i])
            for i in range(self.meta_cat)
        ])

        numeric_dim = self.time_num + self.tgt_num + self.meta_num + self.time_cov_size
        cat_dim = sum(configs.time_cat_embed) + sum(configs.meta_cat_embed)
        input_dim = numeric_dim + cat_dim

        self.input_proj = nn.Linear(input_dim, configs.d_model)
        self.position_embedding = PositionalEmbedding(configs.d_model)
        self.embed_dropout = nn.Dropout(configs.dropout)

        # 2) Transformer Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,  # full attention for the encoder
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=self.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(configs.d_model),
                Transpose(1, 2)
            )
        )

        # 2) Transformer Decoder
        d_layers = getattr(configs, 'd_layers', configs.e_layers)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,  # typically masked attention in the decoder
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=self.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,  # cross-attention over encoder output
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=self.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(d_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(configs.d_model),
                Transpose(1, 2)
            )
        )

        # 3) Projection layer
        self.projection = nn.Linear(self.d_model, self.tgt_num)

    def set_train_mode(self):
        self.is_train = True

    def set_eval_mode(self):
        self.is_train = False

    def _embed_input(
        self,
        x_time_cat,   # shape: (B, T, time_cat)
        x_time_num,   # shape: (B, T, time_num)
        x_tgt,        # shape: (B, T, tgt_num)
        x_cov,        # shape: (B, T, time_cov_size)
        x_meta=None   # shape: (B, meta_num + meta_cat)
    ):
        B, T = x_time_num.shape[0], x_time_num.shape[1]

        # Time-cat embeddings
        cat_emb_list = []
        for i in range(self.time_cat):
            emb = self.time_embed_map[i](x_time_cat[:, :, i].long())  # (B, T, embed_dim_i)
            cat_emb_list.append(emb)

        # Meta-cat embeddings
        meta_cat_list = []
        if (x_meta is not None) and (self.meta_cat > 0):
            for i in range(self.meta_cat):
                meta_emb = self.meta_embed_map[i](x_meta[:, -self.meta_cat + i].long())
                meta_emb = meta_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, embed_dim)
                meta_cat_list.append(meta_emb)

        cat_emb = torch.cat(cat_emb_list + meta_cat_list, dim=-1) if cat_emb_list or meta_cat_list else None

        # Numeric part
        if (x_meta is not None) and (self.meta_num > 0):
            meta_num_part = x_meta[:, :self.meta_num]  # shape (B, meta_num)
            meta_num_part = meta_num_part.unsqueeze(1).expand(-1, T, -1)
            numeric_part = torch.cat([x_time_num, x_tgt, x_cov, meta_num_part], dim=-1)
        else:
            numeric_part = torch.cat([x_time_num, x_tgt, x_cov], dim=-1)

        # Merge numeric + cat
        if cat_emb is not None:
            total_in = torch.cat([numeric_part, cat_emb], dim=-1)
        else:
            total_in = numeric_part

        proj_out = self.input_proj(total_in)
        return proj_out

    def forecast(self, given_enc, x_enc, x_mark_enc, meta_x, output_attention=False):
        B = given_enc.size(0)

        # 1) Optional normalization
        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # 2) Encoder input
        #    given_enc[:, :seq_len] => context portion
        given_enc_context = given_enc[:, :self.seq_len, :]
        x_cov_context = x_mark_enc[:, :self.seq_len, :]

        x_time_num_context = given_enc_context[:, :, :self.time_num]
        x_time_cat_context = given_enc_context[:, :, self.time_num:self.time_num + self.time_cat]

        enc_in = self._embed_input(
            x_time_cat_context,
            x_time_num_context,
            x_enc,
            x_cov_context,
            meta_x
        )
        enc_in = enc_in + self.position_embedding(enc_in)
        enc_in = self.embed_dropout(enc_in)

        enc_out, enc_attns = self.encoder(enc_in, attn_mask=None)

        # 3) Decoder input
        #    zero-pad future target for forecast range
        given_enc_forecast = given_enc[:, self.seq_len:self.seq_len + self.pred_len, :]
        x_cov_forecast = x_mark_enc[:, self.seq_len:self.seq_len + self.pred_len, :]

        x_time_num_forecast = given_enc_forecast[:, :, :self.time_num]
        x_time_cat_forecast = given_enc_forecast[:, :, self.time_num:self.time_num + self.time_cat]

        x_future_tgt = torch.zeros(B, self.pred_len, self.tgt_num, device=enc_in.device)

        dec_in = self._embed_input(
            x_time_cat_forecast,
            x_time_num_forecast,
            x_future_tgt,
            x_cov_forecast,
            meta_x
        )
        dec_in = dec_in + self.position_embedding(dec_in)
        dec_in = self.embed_dropout(dec_in)

        # 4) Decoder forward
        #    Previously: dec_out, dec_attns = self.decoder(...)
        #    which caused the error "too many values to unpack"
        dec_out = self.decoder(dec_in, enc_out, x_mask=None, cross_mask=None)
        dec_attns = None  # or return no attention at all

        # 5) Projection -> forecast
        forecast_tokens = self.projection(dec_out)  # (B, pred_len, tgt_num)

        # Undo normalization if used
        if self.use_norm:
            forecast_tokens = forecast_tokens * stdev
            forecast_tokens = forecast_tokens + means

        # Return attention if requested, else just forecast
        if output_attention:
            return forecast_tokens, dec_attns
        else:
            return forecast_tokens

    def forward(self, given_enc, x_enc, x_mark_enc, meta_x=None, output_attention=False):
        """
        given_enc: (batch_size, window_size, time_num + time_cat)
        x_enc:     (batch_size, seq_len, tgt_num)
        x_mark_enc: (batch_size, window_size, time_cov_size)
        meta_x:    (batch_size, meta_num + meta_cat)
        """
        assert not (not self.output_attention and output_attention), \
            "Model is not configured to return attention if output_attention=False."

        if self.task_name == 'forecast':
            if output_attention:
                dec_out, attns = self.forecast(given_enc, x_enc, x_mark_enc, meta_x, output_attention=True)
                return dec_out, attns
            else:
                dec_out = self.forecast(given_enc, x_enc, x_mark_enc, meta_x, output_attention=False)
                return dec_out
        else:
            raise ValueError(f'Unknown task_name: {self.task_name}')
