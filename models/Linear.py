import torch
import torch.nn as nn
from torch.nn.modules.container import ModuleList


class Linear(nn.Module):
    """
    We flatten out all the time-dependent features and concatenate them with
    the meta features. We then pass this through a linear layer to get the
    forecasted values.
    given_enc: (batch_size, window_size, time_num + time_cat) --> flatten -->
    (batch_size, (time_num + sum(time_cat_embed)) * window_size)
    x_enc: (batch_size, seq_len, tgt_num) --> flatten --> (batch_size, seq_len * tgt_num)
    x_mark_enc: (batch_size, window_size, time_cov_size) --> flatten -->
    (batch_size, window_size * time_cov_size)
    Concatenate the above three with meta_x to get
    (batch_size, (time_num + sum(time_cat_embed) + time_cov_size) * window_size + seq_len * tgt_num + meta_num + sum(meta_cat_embed))
    """

    def __init__(self, configs, device):
        super(Linear, self).__init__()
        self.is_train = True
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.window_size = self.seq_len + self.pred_len
        self.time_cov_size = configs.time_cov_size
        self.time_num = configs.time_num
        self.time_cat = configs.time_cat
        self.tgt_num = configs.tgt_num
        self.meta_num = configs.meta_num
        self.meta_cat = configs.meta_cat
        self.c_out = self.tgt_num
        self.use_norm = configs.use_norm
        self.output_attention = configs.output_attention

        # Embedding
        self.time_embed_map = ModuleList([nn.Embedding(configs.time_num_class[i], configs.time_cat_embed[i])
                                          for i in range(self.time_cat)])
        self.meta_embed_map = ModuleList([nn.Embedding(configs.meta_num_class[i], configs.meta_cat_embed[i])
                                          for i in range(self.meta_cat)])

        # Decoder
        input_embed_size = ((self.time_num + sum(configs.time_cat_embed) + self.time_cov_size) * self.window_size +
                            self.seq_len * self.tgt_num + self.meta_num + sum(configs.meta_cat_embed))
        if self.task_name in ['forecast']:
            self.projection = nn.Linear(input_embed_size, configs.pred_len * self.tgt_num, bias=True)
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
        given_enc_cat_list = [self.time_embed_map[i](given_enc[:, :, -self.time_cat + i].long())
                              for i in range(self.time_cat)]
        given_enc_embed = torch.cat([given_enc[:, :, :-self.time_cat]] + given_enc_cat_list, dim=-1)
        meta_x_cat_list = [self.meta_embed_map[i](meta_x[:, -self.meta_cat + i].long())
                           for i in range(self.meta_cat)]
        meta_x_embed = torch.cat([meta_x[:, :-self.meta_cat]] + meta_x_cat_list, dim=-1)
        enc_in = torch.cat([given_enc_embed.reshape(b_sz, -1), x_enc.reshape(b_sz, -1),
                            x_mark_enc.reshape(b_sz, -1), meta_x_embed], dim=-1)

        dec_out = self.projection(enc_in).reshape(b_sz, self.pred_len, self.tgt_num)

        if self.use_norm:
            dec_out = dec_out * stdev
            dec_out = dec_out + means

        if output_attention:
            # TODO: support attention output in linear model
            return dec_out, None
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
