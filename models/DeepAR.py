import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList


class DeepAR(nn.Module):
    """
    Paper link: https://arxiv.org/abs/1704.04110
    """

    def __init__(self, configs, device):
        super(DeepAR, self).__init__()
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
        self.use_norm = configs.use_norm
        self.output_attention = configs.output_attention

        freq_to_time_cov = {
            'daily': 2
        }
        self.time_cov_size = freq_to_time_cov[configs.freq]

        # Embedding
        # Encoder
        # Decoder

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        raise NotImplementedError

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
