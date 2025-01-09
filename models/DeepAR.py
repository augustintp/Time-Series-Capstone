import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList


class DeepAR(nn.Module):
    """
    Paper link: https://arxiv.org/abs/1704.04110
    
    Teacher-Forcing Approach:
      - The encoder consumes the past target (x_enc).
      - The decoder consumes the future target (x_dec) as input, 
        rather than the model's own predictions (classic teacher forcing).
    """

    def __init__(self, configs, device):
        super(DeepAR, self).__init__()
        self.is_train = True
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.window_size = self.seq_len + self.pred_len
        self.time_num = configs.time_num       # number of continuous time features (per step)
        self.time_cat = configs.time_cat       # number of categorical time features (per step)
        self.tgt_num = configs.tgt_num         # dimension of the target
        self.meta_num = configs.meta_num       # number of continuous meta features (global)
        self.meta_cat = configs.meta_cat       # number of categorical meta features (global)
        self.c_out = self.tgt_num
        self.use_norm = configs.use_norm
        self.output_attention = configs.output_attention

        # freq => map to time_cov_size (e.g. daily => 2)
        freq_to_time_cov = {'daily': 2}
        self.time_cov_size = freq_to_time_cov[configs.freq]

        # -----------------------------------------------------
        # Embedding
        # -----------------------------------------------------
        #
        # We'll create embeddings for:
        #   1) time_cat columns,
        #   2) meta_cat columns,
        #   3) optional linear layers for time_num or meta_num.
        #
        # The final LSTM input dimension is calculated from these.

        # time-cat embeddings (each col has its own embedding)
        if self.time_cat > 0:
            self.time_cat_embeddings = nn.ModuleList([
                nn.Embedding(
                    num_embeddings=configs.time_num_class[i],
                    embedding_dim=configs.time_cat_embed[i]
                )
                for i in range(self.time_cat)
            ])
            self.time_cat_embed_dim = sum(configs.time_cat_embed)
        else:
            self.time_cat_embeddings = None
            self.time_cat_embed_dim = 0

        # meta-cat embeddings
        if self.meta_cat > 0:
            self.meta_cat_embeddings = nn.ModuleList([
                nn.Embedding(
                    num_embeddings=configs.meta_num_class[i],
                    embedding_dim=configs.meta_cat_embed[i]
                )
                for i in range(self.meta_cat)
            ])
            self.meta_cat_embed_dim = sum(configs.meta_cat_embed)
        else:
            self.meta_cat_embeddings = None
            self.meta_cat_embed_dim = 0

        # optional linear for time_num (continuous)
        self.time_num_linear = nn.Linear(self.time_num, self.time_num) if self.time_num > 0 else None
        # optional linear for meta_num (continuous)
        self.meta_num_linear = nn.Linear(self.meta_num, self.meta_num) if self.meta_num > 0 else None

        # -----------------------------------------------------
        # Encoder
        # -----------------------------------------------------
        #
        # We'll define an LSTM that processes the "past" (x_enc).
        # The input dimension is:
        #   time_num + time_cat_embed_dim + time_cov_size + tgt_num
        # + meta_num + meta_cat_embed_dim
        #
        # The hidden dimension is configs.d_model, with num_layers = configs.e_layers.

        self.enc_input_dim = (self.time_num +
                              self.time_cat_embed_dim +
                              self.time_cov_size +
                              self.tgt_num +
                              self.meta_num +
                              self.meta_cat_embed_dim)

        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers

        self.encoder = nn.LSTM(
            input_size=self.enc_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # -----------------------------------------------------
        # Decoder
        # -----------------------------------------------------
        #
        # Another LSTM for the "future" (x_dec), also with the same input dimension.
        # We'll feed the real future target (teacher forcing) each step,
        # along with time/meta features for the future window.

        self.decoder = nn.LSTM(
            input_size=self.enc_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # final projection from hidden -> target dimension
        self.projection = nn.Linear(self.hidden_size, self.tgt_num)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Teacher forcing forecast:
          x_enc: (B, seq_len, tgt_num)   past target
          x_mark_enc: (B, seq_len, time_cov_size)  past time covariates
          x_dec: (B, pred_len, tgt_num)  future target (ground truth => teacher forcing)
          x_mark_dec:(B, pred_len, time_cov_size)  future time covariates

        Return shape: (B, seq_len+pred_len, tgt_num).
                      The last pred_len are the actual forecast.
        """
        B = x_enc.shape[0]

        # optional normalization: we only do it on the "past" portion for stability
        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True)
            x_enc_ = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc_, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc_ = x_enc_ / stdev
        else:
            x_enc_ = x_enc

        # 1) Encode the past
        enc_input = self.build_lstm_input(
            x_enc_, x_mark_enc,  # past target + cov
            meta_x=None  # we handle meta below
        )
        # incorporate meta features (continuous + cat) for each time step
        # (B, seq_len, enc_input_dim)
        enc_input = self.append_meta(enc_input, 0, self.seq_len)  # 0..seq_len-1

        enc_out, (h_enc, c_enc) = self.encoder(enc_input)  # => (B, seq_len, hidden_size)
        past_preds = self.projection(enc_out)              # => (B, seq_len, tgt_num)

        if self.use_norm:
            past_preds = past_preds * stdev + means

        # 2) Decode the future (teacher forcing)
        #
        # We feed x_dec as the "target" portion for each future step, 
        # along with time covariates x_mark_dec, so the model sees real future.
        if self.use_norm:
            # we should also normalize x_dec with the same mean, stdev
            x_dec_ = (x_dec - means) / stdev
        else:
            x_dec_ = x_dec

        dec_input = self.build_lstm_input(
            x_dec_, x_mark_dec,
            meta_x=None  # meta appended next
        )
        # (B, pred_len, enc_input_dim)
        dec_input = self.append_meta(dec_input, self.seq_len, self.pred_len)  # seq_len..seq_len+pred_len-1

        dec_out, (h_dec, c_dec) = self.decoder(dec_input, (h_enc, c_enc))
        future_preds = self.projection(dec_out)  # => (B, pred_len, tgt_num)

        if self.use_norm:
            future_preds = future_preds * stdev + means

        # combine => (B, seq_len+pred_len, tgt_num)
        all_preds = torch.cat([past_preds, future_preds], dim=1)
        return all_preds

    def set_train_mode(self):
        self.is_train = True

    def set_eval_mode(self):
        self.is_train = False

    def forward(self, given_enc, x_enc, x_mark_enc, meta_x=None, output_attention=False):
        """
        Shape (by your definition):
          given_enc:   (B, window_size, time_num + time_cat) [NOT directly used here, unless you incorporate it yourself]
          x_enc:       (B, seq_len, tgt_num)   past target
          x_mark_enc:  (B, window_size, time_cov_size) => we can slice out the first seq_len
          meta_x:      (B, meta_num + meta_cat) => we store for appending
          output_attention: not used (LSTM has no attention by default)

        If you strictly want to pass x_dec, x_mark_dec from the outside, 
        you can add them to the forward signature. 
        For now, we'll just assume you slice them yourself 
        before calling self.forecast(...).
        """
        assert not (not self.output_attention and output_attention), \
            'model is not configured to output attention'

        if self.task_name == 'forecast':
            # For teacher forcing, we need x_dec and x_mark_dec
            # Possibly your code slices them from x_enc, x_mark_enc
            # e.g. x_enc[:,:self.seq_len], x_enc[:,self.seq_len:]
            # or you handle it differently in run.py
            raise NotImplementedError(
                "Please slice x_dec, x_mark_dec from your data loader, "
                "then call self.forecast(x_enc_past, x_mark_enc_past, x_dec_future, x_mark_dec_future)."
            )
        else:
            raise ValueError(f'Unknown task_name: {self.task_name}')

    # ------------------------------------------------------------------------
    # Helper: build the partial input for LSTM (time embeddings + target)
    # ------------------------------------------------------------------------
    def build_lstm_input(self, x_tgt, x_cov, meta_x=None):
        """
        x_tgt: (B, T, tgt_num)
        x_cov: (B, T, time_cov_size)
        meta_x: optional (B, meta_num + meta_cat) if you want to handle it here.
        Return partial: (B, T, ???) => no meta appended yet
        """
        B, T, _ = x_tgt.shape
        feats = []

        # (a) time_cat => from ??? 
        #   We skip it here because you might pass them in x_cov or a separate array. 
        #   If you have time_cat in a separate array, you can embed them here.

        # (b) time_num => also skip if you have them in x_cov or x_tgt. 
        #   If you have them, you'd do self.time_num_linear(...) etc.

        # (c) time cov (month/day)
        feats.append(x_cov)  # => shape (B,T,time_cov_size)

        # (d) target
        feats.append(x_tgt)

        # Note: you could do more merges if you store time_cat/time_num in some array. 
        return torch.cat(feats, dim=-1)  # => (B,T, time_cov_size + tgt_num)

    # ------------------------------------------------------------------------
    # Helper: append meta features (continuous + categorical) across timesteps
    # ------------------------------------------------------------------------
    def append_meta(self, partial_input, t_start, length):
        """
        partial_input: (B, length, partial_dim)
        meta_x is not yet included. We'll read self.meta_num_linear, self.meta_cat_embeddings, etc.
        Then we broadcast them across 'length' timesteps from t_start..t_start+length-1.
        """
        B, T, in_dim = partial_input.shape
        # Expect T == length
        if self.meta_num > 0 and self.meta_num_linear is not None:
            # (B, meta_num) => pass linear => (B, meta_num) => expand => (B, length, meta_num)
            # you'd do something like:
            # meta_cont = ...
            pass

        if self.meta_cat > 0 and self.meta_cat_embeddings is not None:
            # similarly embed cat columns => broadcast
            pass

        # for the sake of minimal teacher forcing example, we skip the actual meta usage
        # in real code, you'd store self.meta_x globally or pass as param, embed, then expand

        return partial_input  # as is, if not using meta

