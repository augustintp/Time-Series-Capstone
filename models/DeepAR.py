import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList

class DeepAR(nn.Module):
    """
    Minimal DeepAR-like model that handles multiple categorical columns for time and meta features.
    """

    def __init__(self, configs, device):
        super(DeepAR, self).__init__()
        self.is_train = True
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.window_size = self.seq_len + self.pred_len
        self.time_num = configs.time_num
        self.time_cat = configs.time_cat            # number of *categorical time* columns
        self.tgt_num = configs.tgt_num
        self.meta_num = configs.meta_num
        self.meta_cat = configs.meta_cat            # number of *categorical meta* columns
        self.c_out = self.tgt_num
        self.use_norm = configs.use_norm
        self.output_attention = configs.output_attention
        self.device = device
       
        freq_to_time_cov = {
            'daily': 2  # e.g., "month" + "day"
        }
        self.time_cov_size = freq_to_time_cov[configs.freq]

        # ----------------------------------------------
        # Embeddings for time_cat columns
        # Suppose your code or data indicates that each time_cat column
        # might have different cardinalities. We'll do a simple approach:
        # a single "embedding_dim" for all, adjust as needed.
        # ----------------------------------------------
        cat_embed_dim = 8  # example dimension
        if self.time_cat > 0:
            # E.g. if time_cat = 3, we have 3 columns for time categoricals
            self.time_cat_embeddings = nn.ModuleList([
                nn.Embedding(num_embeddings=50, embedding_dim=cat_embed_dim)  # placeholder
                for _ in range(self.time_cat)
            ])
        else:
            self.time_cat_embeddings = None
       
        # ----------------------------------------------
        # Embeddings for meta_cat columns
        # E.g. if meta_cat = 3 => 3 separate columns
        # each column gets its own embedding
        # ----------------------------------------------
        if self.meta_cat > 0:
            self.meta_cat_embeddings = nn.ModuleList([
                nn.Embedding(num_embeddings=50, embedding_dim=cat_embed_dim)  # placeholder
                for _ in range(self.meta_cat)
            ])
        else:
            self.meta_cat_embeddings = None
       
        # ----------------------------------------------
        # LSTM hidden size
        # ----------------------------------------------
        self.hidden_size = configs.d_model

        # ----------------------------------------------
        # Figure out input dimension to the LSTM
        # 1) time_num (continuous time features)
        # 2) sum of time_cat embeddings => time_cat * cat_embed_dim
        # 3) time_cov_size
        # 4) tgt_num (since we feed the target)
        # 5) meta_num (continuous meta features)
        # 6) sum of meta_cat embeddings => meta_cat * cat_embed_dim
        # ----------------------------------------------
        self.input_dim = 0
        self.input_dim += self.time_num
        if self.time_cat_embeddings is not None:
            self.input_dim += self.time_cat * cat_embed_dim
        self.input_dim += self.time_cov_size
        self.input_dim += self.tgt_num
        self.input_dim += self.meta_num
        if self.meta_cat_embeddings is not None:
            self.input_dim += self.meta_cat * cat_embed_dim

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )

        # final projection (hidden -> target)
        self.projection = nn.Linear(self.hidden_size, self.tgt_num)

    def set_train_mode(self):
        self.is_train = True

    def set_eval_mode(self):
        self.is_train = False

    def forward(self, given_enc, x_enc, x_mark_enc, meta_x=None, output_attention=False):
        """
        Args:
          given_enc:  (B, window_size, time_num + time_cat)
                      i.e. the time-continuous + time-categorical features
          x_enc:      (B, seq_len, tgt_num) = real past target
          x_mark_enc: (B, window_size, time_cov_size)
                      e.g. (month, day) for daily frequency
          meta_x:     (B, meta_num + meta_cat)
                      time-independent metadata
        """
        assert not (not self.output_attention and output_attention), \
            'model is not configured to output attention'
       
        if self.task_name != 'forecast':
            raise ValueError(f'Unknown task_name: {self.task_name}')
       
        if output_attention:
            dec_out, attns = self.forecast(given_enc, x_enc, x_mark_enc, meta_x, output_attention=True)
            return dec_out[:, -self.pred_len:, :], attns
        else:
            dec_out = self.forecast(given_enc, x_enc, x_mark_enc, meta_x, output_attention=False)
            return dec_out[:, -self.pred_len:, :]

    def forecast(self, given_enc, x_enc, x_mark_enc, meta_x=None, output_attention=False):
        """
        Main logic:
        1) Single pass for 'past' (seq_len steps).
        2) Autoregressive loop for 'future' (pred_len).
        """
        B = given_enc.size(0)

        # -------------------------------
        # Split out meta feature arrays
        # meta_x => (B, meta_num + meta_cat)
        # We'll keep continuous as meta_cont, cat as meta_cat
        # -------------------------------
        meta_cont = None
        meta_cat_ = None
        if meta_x is not None and meta_x.size(1) > 0:
            if self.meta_num > 0:
                meta_cont = meta_x[:, :self.meta_num]  # (B, meta_num)
            if self.meta_cat > 0:
                meta_cat_ = meta_x[:, self.meta_num:self.meta_num + self.meta_cat]  # (B, meta_cat)

        # Past portion => [0 : seq_len]
        time_cont_past = given_enc[:, :self.seq_len, :self.time_num]  # (B, seq_len, time_num)

        # time_cat_past => multiple columns
        time_cat_past = None
        if self.time_cat_embeddings is not None and self.time_cat > 0:
            # shape => (B, seq_len, time_cat)
            time_cat_past = given_enc[:, :self.seq_len, self.time_num:self.time_num+self.time_cat]

        time_cov_past = x_mark_enc[:, :self.seq_len, :]  # (B, seq_len, time_cov_size)
       
        # We'll feed the real past target (x_enc) => (B, seq_len, tgt_num)

        # Merge features for "past" steps
        past_input = self.merge_features(
            time_cont_past,
            time_cat_past,
            time_cov_past,
            x_enc,
            meta_cont,
            meta_cat_
        )  # => shape (B, seq_len, input_dim)

        # Single pass for the past
        lstm_out, (h, c) = self.lstm(past_input)    # => (B, seq_len, hidden_size)
        past_preds = self.projection(lstm_out)      # => (B, seq_len, tgt_num)

        # Now let's handle the future steps => [seq_len : seq_len + pred_len]
        future_preds = []
        # Start with last real past target
        last_target = x_enc[:, -1, :]  # => (B, tgt_num)

        for t in range(self.pred_len):
            # Time step = seq_len + t
            time_cont_f = given_enc[:, self.seq_len + t, :self.time_num].unsqueeze(1)  # (B,1,time_num)
           
            time_cat_f = None
            if self.time_cat_embeddings is not None and self.time_cat > 0:
                time_cat_f = given_enc[:, self.seq_len + t, self.time_num:self.time_num + self.time_cat].unsqueeze(1)
                # => shape (B,1,time_cat)
           
            time_cov_f = x_mark_enc[:, self.seq_len + t, :].unsqueeze(1)  # (B,1,time_cov_size)

            # If purely autoregressive, feed last_target as input
            # (If you have future ground-truth, you can feed that for teacher forcing)
            target_f = last_target.unsqueeze(1)  # => (B,1,tgt_num)

            one_step_input = self.merge_features(
                time_cont_f,
                time_cat_f,
                time_cov_f,
                target_f,
                meta_cont,
                meta_cat_
            )  # => (B,1,input_dim)

            out, (h, c) = self.lstm(one_step_input, (h, c))  # => (B,1,hidden_size)
            pred = self.projection(out)  # => (B,1,tgt_num)
            future_preds.append(pred)

            # Update last_target
            last_target = pred.squeeze(1)  # => (B,tgt_num)

        future_preds = torch.cat(future_preds, dim=1)  # => (B, pred_len, tgt_num)

        # Combine past + future => (B, seq_len + pred_len, tgt_num)
        all_preds = torch.cat([past_preds, future_preds], dim=1)

        if output_attention:
            dummy_attn = torch.zeros(1, 1)  # no attention in LSTM
            return all_preds, dummy_attn
        else:
            return all_preds

    def merge_features(self, time_cont, time_cat, time_cov, target, meta_cont, meta_cat_):
        """
        time_cont: (B, T, time_num)
        time_cat:  (B, T, time_cat)        - multiple cat columns
        time_cov:  (B, T, time_cov_size)
        target:    (B, T, tgt_num)
        meta_cont: (B, meta_num)           - repeated across T
        meta_cat_: (B, meta_cat)           - multiple cat columns
        Return shape: (B, T, input_dim)
        """
        B, T, _ = time_cont.shape
        feats = []

        # 1) time_cont (continuous)
        feats.append(time_cont)

        # 2) time_cat => embed each column in a loop, then concat
        if self.time_cat_embeddings is not None and time_cat is not None:
            # time_cat => shape (B, T, time_cat)
            # We'll iterate over each cat column in dim=2
            emb_list = []
            for i in range(self.time_cat):
                col_i = time_cat[:, :, i]             # => (B, T)
                emb_i = self.time_cat_embeddings[i](col_i.long())  # => (B,T,cat_embed_dim)
                emb_list.append(emb_i)
            # Concat along last dim => (B,T, time_cat*cat_embed_dim)
            time_cat_emb = torch.cat(emb_list, dim=-1)
            feats.append(time_cat_emb)

        # 3) Additional time covariates
        feats.append(time_cov)

        # 4) Target
        feats.append(target)

        # 5) meta_cont => repeat along T if present
        if meta_cont is not None and meta_cont.size(1) > 0:
            # (B, meta_num) => (B,1,meta_num) => (B,T,meta_num)
            meta_cont_3d = meta_cont.unsqueeze(1).expand(-1, T, -1)
            feats.append(meta_cont_3d)

        # 6) meta_cat => also embed each column, then repeat across T
        if self.meta_cat_embeddings is not None and meta_cat_ is not None and meta_cat_.size(1) > 0:
            # meta_cat_ => (B, meta_cat)
            emb_list = []
            for i in range(self.meta_cat):
                col_i = meta_cat_[:, i]            # => (B,)
                emb_i = self.meta_cat_embeddings[i](col_i.long())  # => (B,cat_embed_dim)
                emb_list.append(emb_i)
            # Concat => (B, sum_of_cat_embed_dims = meta_cat*cat_embed_dim)
            meta_cat_emb = torch.cat(emb_list, dim=-1)
            # => (B, 1, sum_of_cat_embed_dims) => (B, T, sum_of_cat_embed_dims)
            meta_cat_emb = meta_cat_emb.unsqueeze(1).expand(-1, T, -1)
            feats.append(meta_cat_emb)

        # Final => (B, T, input_dim)
        return torch.cat(feats, dim=-1)
