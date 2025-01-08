import os
import torch
from models import Linear, DeepAR, Enc_Only_Transformer, Enc_Dec_Transformer

from torch import nn
from torch.nn.parallel import DistributedDataParallel
from utils.tools import CosAnnealWarmupRestarts


class Exp_Basic(object):
    def __init__(self, args, writer):
        self.args = args
        self.root_path = args.root_path
        self.accum_iter = args.accum_iter
        self.writer = writer
        self.model_dict = {
            'Linear': Linear,
            'DeepAR': DeepAR,
            'Enc_Only_Transformer': Enc_Only_Transformer,
            'Enc_Dec_Transformer': Enc_Dec_Transformer
        }
        self.device = self._acquire_device()
        self.model = None
        self.last_checkpoint_path = None
        self.clip_grad = args.clip_grad
        self.loaded_checkpoint = False

    def init_model(self, dataset=None):
        if self.model is None:
            self.model = self._build_model().to(self.device)
            if self.args.load_checkpoint is not None:
                self.loading_checkpoint()
            else:
                self.loaded_checkpoint = False

    def loading_checkpoint(self):
        self.loaded_checkpoint = True
        if torch.cuda.is_available():
            checkpoint = torch.load(self.args.load_checkpoint, map_location='cuda')
        else:
            checkpoint = torch.load(self.args.load_checkpoint, map_location='cpu')

        model_state_dict = {k: v for k, v in checkpoint.items()}
        self.model.load_state_dict(model_state_dict, strict=False)

    def _build_model(self):
        model_fn = getattr(self.model_dict[self.args.model], self.args.model)
        model = model_fn(self.args, self.device).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _select_scheduler(self, optimizer, lr, max_steps):
        warmup_steps = max_steps * self.args.warmup_ratio
        scheduler = CosAnnealWarmupRestarts(optimizer, max_lr=lr, min_lr=0.0,
                                            warmup_steps=warmup_steps, max_steps=max_steps, alpha=0)
        return scheduler

    def get_model(self):
        """Return the model maybe wrapped inside `model`."""
        return self.model.module if isinstance(self.model, (DistributedDataParallel, nn.DataParallel)) else self.model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self, setting):
        pass

    def test(self):
        pass
