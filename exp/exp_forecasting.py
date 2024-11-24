import json
import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch import optim
from tqdm import tqdm, trange

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import plot_by_time_group

warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):
    def __init__(self, args, writer):
        self.criterion = None
        super(Exp_Forecast, self).__init__(args, writer)
        self.model_name = args.model

    def init_model(self, dataset):
        if self.model is None:
            self.args.time_cat_embed = dataset.time_cat_embed
            self.args.time_cov_size = dataset.time_cov_size
            self.args.time_num = dataset.time_num
            self.args.time_cat = dataset.time_cat
            self.args.time_num_class = dataset.time_num_class
            self.args.tgt_num = dataset.tgt_num
            self.args.meta_cat_embed = dataset.meta_cat_embed
            self.args.meta_num = dataset.meta_num
            self.args.meta_cat = dataset.meta_cat
            self.args.meta_num_class = dataset.meta_num_class

        super(Exp_Forecast, self).init_model()

    def _get_data(self, flag, scalers=None, df_raw=None, meta_df=None):
        data_set, data_loader = data_provider(self.args, flag, scalers, df_raw, meta_df)
        return data_set, data_loader

    def _select_optimizer(self):
        no_decay = ["bias", "norm2", "norm1"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.get_model().named_parameters() if not any(nd in n for nd in no_decay)],
            },
            {
                "params": [p for n, p in self.get_model().named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }]
        model_optim = optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        self.criterion = nn.MSELoss()

    def vali(self, vali_data, vali_loader, global_step):
        self.init_model(vali_data)
        self.model.set_eval_mode()

        all_time, all_id, all_pred, all_true = [], [], [], []
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_tuple in tqdm(vali_loader):
                batch_time, batch_id = batch_tuple[:2]
                batch_given, batch_x, batch_y, batch_mark, batch_meta_x = map(lambda x: x.float().to(self.device),
                                                                                batch_tuple[2:])

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        dec_out = self.model(batch_given, batch_x, batch_mark, meta_x=batch_meta_x)
                else:
                    dec_out = self.model(batch_given, batch_x, batch_mark, meta_x=batch_meta_x)

                all_time.append(batch_time)
                all_id.append(batch_id)
                all_pred.append(dec_out[:, -self.args.pred_len:].data.cpu().numpy())
                all_true.append(batch_y[:, -self.args.pred_len:].data.cpu().numpy())

                tgt_loss = self.criterion(dec_out[:, -self.args.pred_len:], batch_y[:, -self.args.pred_len:]).item()
                total_loss.append(tgt_loss)

        total_loss = np.average(np.array(total_loss))

        if self.writer is not None:
            self.writer.add_scalar("vali/vali_loss", total_loss, global_step)

        all_time = torch.cat(all_time, dim=0).data.numpy()
        all_id = torch.cat(all_id, dim=0).data.numpy()
        all_pred = np.concatenate(all_pred, axis=0)
        all_true = np.concatenate(all_true, axis=0)
        forecast_horizon = all_time.shape[1]
        num_batch = all_time.shape[0]
        num_tgt = all_pred.shape[-1]

        scaled_gt = np.stack(
            [vali_data.inverse_transform(all_true[i, :, -self.args.c_out:]) for i in range(num_batch)], axis=0)
        scaled_pd = np.stack(
            [vali_data.inverse_transform(all_pred[i, :, -self.args.c_out:]) for i in range(num_batch)], axis=0)

        horizon_idx = np.repeat(np.arange(forecast_horizon)[np.newaxis, :], num_batch, axis=0)
        df = pd.DataFrame({'(KEY) time_from_start': all_time.reshape(-1),
                           'id': all_id.reshape(-1),
                           'horizon': horizon_idx.reshape(-1)})
        scaled_gt = scaled_gt.reshape(-1, num_tgt)
        cluster_gt_cols = []
        all_pred_cols = []
        assert num_tgt == len(vali_data.tgt_cols)
        for i, tgt_col in enumerate(vali_data.tgt_cols):
            df[f'gt_{tgt_col}'] = scaled_gt[:, i]
            cluster_gt_cols.append(f'gt_{tgt_col}')
            all_pred_cols.append(tgt_col + '_pred')
            df[tgt_col + '_pred'] = scaled_pd[:, :, i].reshape(-1)

        df = df[cluster_gt_cols + all_pred_cols +
                ['id', '(KEY) time_from_start']].groupby(['id', '(KEY) time_from_start']).mean().reset_index()
        all_mae, all_mse, _, _, _ = metric(df[all_pred_cols].to_numpy(), df[cluster_gt_cols].to_numpy())
        print('VALIDATION - mse:{:.3f}, mae:{:.3f}'.format(all_mse, all_mae))

        if self.writer is not None:
            self.writer.add_scalar("vali_scaled/mae", all_mae, global_step)
            self.writer.add_scalar("vali_scaled/mse", all_mse, global_step)

        self.model.train()
        self.model.set_train_mode()
        return all_mae

    def train(self, setting):
        eval_every = 500
        save_every = 500
        assert save_every % eval_every == 0

        train_data, train_loader = self._get_data(flag='train')
        scalers = train_data.id_scaler, train_data.scaler, train_data.scaler_tgt, train_data.scaler_meta
        vali_data, vali_loader = self._get_data(flag='val', scalers=scalers, df_raw=train_data.df_raw,
                                                meta_df=train_data.meta_df)
        test_data, test_loader = self._get_data(flag='test', scalers=scalers, df_raw=train_data.df_raw,
                                                meta_df=train_data.meta_df)
        self.init_model(train_data)

        path = os.path.join('runs', setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # only save serializable args
        args_dict = vars(self.args)
        serializable_args = {}
        for key, value in args_dict.items():
            try:
                json.dumps(value)  # This will succeed if value is serializable
                serializable_args[key] = value
            except TypeError:
                serializable_args[key] = None  # or 'Non-serializable item'
        with open(f'runs/{setting}/args.json', 'w') as fp:
            json.dump(serializable_args, fp, sort_keys=True, indent=4)

        time_now = time.time()

        model_optim = self._select_optimizer()
        self._select_criterion()
        scheduler = self._select_scheduler(model_optim, self.args.learning_rate, self.args.total_iters)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        self.model.train()
        global_step = 0
        best_vali_metrics, best_test_metrics, best_val_step = 1e3, (1e3, 1e3, 1e3), 0
        train_loss = []
        iterator = iter(train_loader)
        model_optim.zero_grad()

        for it in trange(self.args.total_iters * self.accum_iter):

            try:  # https://stackoverflow.com/a/58876890/8365622
                batch_tuple = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch_tuple = next(iterator)

            batch_given, batch_x, batch_y, batch_mark, batch_meta_x = map(lambda x: x.float().to(self.device),
                                                                            batch_tuple[2:])

            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    dec_out = self.model(batch_given, batch_x, batch_mark, meta_x=batch_meta_x)
            else:
                dec_out = self.model(batch_given, batch_x, batch_mark, meta_x=batch_meta_x)

            tgt_loss = self.criterion(dec_out[:, -self.args.pred_len:], batch_y[:, -self.args.pred_len:])
            train_loss.append(tgt_loss.item())

            tgt_loss = tgt_loss / self.accum_iter
            if self.args.use_amp:
                scaler.scale(tgt_loss).backward()
            else:
                tgt_loss.backward()

            if it % self.accum_iter == 0:
                loss_value = sum(train_loss) / len(train_loss)
                train_loss = []

                global_step += 1
                max_grad_norm_b4 = max(p.grad.data.norm(2).item() for p in self.get_model().parameters()
                                       if p.grad is not None)
                clip_grad_norm_(self.get_model().parameters(), self.clip_grad)
                max_grad_norm = max(p.grad.data.norm(2).item() for p in self.get_model().parameters()
                                    if p.grad is not None)

                if self.args.use_amp:
                    scaler.step(model_optim)
                    scaler.update()
                    scheduler.step()
                    model_optim.zero_grad()
                else:
                    model_optim.step()
                    scheduler.step()
                    model_optim.zero_grad()

                if self.writer is not None:
                    self.writer.add_scalar("train/train_loss", loss_value, global_step)
                    self.writer.add_scalar("train/max_grad_norm_b4", max_grad_norm_b4, global_step)
                    self.writer.add_scalar("train/max_grad_norm", max_grad_norm, global_step)
                    self.writer.add_scalar("train/learning_rate", model_optim.param_groups[0]["lr"],
                                           global_step)

                if (global_step + 1) % 500 == 0:
                    print(f"\titers: {it + 1} | loss: {loss_value:.7f}")
                    speed = (time.time() - time_now) / global_step
                    left_time = speed * (self.args.total_iters - it)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')

                if (global_step + 1) % eval_every == 0:
                    if (global_step + 1) % save_every == 0:
                        last_checkpoint_path = os.path.join(path, f'iter_{global_step}_checkpoint.pth')
                        torch.save(self.model.state_dict(), last_checkpoint_path)
                    vali_metrics = self.vali(vali_data, vali_loader, global_step)
                    merged_df, test_unscaled_mse, test_mae, test_mse \
                        = self.test_base(test_data, test_loader, setting, global_step)

                    if vali_metrics < best_vali_metrics:
                        best_vali_metrics = vali_metrics
                        best_test_metrics = (test_unscaled_mse, test_mae, test_mse)
                        best_val_step = global_step
                        best_model_path = os.path.join(path, f'best_checkpoint.pth')
                        torch.save(self.model.state_dict(), best_model_path)
                        folder_path = './test_results/' + setting + '/'
                        merged_df.to_csv(folder_path + 'pred.csv', index=False)

                    if self.writer is not None:
                        self.writer.add_scalar("best_test/best_val_step", best_val_step, global_step)
                        self.writer.add_scalar("best_test/unscaled_mse", best_test_metrics[0], global_step)
                        self.writer.add_scalar("best_test/mae", best_test_metrics[1], global_step)
                        self.writer.add_scalar("best_test/mse", best_test_metrics[2], global_step)

                    print(f"Step: {global_step} | Train: {loss_value:.5f} Vali: {vali_metrics:.5f} "
                          f"Test: {test_unscaled_mse:.5f} | Best Step: {best_val_step}, Vali: {best_vali_metrics:.5f}, "
                          f"Test: {best_test_metrics[0]:.5f}")

        best_model_path = path + '/' + 'best_checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        folder_path = './test_results/' + setting + '/'
        test_data, test_loader = self._get_data(flag='test')
        self.init_model(test_data)
        merged_df, unscaled_mse, _, _ = self.test_base(test_data, test_loader, setting, global_step=-1)
        merged_df.to_csv(folder_path + 'pred.csv', index=True)
        return unscaled_mse

    def test_base(self, test_data, test_loader, setting, global_step):
        self.model.set_eval_mode()
        output_attn_bool = self.args.output_attention

        all_time, all_id, all_pred, all_true, all_attn = [], [], [], [], []
        total_loss = []
        folder_path = './test_results/' + setting + '/'

        self.model.eval()
        with torch.no_grad():
            for batch_tuple in tqdm(test_loader):
                batch_time, batch_id = batch_tuple[:2]
                batch_given = batch_tuple[2].float().to(self.device)
                batch_x = batch_tuple[3].float().to(self.device)
                batch_y = batch_tuple[4]
                batch_mark = batch_tuple[5].float().to(self.device)
                batch_meta_x = batch_tuple[6].float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        output_tuple = self.model(batch_given, batch_x, batch_mark, meta_x=batch_meta_x,
                                                  output_attention=output_attn_bool)
                else:
                    output_tuple = self.model(batch_given, batch_x, batch_mark, meta_x=batch_meta_x,
                                              output_attention=output_attn_bool)
                if output_attn_bool:
                    dec_out, attns = output_tuple
                    all_attn.append(attns.mean(dim=0).cpu().numpy())
                else:
                    dec_out = output_tuple

                all_time.append(batch_time)
                all_id.append(batch_id)
                all_pred.append(dec_out[:, -self.args.pred_len:].data.cpu().numpy())
                all_true.append(batch_y[:, -self.args.pred_len:].data.numpy())

                tgt_loss = self.criterion(dec_out[:, -self.args.pred_len:].cpu(),
                                          batch_y[:, -self.args.pred_len:]).item()

                total_loss.append(tgt_loss)

        self.model.set_train_mode()
        if output_attn_bool:
            all_attn = np.stack(all_attn, axis=0).mean(axis=0)
            np.save(os.path.join(folder_path, 'attn.npy'), all_attn)

        total_loss = np.average(np.array(total_loss))

        if self.writer is not None:
            self.writer.add_scalar("test_loss/vali_loss", total_loss, global_step)
            self.writer.add_scalar("test_loss/tgt_loss", total_loss, global_step)

        all_time = torch.cat(all_time, dim=0).data.numpy()
        all_id = torch.cat(all_id, dim=0).data.numpy()
        all_pred = np.concatenate(all_pred, axis=0)
        all_true = np.concatenate(all_true, axis=0)
        forecast_horizon = all_time.shape[1]
        num_batch = all_time.shape[0]
        num_tgt = all_pred.shape[-1]

        scaled_gt = np.stack(
            [test_data.inverse_transform(all_true[i, :, -self.args.c_out:]) for i in range(num_batch)], axis=0)
        scaled_pd = np.stack(
            [test_data.inverse_transform(all_pred[i, :, -self.args.c_out:]) for i in range(num_batch)], axis=0)
        horizon_idx = np.repeat(np.arange(forecast_horizon)[np.newaxis, :], num_batch, axis=0)
        df = pd.DataFrame({'(KEY) time_from_start': all_time.reshape(-1),
                           'id': all_id.reshape(-1),
                           'horizon': horizon_idx.reshape(-1)})
        scaled_gt = scaled_gt.reshape(-1, num_tgt)
        gt_cols = []
        all_pred_cols = []
        assert num_tgt == len(test_data.tgt_cols)
        for i, tgt_col in enumerate(test_data.tgt_cols):
            df[f'gt_{tgt_col}'] = scaled_gt[:, i]
            gt_cols.append(f'gt_{tgt_col}')
            all_pred_cols.append(tgt_col + '_pred')
            df[tgt_col + '_pred'] = scaled_pd[:, :, i].reshape(-1)

        df = df[gt_cols + all_pred_cols +
                ['id', '(KEY) time_from_start']].groupby(['id', '(KEY) time_from_start']).mean().reset_index()

        # Join on time_from_start and cluster_id. Apply to raw dataframe for each instance.
        merged_df = pd.merge(test_data.eval_df, df, on=['(KEY) time_from_start', 'id'], how='left')
        metrics_df = merged_df[merged_df['(KEY) time_from_start'] >= test_data.border1 + self.args.seq_len]

        # calculate metrics
        all_mae, all_mse, _, _, _ = metric(metrics_df[all_pred_cols].to_numpy(),
                                           metrics_df[test_data.tgt_cols].to_numpy())

        if self.writer is not None:
            self.writer.add_scalar("test/mae", all_mae, global_step)
            self.writer.add_scalar("test/mse", all_mse, global_step)

        print(f'TEST - mse:{all_mse:.3f}, mae:{all_mae:.3f}')

        grouped_metrics_df = pd.merge(merged_df, test_data.plot_meta_df, on=['id'], how='left')
        grouped_metrics_df['month'] = grouped_metrics_df['date'].dt.month
        # Group by necessary columns
        grouped = grouped_metrics_df.groupby(['state_id', 'cat_id', 'month'])

        # Calculate metrics for each group
        def calculate_metrics(group):
            pred = group['target']
            true = group['gt_target']
            mae, mse, _, _, _ = metric(pred, true)
            return pd.Series({'mae': mae, 'mse': mse})

        # Apply the function to each group
        grouped_metrics_df = grouped.apply(calculate_metrics).reset_index()
        grouped_metrics_df.to_csv(os.path.join(folder_path, 'metrics.csv'), index=False)
        del grouped_metrics_df, grouped

        # visualize results for each day/month
        plot_by_time_group(global_step, setting, folder_path, merged_df, all_pred_cols, test_data.tgt_cols,
                           plot_title=f'Step{global_step}', writer=self.writer)

        all_ids = sorted(merged_df['id'].unique())
        # random version
        random_visual_samples = np.random.choice(all_ids, size=20, replace=False)
        # deterministic version
        # random_visual_samples = [all_ids[i] for i in range(0, len(all_ids), len(all_ids) // 20)]

        y_pred = np.arange(self.args.seq_len + self.args.pred_len)
        for sample_n, visual_id in enumerate(random_visual_samples):
            visual_id_df = merged_df[merged_df['id'] == visual_id][
                test_data.time_cov_cols + test_data.tgt_cols + all_pred_cols +
                ['date', '(KEY) time_from_start']].groupby(['(KEY) time_from_start']).mean().reset_index()
            id_df_plot_meta = test_data.plot_meta_df[test_data.plot_meta_df['id'] == visual_id] \
                [test_data.plot_meta_cols].iloc[0]

            # random version
            # plot_start_time = random.randrange(visual_id_df['(KEY) time_from_start'].min(),
            #                                    visual_id_df['(KEY) time_from_start'].max() - self.args.seq_len + 1)
            # deterministic version
            possible_start = (visual_id_df['(KEY) time_from_start'].max() - self.args.seq_len - self.args.pred_len + 1 -
                              visual_id_df['(KEY) time_from_start'].min())
            plot_start_time = (((possible_start // 10) * sample_n) % possible_start +
                               visual_id_df['(KEY) time_from_start'].min())

            nrows = 2
            ncols = max(len(test_data.time_cov_cols), len(test_data.tgt_cols))
            f = plt.figure(figsize=(4 * ncols, 8), constrained_layout=True)
            ax = np.array(f.subplots(nrows, ncols)).reshape(nrows, ncols)  # Ensure ax is always 2D
            selected_df = visual_id_df[
                (visual_id_df['(KEY) time_from_start'] < plot_start_time + self.args.seq_len + self.args.pred_len) &
                (visual_id_df['(KEY) time_from_start'] >= plot_start_time)]
            plot_ticks = (0, self.args.seq_len // 2, self.args.seq_len - 1)
            dates_labels = [
                test_data.plot_date_string(visual_id_df[visual_id_df['(KEY) time_from_start'] ==
                                                        plot_start_time + i]['date']) for i in plot_ticks]

            for k, visual_col in enumerate(test_data.time_cov_cols):
                given_total_df = selected_df[['(KEY) time_from_start', visual_col]].groupby(['(KEY) time_from_start']).mean()
                ax[0, k].plot(y_pred, given_total_df[visual_col].to_numpy())
                ax[0, k].axvline(self.args.seq_len, color='r', linestyle='dashed', alpha=0.7)
                ax[0, k].set_xticks(plot_ticks, minor=False)
                ax[0, k].set_xticklabels(dates_labels, fontdict=None, minor=False, rotation=10)
                ax[0, k].set_title(f'{visual_col}')

            for k, visual_col in enumerate(test_data.tgt_cols):
                tgt_total_df = selected_df[
                    ['(KEY) time_from_start', visual_col, f'{visual_col}_pred']].groupby(['(KEY) time_from_start']).mean()

                ax[1, k].plot(y_pred[self.args.seq_len:],
                              tgt_total_df[f'{visual_col}_pred'].to_numpy()[self.args.seq_len:], label='pred')
                ax[1, k].plot(y_pred, tgt_total_df[visual_col].to_numpy(), label='GT')
                ax[1, k].axvline(self.args.seq_len, color='r', linestyle='dashed', alpha=0.7)
                ax[1, k].set_xticks(plot_ticks, minor=False)
                ax[1, k].set_xticklabels(dates_labels, fontdict=None, minor=False, rotation=10)
                ax[1, k].set_title(f'Predicted {visual_col}')
                ax[1, k].legend()

            f.suptitle(f'ID {visual_id}: ' + '-'.join(list(id_df_plot_meta)))
            f.savefig(os.path.join(folder_path, 'figures',
                                   f'step{global_step}_ID{visual_id}_{"-".join(list(id_df_plot_meta))}.png'),
                      bbox_inches='tight')
            plt.close()

        self.model.train()
        return merged_df, total_loss, all_mae, all_mse
