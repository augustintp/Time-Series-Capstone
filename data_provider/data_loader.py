import os
import numpy as np
import pandas as pd
from pathlib import Path
import data_provider.transforms as transforms
from tqdm import tqdm
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')


class Dataset_M5(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, sales_path='sales_train_evaluation.csv',
                 calendar_path='calendar.csv', price_path='sell_prices.csv', scalers=None,
                 df_raw=None, meta_df=None):
        """
        train and validation uses household 0-5000, split by home_id
        test uses household 15000-20000
        """
        self.model = args.model
        self.test_results_path = os.path.join('test_results', args.setting, 'figures')
        Path(self.test_results_path).mkdir(exist_ok=True, parents=True)

        if size is None:
            self.seq_len = 90
            self.label_len = 0
            self.pred_len = 10
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        self.window_size = self.seq_len + self.pred_len

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.freq = 'daily'

        self.root_path = root_path
        self.sales_path = sales_path
        self.calendar_path = calendar_path
        self.price_path = price_path
        self.df_raw = df_raw
        self.meta_df = meta_df
        self.data_version = args.data_version
        if scalers is not None:
            self.id_scaler, self.scaler, self.scaler_tgt, self.scaler_meta = scalers
        else:
            self.id_scaler, self.scaler, self.scaler_tgt, self.scaler_meta = None, None, None, None

        self.__extract_cols__()
        if self.df_raw is None:
            self.df_raw, self.meta_df = self.__read_data__()
        self.max_time = max(self.df_raw['time_from_start'])
        self.__process_data__()

    def __extract_cols__(self):
        """
        Variables needed in model:
        - key_cols (list of str): columns that are used as keys to uniquely identify each row in df_raw
        - meta_key_cols (list of str): columns that are used as keys to uniquely identify each row in meta_df
        - time_cov_cols (list of str): all the time-dependent variables used in the model, as found in df_raw
        - tgt_cols (list of str): all the target variables used in the model, as found in df_raw
        - time_num (int): number of time-dependent, numerical variables
        - time_cat (int): number of time-dependent, categorical variables
        - time_num_class (list of int): number of possible categories in each time-dependent, categorical variable
        - time_cat_embed (list of int): number of possible categories in each time-dependent, categorical variable
        - meta_cols (list of str): all the time-independent variables used in the model, as found in meta_df
        - meta_num (int): number of time-independent, numerical variables
        - meta_cat (int): number of time-independent, categorical variables
        - meta_num_class (list of int): number of possible categories in each time-independent, categorical variable
        - meta_cat_embed (list of int): number of possible categories in each time-independent, categorical variable
        - plot_meta_cols (list of str): columns in meta_df that are used for plotting
        """

        # For time_cov_cols, tgt_cols, and meta_cols, numerical cols should be placed before categorical cols
        self.key_cols = ['date', '(KEY) time_from_start', 'id']
        self.meta_key_cols = ['id']
        self.timestamp_cov_cols = ['month', 'day']
        self.time_cov_size = len(self.timestamp_cov_cols)
        self.plot_meta_cols = ['store_id', 'cat_id', 'state_id']
        if self.data_version == 'v0':
            self.time_cov_cols = ['time_from_start', 'snap_accepted', 'Sporting', 'Cultural', 'National', 'Religious']
            self.time_cat_embed = [2, 2, 2, 2, 2]
            self.tgt_cols = ['target']
            self.meta_cols = ['store_id', 'cat_id', 'state_id']
            self.meta_cat_embed = [3, 3, 3]
        else:
            raise ValueError(f'Unsupported data version: {self.data_version}!')

        self.var_cols = self.time_cov_cols + self.timestamp_cov_cols + self.meta_cols + self.tgt_cols
        name_map = {
            'month': 'Month',
            'day': 'Day',
            'time_from_start': 'Time from Start',
            'snap_accepted': 'Whether SNAP is Accepted',
            'Sporting': 'Whether is a Sporting Event',
            'Cultural': 'Whether is a Cultural Event',
            'National': 'Whether is a National Event',
            'Religious': 'Whether is a Religious Event',
            'store_id': 'Store ID',
            'cat_id': 'Category ID',
            'state_id': 'State ID',
            'target': 'Target'
        }
        self.var_cols = [name_map[col] if col in name_map else col for col in self.var_cols]
        with open(os.path.join(self.test_results_path, 'var_cols.txt'), 'w') as f:
            for line in self.var_cols:
                f.write(f"{line}\n")
        print('length var_cols:', len(self.var_cols))

        self.time_cat = len(self.time_cat_embed)
        self.time_num = len(self.time_cov_cols) - self.time_cat
        self.tgt_len = len(self.tgt_cols)
        self.tgt_num = self.tgt_len
        self.meta_cat = len(self.meta_cat_embed)
        self.meta_num = len(self.meta_cols) - self.meta_cat
        print(f'time_num: {self.time_num}, time_cat: {self.time_cat}, tgt_num: {self.tgt_num}, '
              f'meta_num: {self.meta_num}, meta_cat: {self.meta_cat}')

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.sales_path))
        calendar_df = pd.read_csv(os.path.join(self.root_path, self.calendar_path))

        # TODO: price_df, which contains the sales price for each item, is not currently used
        # price_df = pd.read_csv(os.path.join(self.root_path, self.price_path))

        cat_cols = ['id', 'cat_id', 'store_id', 'state_id']
        num_cols = [col for col in df_raw.columns if col not in (cat_cols + ['item_id', 'dept_id'])]

        # Aggregate the products at the category level for each store
        numeric_grouped = df_raw.groupby(['store_id', 'cat_id'])[num_cols].sum()

        # For categorical columns, keep only unique entries for each group
        cat_grouped = df_raw.groupby(['store_id', 'cat_id'])[cat_cols].first()

        df_raw = pd.concat([cat_grouped, numeric_grouped], axis=1)
        df_raw = df_raw.drop(columns=['cat_id', 'store_id']).reset_index()
        df_raw['id'] = df_raw.apply(lambda row: str(row['store_id']) + '-' + str(row['cat_id']), axis=1)

        meta_df = df_raw[['id', 'cat_id', 'store_id', 'state_id']]

        df_raw = df_raw.drop(columns=[col for col in meta_df.columns if col not in self.meta_key_cols])
        df_raw = pd.melt(df_raw, id_vars='id', var_name='date', value_name='target')

        calendar_df.set_index(['d'], inplace=True)
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
        all_min_date = min(calendar_df['date'])
        calendar_df['time_from_start'] = (calendar_df['date'] - all_min_date).dt.days
        df_raw['time_from_start'] = df_raw['date'].map(calendar_df['time_from_start'].to_dict())
        df_raw['date'] = df_raw['date'].map(calendar_df['date'].to_dict())
        assert not df_raw['time_from_start'].isnull().values.any(), \
            'All dates in df_raw must be found in calendar_df!'

        event_types = ['Sporting', 'Cultural', 'National', 'Religious']
        for event in event_types:
            calendar_df[event] = False
        # Set True where event_type_1 or event_type_2 matches the event type
        for event in event_types:
            calendar_df[event] = calendar_df[['event_type_1', 'event_type_2']].apply(
                lambda row: event in row.values if not row.isnull().all() else False, axis=1
            )

        df_raw['state'] = df_raw['id'].str[:2]

        # Select only the necessary columns from calendar_df (date + snap columns)
        snap_columns = [col for col in calendar_df.columns if col.startswith('snap_')]

        # Merge df_raw with the filtered calendar_df on date
        merged_df = df_raw.merge(calendar_df[['date'] + snap_columns], on='date', how='left')

        # Create the snap_accepted column by looking up the appropriate snap_{state} column
        merged_df['snap_accepted'] = merged_df.apply(
            lambda row: row[f'snap_{row["state"]}'] if pd.notnull(row[f'snap_{row["state"]}']) else False, axis=1
        )
        df_raw = merged_df.drop(columns=['state'] + snap_columns)
        calendar_df.drop(
            columns=['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'wm_yr_wk', 'weekday', 'wday',
                     'month', 'year', 'date'] + snap_columns, inplace=True)
        df_raw = df_raw.merge(calendar_df, on='time_from_start', how='left')
        print('After merging with calendar_df: ')
        print(df_raw)
        return df_raw, meta_df

    def __process_data__(self):
        df_raw = self.df_raw.copy(deep=True)
        meta_df = self.meta_df.copy(deep=True)

        # Append "(KEY) " in front of self.key_cols. Remove columns that are not in self.time_cov_cols
        df_raw[f'(KEY) time_from_start'] = df_raw['time_from_start']

        # train-vali-test split by time_from_start
        border1s = [0, int(self.max_time * 0.8) - self.seq_len, int(self.max_time * 0.9) - self.seq_len]
        border2s = [int(self.max_time * 0.8), int(self.max_time * 0.9), self.max_time]
        print(f'border1s: {border1s}, border2s: {border2s}')
        self.border1 = border1s[self.set_type]
        self.border2 = border2s[self.set_type]
        if self.flag == 'test':
            self.eval_df = df_raw[(self.border1 <= df_raw['(KEY) time_from_start']) &
                                  (df_raw['(KEY) time_from_start'] < self.border2)].copy()
            plot_meta_df_cols = sorted(list(set(['id'] + self.plot_meta_cols)))
            self.plot_meta_df = meta_df[plot_meta_df_cols].copy()

        '''
       Reorder columns so that they follow the format below.
       df_raw.columns: ['date', 'time_from_start', plot features, other features, total_consumption, target latents]
       '''
        all_cols = self.key_cols + self.time_cov_cols + self.tgt_cols
        cols = [col for col in list(df_raw.columns) if col not in all_cols]
        all_meta_cols = self.meta_key_cols + self.meta_cols
        meta_cols = [col for col in list(meta_df.columns) if col not in all_meta_cols]
        print(f'Time cols not used: {cols}, meta cols not used: {meta_cols}')
        df_raw = df_raw[all_cols]
        meta_df = meta_df[all_meta_cols]
        print(f'Final time cols: {df_raw.columns}, meta cols: {meta_df.columns}')
        print(df_raw)
        print(meta_df)

        if self.scaler is None:
            train_data = df_raw[(border1s[0] <= df_raw['(KEY) time_from_start']) &
                                (df_raw['(KEY) time_from_start'] < border2s[0])]
            self.id_scaler = transforms.OneHotScaler(train_data['id'])

            train_data = train_data.drop(columns=self.key_cols)
            train_meta = meta_df.drop(columns=self.meta_key_cols)
            self.scaler = []
            for i in range(len(self.time_cov_cols)):
                if i < self.time_num:
                    self.scaler.append(transforms.DeepARScaler(train_data.iloc[:, i]))
                else:
                    self.scaler.append(transforms.OneHotScaler(train_data.iloc[:, i]))

            self.scaler_tgt = []
            for i in range(self.tgt_num):
                self.scaler_tgt.append(transforms.DeepARScaler(train_data.iloc[:, -self.tgt_num + i]))

            self.scaler_meta = []
            for i in range(len(self.meta_cols)):
                if i < self.meta_num:
                    self.scaler_meta.append(transforms.DeepARScaler(train_meta.iloc[:, i]))
                else:
                    self.scaler_meta.append(transforms.OneHotScaler(train_meta.iloc[:, i]))

        self.time_num_class, self.meta_num_class = [], []

        df_raw = df_raw[(self.border1 <= df_raw['(KEY) time_from_start']) &
                        (df_raw['(KEY) time_from_start'] < self.border2)]
        print(f'Final number of selected IDs: {df_raw["id"].nunique()}')
        # onehot encode all categorical variables
        df_raw['id'] = self.id_scaler.transform(df_raw['id'])
        meta_df['id'] = self.id_scaler.transform(meta_df['id'])
        if self.flag == 'test':
            self.eval_df['id'] = self.id_scaler.transform(self.eval_df['id'])
            self.plot_meta_df['id'] = self.id_scaler.transform(self.plot_meta_df['id'])
        for i in range(self.time_num, len(self.time_cov_cols)):
            self.time_num_class.append(self.scaler[i].get_num_class())
            df_raw.iloc[:, len(self.key_cols) + i] = self.scaler[i].transform(df_raw.iloc[:, len(self.key_cols) + i])
        for i in range(self.meta_num, len(self.meta_cols)):
            self.meta_num_class.append(self.scaler_meta[i].get_num_class())
            meta_df.iloc[:, len(self.meta_key_cols) + i] = self.scaler_meta[i].transform(
                meta_df.iloc[:, len(self.meta_key_cols) + i])

        # Generate the full range of hourly datetime values
        selected_max_date = max(df_raw['date'])
        date_range = pd.date_range(start=min(self.df_raw['date']), end=selected_max_date, freq='H')

        # Create a new DataFrame
        df_timestamp = pd.DataFrame({
            'date': date_range,
            '(KEY) time_from_start': range(len(date_range))
        })

        # Calculate normalized month, day, and hour directly
        df_timestamp['month'] = (df_timestamp['date'].dt.month - 1 - 5.5) / 5.5
        df_timestamp['day'] = (df_timestamp['date'].dt.dayofweek - 3) / 3

        # Convert timestamp dataframe into a numpy array
        self.timestamp_np = df_timestamp[self.timestamp_cov_cols].to_numpy(dtype=np.float32)

        print('Getting valid sampling locations.')
        self.ranges = []
        self.split_data_map = {}
        if meta_df.shape[-1] == 1:
            self.split_meta_map = None
        else:
            self.split_meta_map = meta_df.set_index('id').apply(lambda row: np.array(row, dtype=np.float32),
                                                                axis=1).to_dict()

        df_raw.drop(columns='date', inplace=True)
        limited_ids = []
        for identifier, sliced in tqdm(df_raw.groupby('id')):
            self.split_data_map[identifier] = sliced.to_numpy().astype(np.float32)
            min_id_time, max_id_time = int(min(sliced['(KEY) time_from_start'])), int(max(sliced['(KEY) time_from_start']))
            num_entries = len(sliced)
            if max_id_time - min_id_time + 1 >= self.window_size:
                if self.set_type != 0:
                    self.ranges += [(identifier,
                                     i + min_id_time, i) for i in range(0, num_entries - self.window_size + 1, 1)]
                    # non-overlapping evaluation, can change the last param
                else:  # subsample for train
                    valid_idx = np.arange(num_entries - self.window_size + 1)
                    self.ranges += [(identifier, i + min_id_time, i) for i in valid_idx]
            else:
                limited_ids.append(identifier)
                print(f'WARNING: ID={identifier} only has {num_entries} time steps!')

        self.total_windows = len(self.ranges)
        print(f'Total windows for {self.flag}: {self.total_windows}')

    def __getitem__(self, index):
        identifier, time_from_start_idx, start_idx = self.ranges[index]

        seq_given_unscaled = self.split_data_map[identifier][start_idx: start_idx + self.window_size,
                         len(self.key_cols) - 1:-self.tgt_len]
        seq_given_num = np.stack([self.scaler[i].transform(seq_given_unscaled[:, i])
                              for i in range(self.time_num)], axis=-1)
        seq_given = np.concatenate([seq_given_num, seq_given_unscaled[:, self.time_num:]], axis=-1)

        seq_tgt_unscaled = self.split_data_map[identifier][start_idx: start_idx + self.window_size, -self.tgt_len:]
        seq_tgt_num = np.stack([self.scaler_tgt[i].transform(seq_tgt_unscaled[:, i])
                                for i in range(self.tgt_num)], axis=-1)
        seq_x = seq_tgt_num[:self.seq_len]
        seq_y = seq_tgt_num[self.seq_len:]

        seq_mark = self.timestamp_np[time_from_start_idx:time_from_start_idx + self.window_size]
        seq_x_time = np.arange(self.pred_len, dtype=np.float32) + time_from_start_idx + self.seq_len
        seq_id = np.full((self.pred_len,), identifier, dtype=int)

        if self.split_meta_map is not None:
            meta_x_unscaled = self.split_meta_map[identifier][:self.meta_num + self.meta_cat]
            if self.meta_num == 0:
                meta_x_num = np.array([], dtype=np.float32)
            else:
                meta_x_num = np.stack([self.scaler_meta[i].transform(meta_x_unscaled[i])
                                       for i in range(self.meta_num)], axis=-1)
            meta_x = np.concatenate([meta_x_num, meta_x_unscaled[self.meta_num:]], axis=-1)

        else:
            meta_x = np.array([], dtype=np.float32)

        return seq_x_time, seq_id, seq_given, seq_x, seq_y, seq_mark, meta_x

    def __len__(self):
        return self.total_windows

    def inverse_transform(self, data, instance_id=None, month_id=None):
        scaled_data = np.stack([self.scaler_tgt[i].scale_back(data[:, i]) for i in range(len(self.scaler_tgt))],
                               axis=-1)
        return scaled_data

    def plot_date_string(self, datetime_obj):
        if self.freq == 'daily':
            return datetime_obj.iloc[0].strftime('%Y-%m-%d')
        else:
            raise ValueError(f'freq = {self.freq} not supported')

    def inverse_transform_id(self, instance_id):
        return self.id_scaler.scale_back(instance_id)
