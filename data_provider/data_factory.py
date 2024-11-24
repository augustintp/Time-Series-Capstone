from data_provider.data_loader import Dataset_M5
from torch.utils.data import DataLoader, WeightedRandomSampler

data_dict = {
    'M5': Dataset_M5,
}

def data_provider(args, flag, scalers=None, df_raw=None, meta_df=None):

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.eval_batch_size  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size  # bsz for train and valid

    Data = Dataset_M5

    data_set = Data(
            args=args,
            root_path=args.root_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            scalers=scalers,
            df_raw=df_raw,
            meta_df=meta_df
        )
    print(flag, len(data_set))

    if hasattr(data_set, 'sample_weight'):
        print('Using weighted sampler...')
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=WeightedRandomSampler(data_set.sample_weight, len(data_set.sample_weight)),
            num_workers=args.num_workers,
            drop_last=drop_last)
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
    return data_set, data_loader
