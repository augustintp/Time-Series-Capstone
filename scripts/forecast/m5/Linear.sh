export CUDA_VISIBLE_DEVICES=0

model_name=Linear

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/m5/ \
  --model_id simple_implementation \
  --model $model_name \
  --data M5 \
  --seq_len 90 \
  --label_len 0 \
  --pred_len 10 \
  --e_layers 1 \
  --freq 'daily' \
  --data_version 'v0' \
  --des '1118' \
  --d_model 128 \
  --d_ff 256 \
  --batch_size 64 \
  --learning_rate 0.0005 \
  --itr 1 \
  --track