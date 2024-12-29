export CUDA_VISIBLE_DEVICES=7

model_name=Enc_Only_Transformer

python -u /content/Time-Series-Capstone/run.py \
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
  --d_model 32 \
  --d_ff 64 \
  --batch_size 256 \
  --learning_rate 0.0005 \
  --itr 1 \
  --track
