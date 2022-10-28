nohup python run_race_1.py \
--model_name_or_path bert-base-uncased \
--do_train \
--do_eval \
--learning_rate 1e-5   \
--num_train_epochs 3  \
--output_dir /tmp/race_base \
--per_gpu_eval_batch_size=32 \
--per_device_train_batch_size=1 \
--overwrite_output \
--fp16 \
>> output_bert_base.out 2>&1 &