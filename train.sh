#!/bin/bash
FOLDER=("fiveflow-t5-small", "fiveflow-t5-base", "fiveflow-t5-large", "fiveflow-t5-xlarge")
BSZ = ("128", "64", "32", "16")

for (( i=0; i<4; i++ ))
do
    python run_t5_mlm_flax.py \
	--output_dir=FOLDER[$i]\
	--model_type="t5" \
	--config_name=FOLDER[$i]\
	--tokenizer_name=FOLDER[$i]\
    --dataset_name="fiveflow/processed_pretraining"\
	--max_seq_length="512" \
	--per_device_train_batch_size=BSZ[$i] \
	--per_device_eval_batch_size=BSZ[$i] \
	--adafactor=True \
    --num_train_epochs=5.0\
	--learning_rate="0.005" \
	--weight_decay="0.001" \
	--warmup_steps="2000" \
	--overwrite_output_dir \
	--logging_steps="500" \
	--save_steps="10000" \
	--eval_steps="2500" \
	
done