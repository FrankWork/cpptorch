

export DIR=.
export DATA_DIR=.
export BERT_BASE_DIR=chinese_L-12_H-768_A-12_atec

python run_classifier.py \
  --task_name atec \
  --no_cuda \
  --do_submit \
  --test_in_file $1 \
  --test_out_file $2 \
  --do_lower_case \
  --data_dir $DATA_DIR \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin \
  --max_seq_length 169 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/atec_output/
