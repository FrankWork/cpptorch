
export DIR=${HOME}/bert
export DATA_DIR=${DIR}/atec_nlp
#export BERT_BASE_DIR=${DIR}/chinese_L-12_H-768_A-12
export BERT_BASE_DIR=${DIR}/atec_submit/chinese_L-12_H-768_A-12_atec
#  --do_train \
python run_classifier.py \
  --task_name atec \
  --gpu_id 0 \
  --do_eval \
  --train_devset \
  --do_lower_case \
  --data_dir $DATA_DIR \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin \
  --max_seq_length 169 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir atec_output/
