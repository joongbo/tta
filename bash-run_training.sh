CUDA_VISIBLE_DEVICES=0 python run_training.py \
    --config_file configs/config.layer-3.vocab-lower.sub-32k.json \
    --input_file "tfrecords/tta-librispeech-lower-sub-32k/train-*" \
    --eval_input_file "tfrecords/tta-librispeech-lower-sub-32k/eval-*" \
    --output_dir "models/tta-layer-3-librispeech-lower-sub-32k" \
    --num_train_steps 2000000 \
    --learning_rate 0.0001