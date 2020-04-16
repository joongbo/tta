## librispeech-32k
python create_tfrecords.py \
    --input_file data/corpus-eval.librispeech-lower.sub-32k.txt \
    --vocab_file configs/vocab.enwiki-lower.sub-32k.txt \
    --output_file tfrecords/tta-librispeech-lower-sub-32k/eval.tfrecord \
    --num_output_split 1

python create_tfrecords.py \
    --input_file data/corpus-train.librispeech-lower.sub-32k.txt \
    --vocab_file configs/vocab.enwiki-lower.sub-32k.txt \
    --output_file tfrecords/tta-librispeech-lower-sub-32k/train.tfrecord