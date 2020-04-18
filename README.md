# T-TA

## Introduction

**T-TA**, or **T**ransformer-based **T**ext **A**utoencoder, 
is a new deep bidirectional language model for unsupervised learning tasks.
T-TA learns the straightforward learning objectives, *language autoencoding*,
where the objective is to predict every token in a text sequence at once 
without merely copying the input to output.
Rather than fine-tuning the entire pre-trained model,
T-TA is especially beneficial to obtain contextual embeddings, 
which are fixed representations of each input token
generated from the hidden layers of the trained language model.

This repository is for the paper ["Fast and Accurate Deep Bidirectional 
Language Representations for Unsupervised Learning"](https://arxiv.org/abs/1810.04805), 
which describes our method in detail.

**This code is tested on Tensorflow 1.12.0**

T-TA is a variant of the [BERT](https://arxiv.org/abs/1810.04805) model arhitecture,
which is mostly a standard [Transformer](https://arxiv.org/abs/1706.03762) architecture.
Our code is based on [Google's BERT github](https://github.com/google-research/bert),
which includes methods for building customized vocabulary and preparing the Wikipedia dataset.


## Usage of the T-TA

### Data Prepare

We release the librispeech text-only data with pre-processing (1.66 GB tar.gz file).

```shell
cd data
wget https://milabfile.snu.ac.kr:16000/tta/data/corpus.librispeech-lower.sub-32k.tar.gz
tar -xvzf corpus.librispeech-lower.sub-32k.tar.gz
cd ..
```

Then, *corpus-eval.librispeech-lower.sub-32k.txt* and 
*corpus-train.librispeech-lower.sub-32k.txt* will be appear in *data/* folder. 
Creating tfrecords from librispeech train data will take some time.

```shell
python create_tfrecords.py \
    --input_file data/corpus-eval.librispeech-lower.sub-32k.txt \
    --vocab_file configs/vocab.enwiki-lower.sub-32k.txt \
    --output_file tfrecords/tta-librispeech-lower-sub-32k/eval.tfrecord \
    --num_output_split 1

python create_tfrecords.py \
    --input_file data/corpus-train.librispeech-lower.sub-32k.txt \
    --vocab_file configs/vocab.enwiki-lower.sub-32k.txt \
    --output_file tfrecords/tta-librispeech-lower-sub-32k/train.tfrecord
```


### Training

```shell
python run_training.py \
    --config_file configs/config.layer-3.vocab-lower.sub-32k.json \
    --input_file "tfrecords/tta-librispeech-lower-sub-32k/train-*" \
    --eval_input_file "tfrecords/tta-librispeech-lower-sub-32k/eval-*" \
    --output_dir "models/tta-layer-3-librispeech-lower-sub-32k" \
    --num_train_steps 2000000 \
    --learning_rate 0.0001
```

### Test on STS Benchmark
** numpy, scipy, and sklearn packages are needed for running run_test_stsb.py**
```shell
pythoon run_test_stsb.py
```

### License

All code *and* models are released under the Apache 2.0 license. See the
`LICENSE` file for more information.

### Citation

For now, cite [the Arxiv paper](https://arxiv.org/abs/1810.04805):

```
@article{shin2020fast,
  title={Fast and Accurate Deep Bidirectional Language Representations for Unsupervised Learning},
  author={Shin, Joongbo and Lee, Yoonhyung and Yoon, Seunghyun and Jung, Kyomin},
  journal={arXiv preprint arXiv:1810.04805},
  year={2020}
}
```

### Contact information

For help or issues using T-TA, please submit a GitHub issue.

For personal communication related to T-TA, please contact Joongbo Shin 
(`jbshin@snu.ac.kr`).
