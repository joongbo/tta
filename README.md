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

T-TA is a variant of the [BERT](https://arxiv.org/abs/1810.04805) model arhitecture,
which is mostly a standard [Transformer](https://arxiv.org/abs/1706.03762) architecture.
Our code is based on [Google's BERT github](https://github.com/google-research/bert),
which includes methods for building customized vocabulary and preparing the Wikipedia dataset.

### This code is tested on

```
Python 3.6.10
TensorFlow 1.12.0
```

## Usage of the T-TA

```shell
git clone https://github.com/joongbo/tta.git
cd tta
```

### Pre-trained Model

We release the pre-trained T-TA model (262.2 MB tar.gz file).
For now, the model works on `max_seq_length=128`

```shell
cd model
wget https://milabfile.snu.ac.kr:16000/tta/data/tta-layer-3-enwiki-lower-sub-32k.tar.gz
tar -xvzf tta-layer-3-enwiki-lower-sub-32k.tar.gz
cd ..
```

### Task: Unsupervised Semantic Textual Similarity on STS Benchmark

We release the code `run_unsupervisedstsb.py` as an example of the usage of T-TA.
For running this code, you may need several python packages: `numpy`, `scipy`, and `sklearn`

To obtain the STS Benchmark dataset,
```shell
cd data
wget http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz
tar -xvzf Stsbenchmark.tar.gz
cd ..
```
Then, `stsbenchmark` folder will be appear in `data/` folder. 

Run:
```shell
pythoon run_unsupervisedstsb.py \
    --config_file models/tta-layer-3-enwiki-lower-sub-32k/config.layer-3.vocab-lower.sub-32k.json \
    --model_checkpoint models/tta-layer-3-enwiki-lower-sub-32k/model.ckpt-2000000 \
    --vocab_file models/tta-layer-3-enwiki-lower-sub-32k/vocab-lower.sub-32k.txt
```

Output:
```
STSb-dev 'context': 71.5
STSb-dev 'embed': 71.5
STSb-test 'context': 71.5
STSb-test 'embed': 71.5
```

### Training: Language AutoEncoding with T-TA

#### Data Prepare

We release the *pre-processed* librispeech text-only data (1.66 GB tar.gz file).
In this corpus, each line is a single sentence, 
so we use the sentence unit (rather than the paragraph unit) for a training instance.
The original data can be found in [LibriSpeech-LM](http://www.openslr.org/11/).

```shell
cd data
wget https://milabfile.snu.ac.kr:16000/tta/data/corpus.librispeech-lower.sub-32k.tar.gz
tar -xvzf corpus.librispeech-lower.sub-32k.tar.gz
cd ..
```
Then, `corpus-eval.librispeech-lower.sub-32k.txt` and 
`corpus-train.librispeech-lower.sub-32k.txt` will be appear in `data/` folder. 

After getting the pre-processed plain text data, we make tfrecords
(it takes some time for creating tfrecords of train data):

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
