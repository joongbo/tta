# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Calculate pseudo-perplexity of a sentence using TTA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import modeling
import tensorflow as tf
import tokenization
import numpy as np
import scipy as sp
import csv
from sklearn.metrics.pairwise import cosine_similarity

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("config_file", "",
                    "The config json file corresponding to the trained BERT model. "
                    "This specifies the model architecture.")

flags.DEFINE_string("model_checkpoint", "",
                    "checkpoint")

flags.DEFINE_string("vocab_file", "",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_integer("max_seq_length", 128, "The length of maximum sequence.")


class TestingInstance(object):
    """A single test instance (sentence pair)."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.input_tokens = tokens
        self.target_tokens = tokens

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()

def create_testing_instances(sentence, tokenizer, max_seq_length=128):
    """Create `TestInstance`s from raw text."""
    max_token_num = max_seq_length - 2
    
    tokens = tokenizer.tokenize(sentence)
    if len(tokens) > max_token_num:
        tokens = tokens[:max_token_num]
    if tokens[0] is not "[SOS]":
        tokens.insert(0, "[SOS]")
    if tokens[-1] is not "[EOS]":
        tokens.append("[EOS]")

    instances = []
    instances.append(create_instances_from_tokens(tokens))

    return instances   

def create_instances_from_tokens(tokens):
    """Creates `TestInstance`s for a single sentence."""
    instance = TestingInstance(tokens)

    return instance


# load tokenizer
tokenizer = tokenization.FullTokenizer(
    vocab_file = FLAGS.vocab_file, 
    do_lower_case=True)
word_to_id = tokenizer.vocab

# load trained model
config = modeling.BertConfig.from_json_file(FLAGS.config_file)
tf.reset_default_graph()
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True
sess = tf.Session(config=session_config)

input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None])

model = modeling.BertModel(
        config=config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        use_one_hot_embeddings=False)

input_tensor = model.get_sequence_output()
input_embeddings = model.get_embedding_output()

input_shape = modeling.get_shape_list(input_tensor, expected_rank=3)
input_tensor = tf.reshape(input_tensor, [input_shape[0]*input_shape[1], input_shape[2]])

saver = tf.train.Saver()
saver.restore(sess, FLAGS.model_checkpoint)
print()


# load STSb-dev-set
labels = []
refs = []
hyps = []
with open('data/stsbenchmark/sts-dev.csv') as f:
    reader = csv.reader(f, delimiter='\n')
    dev_list = []
    for line in reader:
        dev = line[0].split('\t')
        labels.append(float(dev[4]))
        refs.append(dev[5])
        hyps.append(dev[6])

        
# calculate correlation
print('Get scores on STSb-dev. Processing ..')
similarity_scores_representation = []
# similarity_scores_embeddings = []
for cnt, (ref, hyp) in enumerate(zip(refs, hyps)):
    if (cnt+1) % 200 == 0:
        print(cnt+1, end=', ')
    instances = create_testing_instances(ref, tokenizer, 
                                         FLAGS.max_seq_length)

    batch_input_ids = []
    batch_input_mask = []
    for _instance in instances:
        _input_ids = [word_to_id[_token] for _token in _instance.input_tokens]
        _input_mask = [1] * len(_input_ids)

        batch_input_ids.append(_input_ids)
        batch_input_mask.append(_input_mask)

    feed_dict = {input_ids : batch_input_ids,
                 input_mask : batch_input_mask,
                 }

    [representations_ref, embeddings_ref] = sess.run([input_tensor, input_embeddings], feed_dict=feed_dict)
    
    
    instances = create_testing_instances(hyp, tokenizer, 
                                         FLAGS.max_seq_length)

    batch_input_ids = []
    batch_input_mask = []
    for _instance in instances:
        _input_ids = [word_to_id[_token] for _token in _instance.input_tokens]
        _input_mask = [1] * len(_input_ids)

        batch_input_ids.append(_input_ids)
        batch_input_mask.append(_input_mask)

    feed_dict = {input_ids : batch_input_ids,
                 input_mask : batch_input_mask,
                 }

    [representations_hyp, embeddings_hyp] = sess.run([input_tensor, input_embeddings], feed_dict=feed_dict)

    sentence_representation_mean_ref = np.mean(representations_ref[1:-1], axis=0)
    sentence_representation_mean_hyp = np.mean(representations_hyp[1:-1], axis=0)
    score = cosine_similarity([sentence_representation_mean_ref], [sentence_representation_mean_hyp])
    similarity_scores_representation.append(score[0][0])
    
#     sentence_embeddings_mean_ref = np.mean(embeddings_ref[0][1:-1], axis=0)
#     sentence_embeddings_mean_hyp = np.mean(embeddings_hyp[0][1:-1], axis=0)
#     score = cosine_similarity([sentence_embeddings_mean_ref], [sentence_embeddings_mean_hyp])
#     similarity_scores_embeddings.append(score[0][0])

print('')
print('STSb-dev (context):', sp.stats.pearsonr(labels, similarity_scores_representation)[0])
# print('STSb-dev (embed)  :', sp.stats.pearsonr(labels, similarity_scores_embeddings)[0])


# load STSb-test-set
labels = []
refs = []
hyps = []
with open('data/stsbenchmark/sts-test.csv') as f:
    reader = csv.reader(f, delimiter='\n')
    test_list = []
    for line in reader:
        test = line[0].split('\t')
        labels.append(float(test[4]))
        refs.append(test[5])
        hyps.append(test[6])


# calculate correlation
print('Get scores on STSb-test. Processing ..')
similarity_scores_representation = []
# similarity_scores_embeddings = []
for cnt, (ref, hyp) in enumerate(zip(refs, hyps)):
    if (cnt+1) % 200 == 0:
        print(cnt+1, end=', ')
    instances = create_testing_instances(ref, tokenizer, 
                                         FLAGS.max_seq_length)

    batch_input_ids = []
    batch_input_mask = []
    for _instance in instances:
        _input_ids = [word_to_id[_token] for _token in _instance.input_tokens]
        _input_mask = [1] * len(_input_ids)

        batch_input_ids.append(_input_ids)
        batch_input_mask.append(_input_mask)

    feed_dict = {input_ids : batch_input_ids,
                 input_mask : batch_input_mask,
                 }

    [representations_ref, embeddings_ref] = sess.run([input_tensor, input_embeddings], feed_dict=feed_dict)
    
    
    instances = create_testing_instances(hyp, tokenizer, 
                                         FLAGS.max_seq_length)

    batch_input_ids = []
    batch_input_mask = []
    for _instance in instances:
        _input_ids = [word_to_id[_token] for _token in _instance.input_tokens]
        _input_mask = [1] * len(_input_ids)

        batch_input_ids.append(_input_ids)
        batch_input_mask.append(_input_mask)

    feed_dict = {input_ids : batch_input_ids,
                 input_mask : batch_input_mask,
                 }

    [representations_hyp, embeddings_hyp] = sess.run([input_tensor, input_embeddings], feed_dict=feed_dict)

    sentence_representation_mean_ref = np.mean(representations_ref[1:-1], axis=0)
    sentence_representation_mean_hyp = np.mean(representations_hyp[1:-1], axis=0)
    score = cosine_similarity([sentence_representation_mean_ref], [sentence_representation_mean_hyp])
    similarity_scores_representation.append(score[0][0])
    
#     sentence_embeddings_mean_ref = np.mean(embeddings_ref[0][1:-1], axis=0)
#     sentence_embeddings_mean_hyp = np.mean(embeddings_hyp[0][1:-1], axis=0)
#     score = cosine_similarity([sentence_embeddings_mean_ref], [sentence_embeddings_mean_hyp])
#     similarity_scores_embeddings.append(score[0][0])

print('')
print('STSb-test (context):', sp.stats.pearsonr(labels, similarity_scores_representation)[0])
# print('STSb-test (embed)  :', sp.stats.pearsonr(labels, similarity_scores_embeddings)[0])
