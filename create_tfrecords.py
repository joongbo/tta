# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Create TF examples for TTA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import random
import tensorflow as tf
import tokenization

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string("output_file", None,
                    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the TTA model was trained on.")

flags.DEFINE_bool("do_lower_case", True,
                    "The vocabulary file that the TTA model was trained on.")

flags.DEFINE_integer("num_output_split", 100,
                     "Number of output files to split the processed data.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("random_seed", 21625, "Random seed for data generation.")



class TrainingInstance(object):
    """A single training instance (sentence pair)."""

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


def write_instance_to_example_files(instances, word_to_id, max_seq_length, output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = [word_to_id[token] for token in instance.input_tokens]
        target_ids = [word_to_id[token] for token in instance.target_tokens]
        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            target_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(target_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["target_ids"] = create_int_feature(target_ids)
        features["input_mask"] = create_int_feature(input_mask)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info("%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature

def create_training_instances(all_tokens, vocab_words, max_seq_length, rng):
    """Create `TrainingInstance`s from raw text."""
    rng.shuffle(all_tokens)

    instances = []
    print('Process of "create_training_instances"')
    for tokens in all_tokens:
        instances.append(create_instances_from_sentence(tokens, max_seq_length, rng))

    rng.shuffle(instances)
    print('finised')
    
    return instances

def create_instances_from_sentence(tokens, max_seq_length, rng):
    """Creates `TrainingInstance`s for a single sentence."""
    max_num_tokens = max_seq_length - 2
    assert len(tokens) >= 1
    
    if len(tokens) >= max_num_tokens:
        truncate_seq(tokens, max_num_tokens, rng)
    if tokens[0] is not "[SOS]":
        tokens.insert(0, "[SOS]")
    if tokens[-1] is not "[EOS]":
        tokens.append("[EOS]")

    instance = TrainingInstance(tokens)
    
    return instance

def truncate_seq(tokens, max_num_tokens, rng):
    """Truncates a sequence to a maximum sequence length."""
    while True:
        total_length = len(tokens)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def read_all_sentences(input_files):
    all_sentences = []
    for input_file in input_files:
        with open(input_file, "r") as reader:
            for line in reader.readlines():
                line = line.strip()
                if not line:
                    continue
                else:
                    all_sentences.append(line)

    return all_sentences


tf.logging.set_verbosity(tf.logging.INFO)

input_files = []
for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

tf.logging.info("*** Reading from input files ***")
for input_file in input_files:
    tf.logging.info("  %s", input_file)

tf.logging.info("*** Read all sentences from tokenized (space-splitted) text ***")
all_sentences = read_all_sentences(input_files)


tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
word_to_id = tokenizer.vocab
vocab_words = list(word_to_id.keys())
    
    
rng = random.Random(FLAGS.random_seed)
tf.logging.info("*** Split all sentences with random shuffle ***")
rng.shuffle(all_sentences)

cnt_sent = len(all_sentences)
each = cnt_sent // FLAGS.num_output_split
all_sentences_split = []
for i in range(FLAGS.num_output_split):
    s = i * each
    e = (i + 1) * each
    if i + 1 == FLAGS.num_output_split:
        all_sentences_split.append(all_sentences[s:])
    else:
        all_sentences_split.append(all_sentences[s:e])

output_files = FLAGS.output_file.split(",")
try:
    output_name, output_format = output_files[0].split(".")
except:
    raise TypeError("Filename must be have one dot, given: {}".format(output_files[0]))
    
directory = "/".join(output_files[0].split("/")[:-1])
if not os.path.exists(directory):
    os.makedirs(directory)
    
for current_split in range(FLAGS.num_output_split):
    _tail = "-{:05}-of-{:05}".format(current_split, FLAGS.num_output_split)
    
    output_files[0] = ".".join((output_name+_tail, output_format))
    
    if os.path.exists(output_files[0]):
        raise IOError("File must do not exist, given: {}".format(output_files[0]))
        
    tf.logging.info("*** Check output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)
        
    current_sentences = all_sentences_split[current_split]
    all_tokens = []
    for line in current_sentences:
        tokens = line.split()
        for i in range(len(tokens)):
            if tokens[i] not in word_to_id:
                tokens[i] = '[UNK]'
        all_tokens.append(tokens)

    tf.logging.info("*** Create training instances ***")
    instances = create_training_instances(all_tokens, vocab_words,  FLAGS.max_seq_length, rng)

    tf.logging.info("*** Write training instances ***")
    write_instance_to_example_files(instances, word_to_id, FLAGS.max_seq_length, output_files)

    tf.logging.info("*** Finish - {:05}-of-{:05} ***".format(current_split, FLAGS.num_output_split))