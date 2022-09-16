# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Helpers for preparing pre-training data and supplying them to the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow.compat.v1 as tf

import configure_pretraining
from model import tokenization
from util import utils


def get_input_fn(config: configure_pretraining.PretrainingConfig, is_training,
                 num_cpu_threads=4):
    if config.pap_task:
        return multi_task_get_input_fn(config, is_training, num_cpu_threads)
    else:
        return lm_task_get_input_fn(config, is_training, num_cpu_threads)


def lm_task_get_input_fn(config: configure_pretraining.PretrainingConfig, is_training,
                         num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    input_files = []
    for input_pattern in config.pretrain_tfrecords.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "input_ids": tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
            "input_mask": tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
        }
        if config.cws_input:
            name_to_features["cws_length_mask"] = tf.io.FixedLenFeature([config.max_seq_length], tf.int64)

        d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        d = d.repeat()
        d = d.shuffle(buffer_size=len(input_files))

        # `cycle_length` is the number of parallel files that get read.
        cycle_length = min(num_cpu_threads, len(input_files))

        # `sloppy` mode means that the interleaving is not exact. This adds
        # even more randomness to the training pipeline.
        d = d.apply(
            tf.data.experimental.parallel_interleave(
                tf.data.TFRecordDataset,
                sloppy=is_training,
                cycle_length=cycle_length))
        d = d.shuffle(buffer_size=100)

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don"t* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        return d

    return input_fn


def clip(x, low, hi):
    if x < low:
        return low
    if x > hi:
        return hi
    return x


def multi_task_get_input_fn(config: configure_pretraining.PretrainingConfig, is_training,
                            num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def decode_multi_records(record, name_to_features, task_id):
        example = tf.io.parse_single_example(record, name_to_features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t
        r = {
            'task_id': tf.constant(task_id, tf.int32),
            'a1': tf.zeros((config.max_seq_length,), dtype=tf.int32),
            'a2': tf.zeros((config.max_seq_length,), dtype=tf.int32),
            'a3': tf.zeros((config.max_seq_length,), dtype=tf.int32),
            'a4': tf.zeros((config.max_seq_length,), dtype=tf.int32),
            'b1': tf.zeros((config.pap_title_max_length,), dtype=tf.int32),
            'b2': tf.zeros((config.pap_title_max_length,), dtype=tf.int32),
            'b3': tf.zeros((config.pap_attr_name_max_length,), dtype=tf.int32),
            'b4': tf.zeros((config.pap_attr_name_max_length,), dtype=tf.int32),
            'b5': tf.zeros((config.pap_attr_value_max_words,), dtype=tf.int32),
        }
        if task_id == 0:
            r['a1'] = example['input_ids']
            r['a2'] = example['input_mask']
            r['a3'] = example['segment_ids']
            if config.cws_input:
                r['a4'] = example['cws_length_mask']
        elif task_id == 1:
            r['b1'] = example['t_input_ids']
            r['b2'] = example['t_input_mask']
            r['b3'] = example['n_input_ids']
            r['b4'] = example['n_input_mask']
            r['b5'] = example['v_word_ids']
        else:
            raise Exception("task_id must be 0 or 1.")
        return r

    def read(input_files, task_id, name_to_features, batch_size):
        d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        d = d.repeat()
        d = d.shuffle(buffer_size=len(input_files))

        # `cycle_length` is the number of parallel files that get read.
        cycle_length = min(num_cpu_threads, len(input_files))

        # `sloppy` mode means that the interleaving is not exact. This adds
        # even more randomness to the training pipeline.
        d = d.apply(
            tf.data.experimental.parallel_interleave(
                tf.data.TFRecordDataset,
                sloppy=is_training,
                cycle_length=cycle_length))
        d = d.shuffle(buffer_size=100)

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don"t* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: decode_multi_records(record, name_to_features, task_id),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        return d

    ml_input_files = []
    for input_pattern in config.pretrain_tfrecords.split(","):
        ml_input_files.extend(tf.io.gfile.glob(input_pattern))

    pap_input_files = []
    for input_pattern in config.pap_pretrain_tfrecords.split(","):
        pap_input_files.extend(tf.io.gfile.glob(input_pattern))

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        name_to_features = {
            "input_ids": tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
            "input_mask": tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
        }
        if config.cws_input:
            name_to_features["cws_length_mask"] = tf.io.FixedLenFeature([config.max_seq_length], tf.int64)

        pap_name_to_features = {
            "t_input_ids": tf.io.FixedLenFeature([config.pap_title_max_length], tf.int64),
            "t_input_mask": tf.io.FixedLenFeature([config.pap_title_max_length], tf.int64),
            "n_input_ids": tf.io.FixedLenFeature([config.pap_attr_name_max_length], tf.int64),
            "n_input_mask": tf.io.FixedLenFeature([config.pap_attr_name_max_length], tf.int64),
            "v_word_ids": tf.io.FixedLenFeature([config.pap_attr_value_max_words], tf.int64),
        }
        rand_index = tf.random_uniform([int(clip(config.num_train_steps, 1E3, 1E6))])
        rand_index = tf.cast(rand_index < config.pap_dataset_prob, tf.int64)
        rand_index = tf.data.Dataset.from_tensor_slices(rand_index)
        rand_index = rand_index.repeat()
        d = tf.data.experimental.choose_from_datasets(
            [read(ml_input_files, 0, name_to_features, batch_size),
             read(pap_input_files, 1, pap_name_to_features, batch_size)],
            rand_index)
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    return example


# model inputs - it's a bit nicer to use a namedtuple rather than keep the
# features as a dict
Inputs = collections.namedtuple(
    "Inputs", ["input_ids", "input_mask", "segment_ids", "cws_length_mask",
               "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"])

PAPInputs = collections.namedtuple(
    "PAPInputs", ["t_input_ids", "t_input_mask", "t_segment_ids",
                  "n_input_ids", "n_input_mask", "n_segment_ids",
                  "v_word_ids"])

BertInputs = collections.namedtuple(
    "BertInputs", ["input_ids", "input_mask", "segment_ids"])


def features_to_inputs(features, task_id=0):
    if task_id == 0:
        result = Inputs(
            input_ids=features["input_ids"],
            input_mask=features["input_mask"],
            segment_ids=features["segment_ids"],
            cws_length_mask=features.get("cws_length_mask"),
            masked_lm_positions=features.get("masked_lm_positions"),
            masked_lm_ids=features.get("masked_lm_ids"),
            masked_lm_weights=features.get("masked_lm_weights"))
    elif task_id == 1:
        result = PAPInputs(
            t_input_ids=features["t_input_ids"],
            t_input_mask=features["t_input_mask"],
            t_segment_ids=tf.zeros_like(features["t_input_ids"]),
            n_input_ids=features["n_input_ids"],
            n_input_mask=features["n_input_mask"],
            n_segment_ids=tf.zeros_like(features["n_input_ids"]),
            v_word_ids=features["v_word_ids"])
    else:
        raise RuntimeError("only support task 0 and 1, but got {}".format(task_id))
    return result


def rename_keys(d, task_id):
    if task_id == 0:
        e = {
            "input_ids": d["a1"],
            "input_mask": d["a2"],
            "segment_ids": d["a3"],
            "cws_length_mask": d["a4"]
        }
    elif task_id == 1:
        e = {
            "t_input_ids": d["b1"],
            "t_input_mask": d["b2"],
            "n_input_ids": d["b3"],
            "n_input_mask": d["b4"],
            "v_word_ids": d["b5"]
        }
    else:
        raise RuntimeError("only support task 0 and 1, but got {}".format(task_id))
    return e


def get_updated_inputs(inputs, **kwargs):
    return inputs._replace(**kwargs)
    # features = inputs._asdict()
    # for k, v in kwargs.items():
    #     features[k] = v
    # return features_to_inputs(features)


ENDC = "\033[0m"
COLORS = ["\033[" + str(n) + "m" for n in list(range(91, 97)) + [90]]
RED = COLORS[0]
BLUE = COLORS[3]
CYAN = COLORS[5]
GREEN = COLORS[1]


def print_tokens(inputs: Inputs, inv_vocab, updates_mask=None):
    """Pretty-print model inputs."""
    pos_to_tokid = {}
    for tokid, pos, weight in zip(
            inputs.masked_lm_ids[0], inputs.masked_lm_positions[0],
            inputs.masked_lm_weights[0]):
        if weight == 0:
            pass
        else:
            pos_to_tokid[pos] = tokid

    text = ""
    provided_update_mask = (updates_mask is not None)
    if not provided_update_mask:
        updates_mask = np.zeros_like(inputs.input_ids)
    for pos, (tokid, um) in enumerate(
            zip(inputs.input_ids[0], updates_mask[0])):
        token = inv_vocab[tokid]
        if token == "[PAD]":
            break
        if pos in pos_to_tokid:
            token = RED + token + " (" + inv_vocab[pos_to_tokid[pos]] + ")" + ENDC
            if provided_update_mask:
                assert um == 1
        else:
            if provided_update_mask:
                assert um == 0
        text += token + " "
    utils.log(tokenization.printable_text(text))
