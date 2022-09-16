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

"""Writes out text data as tfrecords that ELECTRA can be pre-trained on."""

import argparse
import multiprocessing
import os
import random
import time

import tensorflow.compat.v1 as tf

from model import tokenization
from util import utils


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


class ExampleBuilder(object):
    """Given a stream of input text, creates pretraining examples."""

    def __init__(self, tokenizer, max_length, cws_input=False):
        self._tokenizer = tokenizer
        self._current_sentences = []
        self._current_length = 0
        self._max_length = max_length
        self._target_length = max_length
        self._cws_input = cws_input

    def add_line(self, line):
        """Adds a line of text to the current example being built."""
        line = line.strip().replace("\n", " ")
        if (not line) and self._current_length != 0:  # empty lines separate docs
            return self._create_example()
        if self._cws_input:
            bert_tokens, cws_length = self._tokenizer.tokenize(line)
            bert_tokids = self._tokenizer.convert_tokens_to_ids(bert_tokens)
            assert len(bert_tokids) == len(cws_length)
            self._current_sentences.append(list(zip(bert_tokids, cws_length)))
            self._current_length += len(bert_tokids)
        else:
            bert_tokens = self._tokenizer.tokenize(line)
            bert_tokids = self._tokenizer.convert_tokens_to_ids(bert_tokens)
            self._current_sentences.append(bert_tokids)
            self._current_length += len(bert_tokids)

        if self._current_length >= self._target_length:
            return self._create_example()
        return None

    @staticmethod
    def cws_length_cut_off(sentence):
        # change the ending part if a chinese word is cut in the middle
        # sentence[:, 1] cws length
        if not sentence:
            return

        debug = 0
        if debug:
            print('>>>DEBUG before', sentence)

        last_length = sentence[-1][1]
        # single char
        if last_length == 1:
            return

        # multi chars
        if last_length == 0:
            n = len(sentence)
            i = n - 1
            while i >= 0 and sentence[i][1] == 0:
                i -= 1
            if i >= 0 and sentence[i][1] > n - i:
                for j in range(i, len(sentence)):
                    sentence[j] = (sentence[j][0], 1)
        else:
            #  head of multi chars
            sentence[-1] = (sentence[-1][0], 1)
        if debug:
            print('>>>DEBUG after ', sentence)

    def _create_example(self):
        """Creates a pre-training example from the current list of sentences."""
        # small chance to only have one segment as in classification tasks
        if random.random() < 0.1:
            first_segment_target_length = 100000
        else:
            # -3 due to not yet having [CLS]/[SEP] tokens in the input text
            first_segment_target_length = (self._target_length - 3) // 2

        first_segment = []
        second_segment = []
        for sentence in self._current_sentences:
            # the sentence goes to the first segment if (1) the first segment is
            # empty, (2) the sentence doesn't put the first segment over length or
            # (3) 50% of the time when it does put the first segment over length
            if (len(first_segment) == 0 or
                    len(first_segment) + len(sentence) < first_segment_target_length or
                    (len(second_segment) == 0 and
                     len(first_segment) < first_segment_target_length and
                     random.random() < 0.5)):
                first_segment += sentence
            else:
                second_segment += sentence

        # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
        first_segment = first_segment[:self._max_length - 2]
        second_segment = second_segment[:max(0, self._max_length -
                                             len(first_segment) - 3)]
        if self._cws_input:
            self.cws_length_cut_off(first_segment)
            self.cws_length_cut_off(second_segment)

        # prepare to start building the next example
        self._current_sentences = []
        self._current_length = 0
        # small chance for random-length instead of max_length-length example
        if random.random() < 0.05:
            self._target_length = random.randint(5, self._max_length)
        else:
            self._target_length = self._max_length

        return self._make_tf_example(first_segment, second_segment)

    def _make_tf_example(self, first_segment, second_segment):
        """Converts two "segments" of text into a tf.train.Example."""
        if self._cws_input:
            first_ids = [x[0] for x in first_segment]
            first_cws_length = [x[1] for x in first_segment]
            second_ids = [x[0] for x in second_segment]
            second_cws_length = [x[1] for x in second_segment]
            cws_length_mask = [0] + first_cws_length + [0]
            if second_segment:
                cws_length_mask += second_cws_length + [0]
            cws_length_mask += [0] * (self._max_length - len(cws_length_mask))
        else:
            first_ids, second_ids = first_segment, second_segment
            cws_length_mask = None

        vocab = self._tokenizer.vocab
        input_ids = [vocab["[CLS]"]] + first_ids + [vocab["[SEP]"]]
        segment_ids = [0] * len(input_ids)
        if second_segment:
            input_ids += second_ids + [vocab["[SEP]"]]
            segment_ids += [1] * (len(second_segment) + 1)
        input_mask = [1] * len(input_ids)
        input_ids += [0] * (self._max_length - len(input_ids))
        input_mask += [0] * (self._max_length - len(input_mask))
        segment_ids += [0] * (self._max_length - len(segment_ids))

        if self._cws_input:
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                "input_ids": create_int_feature(input_ids),
                "input_mask": create_int_feature(input_mask),
                "segment_ids": create_int_feature(segment_ids),
                "cws_length_mask": create_int_feature(cws_length_mask)
            }))
        else:
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                "input_ids": create_int_feature(input_ids),
                "input_mask": create_int_feature(input_mask),
                "segment_ids": create_int_feature(segment_ids)
            }))
        debug = 0
        if debug:
            d = {"input_ids": input_ids,
                 "input_words": self._tokenizer.convert_ids_to_tokens(input_ids),
                 "input_mask": input_mask,
                 "segment_ids": segment_ids,
                 "cws_length_mask": cws_length_mask}
            for k in sorted(d.keys()):
                print('DEBUG', k, d[k])
            print('----')
        return tf_example


class ExampleWriter(object):
    """Writes pre-training examples to disk."""

    def __init__(self, job_id, vocab_file, output_dir, max_seq_length,
                 num_jobs, blanks_separate_docs, do_lower_case,
                 num_out_files=1000, cws_input=False):
        self._blanks_separate_docs = blanks_separate_docs
        tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            cws_input=cws_input)
        self._example_builder = ExampleBuilder(tokenizer, max_seq_length, cws_input=cws_input)
        self._writers = []
        for i in range(num_out_files):
            if i % num_jobs == job_id:
                output_fname = os.path.join(
                    output_dir, "pretrain_data.tfrecord-{:}-of-{:}".format(
                        i, num_out_files))
                self._writers.append(tf.io.TFRecordWriter(output_fname))
        self.n_written = 0

    def write_examples(self, input_file):
        """Writes out examples from the provided input file."""
        with tf.io.gfile.GFile(input_file) as f:
            for line in f:
                line = line.strip()
                if line or self._blanks_separate_docs:
                    example = self._example_builder.add_line(line)
                    if example:
                        self._writers[self.n_written % len(self._writers)].write(
                            example.SerializeToString())
                        self.n_written += 1
            example = self._example_builder.add_line("")
            if example:
                self._writers[self.n_written % len(self._writers)].write(
                    example.SerializeToString())
                self.n_written += 1

    def finish(self):
        for writer in self._writers:
            writer.close()


class PAPExampleWriter(object):
    """Writes pre-training examples to disk for PAP task"""

    def __init__(self, job_id, vocab_file, output_dir, num_out_files,
                 num_jobs, do_lower_case,
                 pap_phrase_file, pap_title_max_length,
                 pap_attr_name_max_length, pap_attr_value_max_words):

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.phrase = tokenization.Phrase(pap_phrase_file)
        self.pap_title_max_length = pap_title_max_length
        self.pap_attr_name_max_length = pap_attr_name_max_length
        self.pap_attr_value_max_words = pap_attr_value_max_words

        self.writers = []
        for i in range(num_out_files):
            if i % num_jobs == job_id:
                output_fname = os.path.join(
                    output_dir, "pretrain_data.tfrecord-{:}-of-{:}".format(i, num_out_files))
                self.writers.append(tf.io.TFRecordWriter(output_fname))
        self.n_written = 0

    def write_examples(self, input_file):
        with tf.io.gfile.GFile(input_file) as f:
            for line in f:
                example = self.add_line(line)
                if example:
                    self.writers[self.n_written % len(self.writers)].write(
                        example.SerializeToString())
                    self.n_written += 1

    def finish(self):
        for writer in self.writers:
            writer.close()

    def add_line(self, line, separator='\t'):
        """one line one example."""
        line = line.strip()
        if not line:
            return None
        title, attr_name, attr_value = [x.strip() for x in line.split(separator)]
        if not all((title, attr_name, attr_value)):
            return None

        title_bert_tokens = self.tokenizer.tokenize(title)
        title_bert_tokids = self.tokenizer.convert_tokens_to_ids(title_bert_tokens)

        attr_name_bert_tokens = self.tokenizer.tokenize(attr_name)
        attr_name_bert_tokids = self.tokenizer.convert_tokens_to_ids(attr_name_bert_tokens)

        attr_value_word_ids = self.phrase.convert(attr_value.split('|'), seq=True)

        return self._make_tf_example(title_bert_tokids, attr_name_bert_tokids, attr_value_word_ids)

    @staticmethod
    def _pad(cls_token_id, sep_token_id, tokens, max_len):
        input_ids = [cls_token_id] + tokens[0:max_len - 2] + [sep_token_id]
        eff_len = len(input_ids)
        pad_len = max_len - eff_len
        input_ids += pad_len * [0]
        input_mask = eff_len * [1] + pad_len * [0]
        return input_ids, input_mask

    def _make_tf_example(self, title, attr_name, attr_value):

        cls_token_id = self.tokenizer.vocab["[CLS]"]
        sep_token_id = self.tokenizer.vocab["[SEP]"]

        t_input_ids, t_input_mask = self._pad(
            cls_token_id, sep_token_id, title, self.pap_title_max_length)
        n_input_ids, n_input_mask = self._pad(
            cls_token_id, sep_token_id, attr_name, self.pap_attr_name_max_length)

        v_word_ids = attr_value[0: self.pap_attr_value_max_words]
        v_word_ids += [0] * (self.pap_attr_value_max_words - len(v_word_ids))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            "t_input_ids": create_int_feature(t_input_ids),
            "t_input_mask": create_int_feature(t_input_mask),
            "n_input_ids": create_int_feature(n_input_ids),
            "n_input_mask": create_int_feature(n_input_mask),
            "v_word_ids": create_int_feature(v_word_ids)
        }))
        debug = 0
        if debug:
            d = {"t_input_ids": t_input_ids,
                 "t_input_words": self.tokenizer.convert_ids_to_tokens(t_input_ids),

                 "n_input_ids": n_input_ids,
                 "n_input_words": self.tokenizer.convert_ids_to_tokens(n_input_ids),

                 "v_word_ids": v_word_ids,
                 "v_words": self.phrase.convert(v_word_ids, fmt='i2a', seq=True)}
            for k in sorted(d.keys()):
                print('DEBUG', k, d[k])
            print('\n')
        return tf_example


def write_examples(job_id, args):
    """A single process creating and writing out pre-processed examples."""

    def log(*args):
        msg = " ".join(map(str, args))
        print("Job {}:".format(job_id), msg)

    if args.pap_task:
        log("Creating example writer for PAP task")
        example_writer = PAPExampleWriter(
            job_id=job_id,
            vocab_file=args.vocab_file,
            output_dir=args.output_dir,
            num_jobs=args.num_processes,
            do_lower_case=args.do_lower_case,
            pap_phrase_file=args.pap_phrase_file,
            pap_title_max_length=args.pap_title_max_length,
            pap_attr_name_max_length=args.pap_attr_name_max_length,
            pap_attr_value_max_words=args.pap_attr_value_max_words,
            num_out_files=args.num_out_files)
    else:
        log("Creating example writer for ELECTRA task")
        example_writer = ExampleWriter(
            job_id=job_id,
            vocab_file=args.vocab_file,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            num_jobs=args.num_processes,
            blanks_separate_docs=args.blanks_separate_docs,
            do_lower_case=args.do_lower_case,
            num_out_files=args.num_out_files,
            cws_input=args.cws_input)
    log("Writing tf examples")
    fnames = sorted(tf.io.gfile.listdir(args.corpus_dir))
    fnames = [f for (i, f) in enumerate(fnames)
              if i % args.num_processes == job_id]
    random.shuffle(fnames)
    start_time = time.time()
    for file_no, fname in enumerate(fnames):
        if file_no > 0:
            elapsed = time.time() - start_time
            log("processed {:}/{:} files ({:.1f}%), ELAPSED: {:}s, ETA: {:}s, {:} examples written".format(
                file_no, len(fnames), 100.0 * file_no / len(fnames), int(elapsed),
                int((len(fnames) - file_no) / (file_no / elapsed)),
                example_writer.n_written))
        example_writer.write_examples(os.path.join(args.corpus_dir, fname))
    example_writer.finish()
    log("Done!")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True,
                        help="Location of pre-training text files.")
    parser.add_argument("--vocab-file", required=True,
                        help="Location of vocabulary file.")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write out the tfrecords.")
    parser.add_argument("--max-seq-length", default=128, type=int,
                        help="Number of tokens per example.")

    parser.add_argument("--pap-task", action='store_true',
                        help="pap task")
    parser.add_argument("--pap-phrase-file",
                        help="Location of phrase file for PAP task.")
    parser.add_argument("--pap-title-max-length", default=128, type=int,
                        help="Number of the title tokens.")
    parser.add_argument("--pap-attr-name-max-length", default=10, type=int,
                        help="Number of the product attribute name tokens.")
    parser.add_argument("--pap-attr-value-max-words", default=20, type=int,
                        help="Number of words for each attribute values.")

    parser.add_argument("--num-processes", default=1, type=int,
                        help="Parallelize across multiple processes.")
    parser.add_argument("--blanks-separate-docs", default=True, type=bool,
                        help="Whether blank lines indicate document boundaries.")
    parser.add_argument("--num-out-files", default=1000, type=int,
                        help="Number of output tfrecord files")

    # toggle lower-case
    parser.add_argument("--do-lower-case", dest='do_lower_case',
                        action='store_true', help="Lower case input text.")
    parser.add_argument("--no-lower-case", dest='do_lower_case',
                        action='store_false', help="Don't lower case input text.")
    parser.add_argument("--cws-input", action='store_true',
                        help="whether input Chinese word segmented text.")

    parser.set_defaults(do_lower_case=True)
    args = parser.parse_args()
    print('debug', args)

    if args.pap_task:
        assert args.pap_phrase_file, "--pap_phrase_file is required when running pap_task"
        if args.cws_input:
            print('[WANR]  --cws-input is not required when running pap_task')

    utils.rmkdir(args.output_dir)
    if args.num_processes == 1:
        write_examples(0, args)
    else:
        jobs = []
        for i in range(args.num_processes):
            job = multiprocessing.Process(target=write_examples, args=(i, args))
            jobs.append(job)
            job.start()
        for job in jobs:
            job.join()


if __name__ == "__main__":
    main()
