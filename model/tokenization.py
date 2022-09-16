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

"""Tokenization classes, the same as used for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata

import numpy as np
import six
import tensorflow.compat.v1 as tf


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.io.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True, cws_input=False):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

        self.cws_input = cws_input
        self.cws_separator = '/'
        self.cws_separator_escape = '__/'
        self.cws_internal_separator = chr(20008)  # ord('丨')==20008

    def cws_pre(self, text):
        # change chinese word segment separators
        return text.replace(self.cws_separator_escape, chr(1)) \
            .replace(self.cws_separator, self.cws_internal_separator) \
            .replace(chr(1), self.cws_separator)

    def cws_post(self, split_tokens):
        # remove chinese word segment separators
        tokens = []
        length = []
        a = b = 0
        span_all = True
        for i, x in enumerate(split_tokens):
            span_all &= (len(x) == 1 and self.basic_tokenizer._is_chinese_char(ord(x)))
            if x == self.cws_internal_separator:
                b = i
                # [a, b) all chinese characters
                if span_all and b - a > 1:
                    length.append(b - a)
                    length.extend([0] * (b - a - 1))
                else:
                    length.extend([1] * (b - a))
                span_all = True
                a = b + 1
            else:
                tokens.append(x)
        b = len(split_tokens)
        if span_all and b - a > 1:
            length.append(b - a)
            length.extend([0] * (b - a - 1))
        else:
            length.extend([1] * (b - a))

        assert len(tokens) == len(length)
        return tokens, length

    def tokenize(self, text):
        if self.cws_input:
            text = self.cws_pre(text)

        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        if self.cws_input:
            cws_split_tokens, cws_length = self.cws_post(split_tokens)
            return cws_split_tokens, cws_length

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

    Args:
      do_lower_case: Whether to lower case the input.
    """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.

    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]

    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.

    Returns:
      A list of wordpiece tokens.
    """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class Phrase(object):
    """PAP task attribute values"""

    def __init__(self, fn):
        self.a2i = self._load_phrase(fn)

        self.i2a = collections.OrderedDict()
        for a, i in self.a2i.items():
            self.i2a[i] = a

    def convert(self, x, fmt='a2i', seq=False, drop_unknown=True, default_value=None):
        assert fmt in ('a2i', 'i2a'), 'only support a2i and i2a modes'
        c = self.a2i if fmt == 'a2i' else self.i2a
        if seq:
            if drop_unknown:
                output = [c[e] for e in x if e in c]
            else:
                output = [c.get(e, default_value) for e in x]
        else:
            output = c.get(x, default_value)
        return output

    def __len__(self):
        return len(self.a2i)

    def preload(self, tokenizer, max_seq_length, invalid_first_phrase=True, return_value='np'):
        assert len(self.i2a) > 0
        assert max_seq_length > 2
        special_cls = [tokenizer.vocab["[CLS]"]]
        special_sep = [tokenizer.vocab["[SEP]"]]
        all_input_ids = np.zeros(shape=(len(self.i2a), max_seq_length), dtype=np.int32)
        all_input_mask = np.zeros(shape=(len(self.i2a), max_seq_length), dtype=np.int32)
        for i, a in self.i2a.items():
            bert_tokens = tokenizer.tokenize(a)
            bert_tokids = tokenizer.convert_tokens_to_ids(bert_tokens)

            input_ids = special_cls + bert_tokids[0:max_seq_length - 2] + special_sep
            input_mask = [1] * len(input_ids)
            input_ids += [0] * (max_seq_length - len(input_ids))
            input_mask += [0] * (max_seq_length - len(input_mask))
            all_input_ids[i] = input_ids
            all_input_mask[i] = input_mask

        # force the first phrase to be pad
        if invalid_first_phrase:
            all_input_ids[0] = 0
            all_input_mask[0] = 0

        if return_value == 'tf':
            all_input_ids = tf.constant(all_input_ids, dtype=tf.int32)
            all_input_mask = tf.constant(all_input_mask, dtype=tf.int32)

        return all_input_ids, all_input_mask

    @staticmethod
    def _load_phrase(fn):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        index = 0
        with tf.io.gfile.GFile(fn, "r") as reader:
            while True:
                token = convert_to_unicode(reader.readline())
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab


def test_phrase(fn, tokenizer):
    ph = Phrase(fn)
    words = ['流行性感冒', '咳嗽', '痰多', '头晕']

    print('\nWord  Index')
    for x in words:
        y = ph.convert(x)
        print(x, y)

    print('\nIndex Word')
    for i in range(15):
        print(i, ph.convert(i, fmt='i2a'))

    print('\nConvert Word List')
    print(ph.convert(words, seq=True))

    print('\nConvert Index List')
    print(ph.convert(range(15), fmt='i2a', seq=True))

    print('\nPreloading')
    input_ids, input_mask = ph.preload(tokenizer, 20)

    print('input_ids:')
    print(input_ids)
    print(type(input_ids), input_ids.dtype, input_ids.shape)

    print('input_mask:')
    print(input_mask)
    print(type(input_mask), input_mask.dtype, input_mask.shape)


def f1():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pap-phrase-file", required=True)
    parser.add_argument("--vocab-file", required=True)
    args = parser.parse_args()
    print('debug, args:', args)

    tokenizer = FullTokenizer(
        vocab_file=args.vocab_file,
        do_lower_case=True,
        cws_input=False)

    test_phrase(args.pap_phrase_file, tokenizer)


if __name__ == '__main__':
    f1()
