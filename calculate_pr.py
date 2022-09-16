#!/usr/bin/env python
#-*- coding: utf8 -*-

import sys
from sklearn.metrics import precision_score, recall_score, f1_score

def multi_F1_score(input_path, output_path, labels):
    output = open(output_path, "a+")
    true_labels = []
    pred_labels = []
    labels = labels.split(',')

    for line in open(input_path):
        sp = line.rstrip('\n').split('\t')
        query, label, pred_label = sp[0:3]
        true_labels.append(label)
        pred_labels.append(pred_label)

    p = precision_score(true_labels, pred_labels, labels=labels, average='micro')
    r = recall_score(true_labels, pred_labels, labels=labels, average='micro')
    f1 = f1_score(true_labels, pred_labels, labels=labels, average='micro')
    output.write("micro:\t" + str(p) + '\t' + str(r) + '\t' + str(f1) + '\n')
    print ("micro:\t" + str(p) + '\t' + str(r) + '\t' + str(f1))
    p = precision_score(true_labels, pred_labels, labels=labels, average='macro')
    r = recall_score(true_labels, pred_labels, labels=labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, labels=labels, average='macro')
    acc = f1_score(true_labels, pred_labels, average='micro')
    output.write("macro:\t" + str(p) + '\t' + str(r) + '\t' + str(f1) + '\n')
    output.write("acc:\t" + str(acc) + '\n')


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    labels = sys.argv[3]
    multi_F1_score(input_path, output_path, labels)
