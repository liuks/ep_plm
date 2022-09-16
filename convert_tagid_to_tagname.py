# coding=utf-8

import sys
import pickle

def convert(label_ids_path, input_path, output_path):
  plk = open(label_ids_path, 'rb')
  label_ids_dict = pickle.load(plk)
  id_label_dict = {}
  print (type(label_ids_dict))
  print (label_ids_dict)
  for label, id_ in label_ids_dict.items():
      id_label_dict[str(id_)] = label

  output = open(output_path, "w")

  for line in open(input_path):
    key, pred, label = line.rstrip('\n').split('\t')[0:3]
    pred_label = id_label_dict[pred]
    true_label = id_label_dict[label]
    output.write('\t'.join([key, pred_label, true_label]) + '\n')

if __name__ == '__main__':
    label_ids_path = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    convert(label_ids_path, input_path, output_path)
