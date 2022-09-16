import operator
import sys


def get_rank(score):
    # Ranking from 0
    xs = [(i, x) for i, x in enumerate(score)]
    xs.sort(key=operator.itemgetter(1), reverse=True)
    return [x[0] for x in xs]


def compare_ranks(ra, rb):
    pos = neg = 0
    for a, b in zip(ra, rb):
        if a == b:
            pos += 1
        else:
            neg += 1
    return pos, neg

def calculate_ir(input_path):
    cnt = 0
    scores_list = []
    labels_list = []
    for line in open(input_path):
        _, label_score, pred_score, _ = line.rstrip('\n').split('\t')
        score = float(pred_score)
        label = int(float(label_score) * 4)
        if cnt % 2 == 0:
            scores_list.append([])
            labels_list.append([])
        scores_list[-1].append(score)
        labels_list[-1].append(label)
        cnt += 1

    total_p, total_n = 0, 0
    for scores, labels in zip(scores_list, labels_list):
        #print (scores, labels)
        r1 = get_rank(scores)
        r2 = get_rank(labels)
        p, n = compare_ranks(r1, r2)
        total_p += p
        total_n += n


    print ("score:\t" + str(total_p) + '\t' + str(total_n) + '\t' + str(total_p / total_n))



if __name__ == '__main__':
    input_path = sys.argv[1]
    calculate_ir(input_path)
