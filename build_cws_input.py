# Chinese word segmentation of text before building tfrecords


import argparse
import multiprocessing
import os
import shutil
import time

import jieba


def write_buffer(buffer):
    block = ''.join(buffer)
    segments = jieba.cut(block, HMM=False)

    flag = False
    out = []
    for i, word in enumerate(segments):
        word = word.replace('/', '__/')  # escape the separator char
        if flag:
            out.append('/')
        out.append(word)
        flag = (word != '\n')

    return ''.join(out)


def convert(input_file, output_file):
    lines = []
    bs = 0
    bs_limit = 102400

    with open(input_file) as f, open(output_file, 'w') as g:
        for line in f:
            if bs >= bs_limit:
                g.write(write_buffer(lines))
                lines.clear()
                bs = 0
            lines.append(line)  # do not strip the newline character
            bs += len(line)
        if lines:
            g.write(write_buffer(lines))


def write_file(job_id, args):
    fnames = sorted(os.listdir(args.input_dir))
    fnames = [f for (i, f) in enumerate(fnames) if i % args.num_processes == job_id]

    start_time = time.time()
    for file_no, fname in enumerate(fnames):
        if file_no > 0:
            elapsed = time.time() - start_time
            print("job_id {}, processed {}/{} files ({:.1f}%), ELAPSED: {}s, ETA: {}s".format(
                job_id, file_no, len(fnames), 100.0 * file_no / len(fnames), int(elapsed),
                int((len(fnames) - file_no) / (file_no / elapsed))))
        convert(os.path.join(args.input_dir, fname),
                os.path.join(args.output_dir, fname))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True,
                        help="Location of pre-training text files.")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write out the Chinese word segemnts.")
    parser.add_argument("--num-processes", default=1, type=int,
                        help="Parallelize across multiple processes.")
    args = parser.parse_args()
    print('debug', args)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)

    if args.num_processes == 1:
        write_file(0, args)
    else:
        jobs = []
        for i in range(args.num_processes):
            job = multiprocessing.Process(target=write_file, args=(i, args))
            jobs.append(job)
            job.start()
        for job in jobs:
            job.join()


if __name__ == "__main__":
    main()
