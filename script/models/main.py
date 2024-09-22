import os
import sys

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('../pattern_classification'))

import argparse
import evpi_batch2
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
import csv
import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--post-tsv', help='File path to post_tsv produced by Lucene', required=True)
    parser.add_argument('--qa-tsv', help='File path to qa_tsv produced by Lucene', required=True)
    parser.add_argument('--utility-tsv', help='File path to utility_tsv produced by Lucene', required=True)
    parser.add_argument('--train-ids', help='File path to train ids', required=True)
    parser.add_argument('--test-ids', help='File path to test ids', required=True)
    parser.add_argument('--embeddings', help='File path to embeddings', required=True)
    parser.add_argument('--output-ranking-file', help='Output file to save ranking', required=True)
    parser.add_argument('--max-p-len', help='Max post length. Only when batch_size>1', default=300, type=int)
    parser.add_argument('--max-q-len', help='Max question length. Only when batch_size>1', default=100, type=int)
    parser.add_argument('--max-a-len', help='Max answer length. Only when batch_size>1', default=100, type=int)
    parser.add_argument('--n-epochs', help='Number of epochs', default=10, type=int)
    parser.add_argument('--batch-size', help='Batch size', type=int, default=1)
    parser.add_argument('--device', help='Use \"cuda\" or \"cpu\"', choices=['cuda', 'cpu'])
    parser.add_argument('--cuda-no', help='Works only when device=cuda', type=int)
    return parser.parse_args()


def run():
    args = parse_args()

    logging.info('Running with parameters: {0}'.format(args))

    w2v_model = read_w2v_model(args.embeddings)

    if args.device == 'cuda':
        cuda = True
        cuda_no = args.cuda_no
    else:
        cuda = False
        cuda_no = None

    if args.batch_size == 1:
        logging.info('Run evpi with batch_size=1')
        results = evpi2.evpi(cuda, w2v_model, args)
    else:
        logging.info('Run evpi with batch_size>1')
        results = evpi_batch2.evpi(cuda, cuda_no, w2v_model, args)

    save_ranking(args.output_ranking_file, results)


def read_w2v_model(path_in):
    path_out = '/'.join(path_in.split('/')[:-1]) + '/w2v_vectors.txt'
    glove2word2vec(path_in, path_out)
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(path_out)
    if '<PAD>' not in w2v_model or w2v_model.vocab['<PAD>'].index != 0:
        raise ValueError('No <PAD> token in embeddings! Provide embeddings with <PAD> token.')
    return w2v_model


def save_ranking(output_file, results):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['issueid', 'post', 'correct_question', 'correct_a']
        for i in range(1, 11):
            header.append('q' + str(i))
            header.append('a' + str(i))
        writer.writerow(header)

        for postid in results:
            post, values, correct = results[postid]

            record = [postid, post, correct[0], correct[1]]

            values = sorted(values, key=lambda x: x[0], reverse=True)
            for score, question, answer in values:
                record.append(question)
                record.append(answer)

            writer.writerow(record)


if __name__ == '__main__':
    run()
