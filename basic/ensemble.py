import argparse
import functools
import gzip
import json
import pickle
from collections import defaultdict
from operator import mul

from tqdm import tqdm
from squad.utils import get_phrase, get_best_span, get_best_span_Snigdha, get_all_spans_Snigdha


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    parser.add_argument('-o', '--out', default='ensemble.json')
    parser.add_argument("--data_path", default="data/squad/data_test.json")
    parser.add_argument("--shared_path", default="data/squad/shared_test.json")
    args = parser.parse_args()
    return args


def ensemble(args):
    max_span_len = 30
    no_of_top_candidates = 200
    e_list = []
    for path in tqdm(args.paths):
        with gzip.open(path, 'r') as fh:
            e = pickle.load(fh)
            e_list.append(e)

    with open(args.data_path, 'r') as fh:
        data = json.load(fh)

    with open(args.shared_path, 'r') as fh:
        shared = json.load(fh)

    out = {}
    for idx, (id_, rx) in tqdm(enumerate(zip(data['ids'], data['*x'])), total=len(e['yp'])):
        if idx >= len(e['yp']):
            # for debugging purpose
            break
        context = shared['p'][rx[0]][rx[1]]
        wordss = shared['x'][rx[0]][rx[1]]
        yp_list = [e['yp'][idx] for e in e_list]
        yp2_list = [e['yp2'][idx] for e in e_list]
        #(scores_dict, span_dict) = ensemble3_Snigdha1(context, wordss, yp_list, yp2_listi, no_of_top_candidates) # Use when you need top N candidates
        (scores_dict, span_dict) = ensemble3_getAllSpans_Snigdha(context, wordss, yp_list, yp2_list, max_span_len) # Use when you need all candidates of length<max_span_len. See line 
        d = {}
        for phrase, score in scores_dict.items():
            d[phrase] = (score, span_dict[phrase])
        #print("Final output="+id_+"\t")# Snigdha: added by Snigdha
        #print(d)# Snigdha: added by Snigdha
        out[id_] = d

    with open(args.out, 'w') as fh:
        json.dump(out, fh)


def ensemble1(context, wordss, y1_list, y2_list):
    """

    :param context: Original context
    :param wordss: tokenized words (nested 2D list)
    :param y1_list: list of start index probs (each element corresponds to probs form single model)
    :param y2_list: list of stop index probs
    :return:
    """
    sum_y1 = combine_y_list(y1_list)
    sum_y2 = combine_y_list(y2_list)
    span, score = get_best_span(sum_y1, sum_y2)
    return get_phrase(context, wordss, span)


def ensemble2(context, wordss, y1_list, y2_list):
    start_dict = defaultdict(float)
    stop_dict = defaultdict(float)
    for y1, y2 in zip(y1_list, y2_list):
        span, score = get_best_span(y1, y2)
        start_dict[span[0]] += y1[span[0][0]][span[0][1]]
        stop_dict[span[1]] += y2[span[1][0]][span[1][1]]
    start = max(start_dict.items(), key=lambda pair: pair[1])[0]
    stop = max(stop_dict.items(), key=lambda pair: pair[1])[0]
    best_span = (start, stop)
    return get_phrase(context, wordss, best_span)


def ensemble3(context, wordss, y1_list, y2_list):
    d = defaultdict(float)
    for y1, y2 in zip(y1_list, y2_list):
        span, score = get_best_span(y1, y2)
        phrase = get_phrase(context, wordss, span)
        d[phrase] += score
    return max(d.items(), key=lambda pair: pair[1])[0]

def ensemble3_getAllSpans_Snigdha(context, wordss, y1_list, y2_list, max_span_len):
    d = defaultdict(float)
    d_span = {}#Snigdha
    for y1, y2 in zip(y1_list, y2_list):
        d_span_scores = get_all_spans_Snigdha(y1, y2, context, wordss, max_span_len)
        for span in d_span_scores:
            phrase = get_phrase(context, wordss, span)
            d[phrase] += d_span_scores[span]
            d_span[phrase] = span
    return (d,d_span)

def ensemble3_Snigdha(context, wordss, y1_list, y2_list):
    d = defaultdict(float)
    d_span = {}#Snigdha
    outputted = list()#Snigdha
    for i in range(0,no_of_top_candidates): # Snigdha: Change here to determine how many candidates you want it to output
        for y1, y2 in zip(y1_list, y2_list):
            span, score = get_best_span_Snigdha(y1, y2, outputted)
            phrase = get_phrase(context, wordss, span)
            d[phrase] += score
            d_span[phrase] = span
            outputted.append(span)#Snigdha
    return (d,d_span)


def combine_y_list(y_list, op='*'):
    if op == '+':
        func = sum
    elif op == '*':
        def func(l): return functools.reduce(mul, l)
    else:
        func = op
    return [[func(yij_list) for yij_list in zip(*yi_list)] for yi_list in zip(*y_list)]


def main():
    args = get_args()
    ensemble(args)

if __name__ == "__main__":
    main()


