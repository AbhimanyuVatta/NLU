from os import listdir
from os.path import isfile, join
import re
import string
import random
from math import log10
from copy import copy
from collections import Counter, defaultdict, deque
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=int, default=1, help="1/2 for respective Task")
    parser.add_argument("-m", "--model", action="store", default='S3', help="S1/S2/S3/S4 for respective Model")
    opts = parser.parse_args()
    return opts
def processFile(n, file_list, perplex):
    range_wrap = range
    punct = string.punctuation.translate(str.maketrans("", "", ".?!'"))
    all_tokens = ''
    for i in file_list:
        #print('Reading File '+i)
        f = open(i, encoding='ISO-8859-1')
        text = f.read()
        all_tokens = all_tokens+ ' '+text
        f.close()

    all_tokens = all_tokens.lower()
    all_tokens = re.sub('\[(.*)\]\n\n','', all_tokens)
    all_tokens = re.sub("(\/)[^\\ ]+", '', all_tokens)
    all_tokens = re.sub('\nchapter .*\n\n', '', all_tokens)
    all_tokens = re.sub('\nvolume .i*\n\n', '', all_tokens)
    all_tokens = re.sub(' [b-hj-z] ', '', all_tokens)
    begin_token = ""
    for i in range_wrap(n - 1):
        begin_token += ' ' + '<s>'
    stoken = ' ' + etoken
    stoken += begin_token

    for i in punct:
        all_tokens = all_tokens.replace(i, ' ' + i + ' ')


    all_tokens = re.sub("[0-9]", " ", all_tokens)
    all_tokens = all_tokens.replace(".", " ." + stoken)
    all_tokens = re.sub("(\!+\?|\?+\!)[?!]*", " \u203D" + stoken, all_tokens)
    all_tokens = re.sub("\!\!+", " !!" + stoken, all_tokens)
    all_tokens = re.sub("\?\?+", " ??" + stoken, all_tokens)
    all_tokens = re.sub("(?<![?!\s])([?!])", r" \1" + stoken, all_tokens)
    all_tokens = re.sub("(?<=[a-zI])('[a-z][a-z]?)\s", r" \1 ", all_tokens)
    all_tokens = re.sub("\n(?=[^\n])", " ", all_tokens)
    if etoken not in all_tokens:
        all_tokens += stoken if n > 1 else etoken

    punct1 = string.punctuation.translate(str.maketrans("", "", ",</>"))
    for i in punct1:
        all_tokens = all_tokens.replace(i, ' ')

    tokens = all_tokens.split()
    tokens_len = len(tokens)
    if perplex:
        tokens_cnt = Counter(tokens)
        for i in range_wrap(tokens_len):
            if tokens_cnt[tokens[i]] < 6:
                tokens[i] = 'unk'
    return tokens, tokens_len


def dict_creator(freq_dict, wrd, words, total_words):
    if words:
        freq_tmp = freq_dict
        count_tmp = total_words[wrd]

        for word in words[:-3]:
            if not freq_tmp or not freq_tmp[word]:
                freq_tmp[word] = defaultdict(dict)
            if not count_tmp or not count_tmp[word]:
                count_tmp[word] = defaultdict(dict)
            freq_tmp = freq_tmp[word]
            count_tmp = count_tmp[word]

        count_tmp[words[-2]] += 1

        if not freq_tmp or not freq_tmp[words[-2]]:
            freq_tmp[words[-2]] = defaultdict(int)
        freq_tmp[words[-2]][words[-1]] += 1


def n_count_pairs(n, file_list, perplex):
    tokens, train_len = processFile(n, file_list, perplex)
    total_words = {token: defaultdict(int) for token in tokens}
    word_freq_pairs = {token: defaultdict(dict) for token in tokens}

    words_infront = []
    for word in tokens[1:n]:
        words_infront.append(word)

    for i, token in enumerate(tokens[:-n]):
        dict_creator(word_freq_pairs[token], token, words_infront, total_words)
        del words_infront[0]
        words_infront.append(tokens[i + n])

    token = tokens[-n]
    dict_creator(word_freq_pairs[token], token, words_infront, total_words)
    types = len(word_freq_pairs) * k_smooth_val

    return word_freq_pairs, total_words, types


def unsmoothed_ngrams(word_freq_pairs, total_words, n):
    prob_dict = word_freq_pairs
    if n == 2:
        items = prob_dict.items()
        for word, nxt_lvl_dict in items:
            nxt_lvl_items = nxt_lvl_dict.items()
            for word_infront, count in nxt_lvl_items:
                nxt_lvl_dict[word_infront] = count / total_words[word]
        return

    for word in prob_dict:
        unsmoothed_ngrams(prob_dict[word], total_words[word], n - 1)

    return prob_dict


def weightedPickN(words, tmp_dict):
    for word in words:
        tmp_dict = tmp_dict[word]

    s = 0.0
    key = ""
    values = tmp_dict.values()
    r = random.uniform(0, sum(values))
    items = tmp_dict.items()
    for key, weight in items:
        s += weight
        if r < s:
            return key
    return key


def generateSentence(n, ngrams):
    sentence = []
    words = [stoken] * (n - 1)
    word = weightedPickN(words, ngrams)

    while word != etoken:
        if n != 1:
            del words[0]
            words.append(word)
        sentence.append(word)
        word = weightedPickN(words, ngrams)
    return sentence


def laplace_ngrams(word_freq_pairs, total_words, n, V):
    stack = [(word_freq_pairs, total_words, n)]
    prob_dict = word_freq_pairs
    while stack:
        word_freq_pairs, total_words, my_n = stack.pop()
        if my_n == 2:
            items = word_freq_pairs.items()
            for top_word, nxt_lvl_dict in items:
                nxt_lvl_items = nxt_lvl_dict.items()
                for bot_word, cnt in nxt_lvl_items:
                    nxt_lvl_dict[bot_word] = ((cnt + k_smooth_val) / (total_words[top_word] + (V)))
        else:
            my_n -= 1
            for word in word_freq_pairs:
                stack.append((word_freq_pairs[word], total_words[word], my_n))
    return prob_dict


def laplace_perplex_recur(tokens, ngrams, total_words, types, n, unk_count):
    help_dict = ngrams
    if n == 1:
        return log10(help_dict.get(tokens[0], k_smooth_val / (total_words + types)))

    nxt_token = tokens.popleft()
    if nxt_token in help_dict:
        return laplace_perplex_recur(tokens, help_dict[nxt_token], total_words[nxt_token], types, n - 1, unk_count)
    if help_dict['unk']:
        return laplace_perplex_recur(tokens, help_dict['unk'], total_words['unk'], types, n - 1, unk_count)
    return log10(k_smooth_val / (sum(total_words.values()) + types))

def laplace_perplex(tokens, n, ngram, tw, types):
    entropy = 0.0
    unk_cnt = sum(tw['unk'].values())
    num_tokens = len(tokens)
    words = deque(tokens[:n])
    range_wrap = range
    for i in range_wrap(num_tokens - n):
        entropy -= laplace_perplex_recur(copy(words), ngram, tw, types, n, unk_cnt)
        del words[0]
        words.append(tokens[i + n])
    entropy -= laplace_perplex_recur(words, ngram, tw, types, n,unk_cnt)

    return 10 ** (entropy / (num_tokens - (n - 1)))


def compute(model, n, sentence, perplexity):
    print(model)
    train_list = []
    test_list = []
    dir_train = model[0]['Training']
    dir_test = model[1]['Testing']
    for i in dir_train:
        train_list += [i + f for f in listdir(i) if isfile(join(i, f))]
    for j in dir_test:
        test_list += [j + f for f in listdir(j) if isfile(join(j, f))]

    if sentence:
        word_freq_pairs, total_words, types = n_count_pairs(n, train_list, perplexity)

        ngrams = unsmoothed_ngrams(word_freq_pairs, total_words, n)

        sentence = generateSentence(n,ngrams)
        while len(sentence)<=10:
            sentence = generateSentence(n,ngrams)
        print(' '.join(sentence[0:10]))

    if perplexity:
        word_freq_pairs, total_words, types = n_count_pairs(n,train_list, perplexity)

        ngrams = laplace_ngrams(word_freq_pairs, total_words, n, types)

        test_t, test_len  = processFile(n, test_list, 0)

        perplexity = laplace_perplex(test_t, n, ngrams, total_words, types)

        print("Perplexity: " + str(perplexity) + '\n\n')


stoken = "<s>"
etoken = "</s>"
k_smooth_val = .004
n=3
sentence = 1
perplexity = 0
options = parse_args()

corpus = {'guttenberg':['model/guttenberg/Training/','model/guttenberg/Testing/'],
         'brown':['model/brown/Training/','model/brown/Testing/']}

model = {'S1':[{'Training':[corpus['guttenberg'][0]]},{'Testing':[corpus['guttenberg'][1]]}],
         'S2':[{'Training':[corpus['brown'][0]]},{'Testing':[corpus['brown'][1]]}],
         'S3':[{'Training':[corpus['guttenberg'][0],corpus['brown'][0]]},{'Testing':[corpus['guttenberg'][0]]}],
         'S4':[{'Training':[corpus['guttenberg'][0],corpus['brown'][0]]},{'Testing':[corpus['brown'][0]]}]}

if options.task == 1:
    compute(model['S1'], n, 0, 1)
    compute(model['S2'], n, 0, 1)
    compute(model['S3'], n, 0, 1)
    compute(model['S4'], n, 0, 1)

if options.task == 2:
    compute(model[options.model], n, 1, 0)