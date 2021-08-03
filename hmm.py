#!/usr/bin/python3

import re

path2trainset = 'datasets/training.txt'
path2testset = 'datasets/test.txt'
path2hmm = 'datasets/result_hmm_regex.txt'

lst_punctuation =  {
    "，",
    "。",
    "“",
    "”",
    "？",
    "！",
    "：",
    "《",
    "》",
    "、",
    "；",
    "·",
    "‘",
    "’",
    "——",
    "—",
    "／"
    ",",
    ".",
    "?",
    "!",
    "`",
    "~",
    "@",
    "#",
    "$",
    #"%",
    "^",
    "&",
    "*",
    "（",
    "）",
    "-",
    "_",
    "+",
    "=",
    "[",
    "]",
    "{",
    "}",
    '"',
    "'",
    "<",
    ">",
    "●"
    "\n",
    "\r"
}

class HmmSeg:
    def __init__(self):
        self.mat_count_trans = {} # state transition count matrix
        self.mat_count_emit = {} # state->output count matrix
        self.vec_count_init = {} # initial states count vector 

        self.mat_trans = {} # state transition probability matrix
        self.mat_emit = {} # state->output probability matrix
        self.vec_init = {} # initial states probability vector 

        self.states = {'B', 'M', 'E', 'S'}
        self.obsrvs = set()

        for state in self.states:
            self.mat_trans[state] = {}
            self.mat_count_trans[state] = {}
            for state2 in self.states:
                self.mat_trans[state][state2] = 0
                self.mat_count_trans[state][state2] = 0
            self.mat_emit[state] = {}
            self.mat_count_emit[state] = {}
            self.vec_init[state] = 0
            self.vec_count_init[state] = 0

#        print(self.mat_trans)
#        print(self.mat_emit)
#        print(self.vec_init)
    
    def train(self):
        corpus = parse_trainset()
        sum_init = 0
        sum_trans = {}
        sum_emit = {}
        for state in self.states:
            sum_trans[state] = 0
            sum_emit[state] = 0

        for l in corpus:
            sum_init += 1
            for i in range(len(l)):
                sum_emit[l[i][1]] += 1
                self.obsrvs.add(l[i][0])
                if i == 0:
                    self.vec_count_init[l[i][1]] += 1
                else:
                    self.mat_count_trans[l[i-1][1]][l[i][1]] += 1
                    sum_trans[l[i-1][1]] += 1
                    if l[i][0] not in self.mat_count_emit[l[i][1]]:
                        self.mat_count_emit[l[i][1]][l[i][0]] = 1
                    else:
                        self.mat_count_emit[l[i][1]][l[i][0]] += 1

        # add one smoothing
        for state in self.states:
            for ob in self.obsrvs:
                if ob not in self.mat_count_emit[state]:
                    self.mat_count_emit[state][ob] = 1
                else:
                    self.mat_count_emit[state][ob] += 1

        # convert count to probability
        for state in self.vec_count_init:
            self.vec_init[state] = float(self.vec_count_init[state]) / sum_init

        for state in self.mat_count_trans:
            for state2 in self.mat_count_trans[state]:
                self.mat_trans[state][state2] = float(self.mat_count_trans[state][state2]) / sum_trans[state]

        for state in self.mat_count_emit:
            for obs in self.mat_count_emit[state]:
                self.mat_emit[state][obs] = float(self.mat_count_emit[state][obs]) / sum_emit[state]

    def int2char(self, i):
        switcher = {
            0: 'B',
            1: 'M',
            2: 'E',
            3: 'S',
        }
        try:
            return switcher.get(i)
        except KeyError as err:
            print('oops', err)
            return

    # veterbi
    # may KeyError be raised if non-recorded char emits?
    def calc(self, V): 
        dp = [{}] # dp[idx][state]
        S = {} # sequence of states corresponding to dp

        minprob = 1e-6 # there are always words dodging trainset, so mat_emit should return a minimum probability to avoid KeyError
        for state in self.states:
            try:
                dp[0][state] = self.vec_init[state] * self.mat_emit[state].get(V[0], minprob)
            except IndexError:
                print(f'state={state} V={V}')
            S[state] = [state]

        for idx in range(1, len(V)):
            dp.append({})
            new_S = {}
            for state in self.states:
                psTup = [] # probability-state tuple
                for state2 in self.states:
                    if dp[idx-1][state2] == 0:
                        continue
                    prob = dp[idx-1][state2] * self.mat_trans[state2][state] * self.mat_emit[state].get(V[idx], minprob)
                    psTup.append((prob, state2))
                peak = max(psTup) 
                dp[idx][state] = peak[0]
                new_S[state] = S[peak[1]] + [state]
            S = new_S

        p, s = max([(dp[len(V)-1][state], state) for state in self.states])
        return S[s]

def parse_trainset():
    corpus = []
    try:
        with open(path2trainset, 'r', encoding='utf-8') as f:
            for line in f:
                l = []
                for word in line.strip().split():
                    if word in lst_punctuation:
                        continue
                    if len(word) == 1:
                        l.append((word[0], 'S'))
                        continue
                    for i in range(len(word)):
                        if i == 0:
                            l.append((word[i], 'B'))
                        elif i == len(word)-1:
                            l.append((word[i], 'E'))
                        else:
                            l.append((word[i], 'M'))
                corpus.append(l)
    except IOError as err:
        print('oops', err)
        return
    else:
        print('Finished trainset parsing')
    return corpus

# puntuations are dropped in this method
#def gen_outfile(seg, infileName = path2testset, outfileName = path2hmm):
#    outfile = open(outfileName, 'w')
#    with open(infileName, 'r', encoding='utf-8') as infile:
#        for line in infile:
#            sents = re.split('|'.join([c for c in lst_punctuation]), line)
#            for sent in sents:
#                if sent in lst_punctuation:
#                    outfile.write(sent + '  ')
#                pred = seg.calc(sent)
#                for i in range(len(sent)):
#                    outfile.write(sent[i])
#                    if pred[i] == 'S' or pred[i] == 'E':
#                        outfile.write('  ')
#            outfile.write('\n')
#    outfile.close()

def gen_outfile(seg, infileName = path2testset, outfileName = path2hmm):
    infile = open(infileName, 'r')
    outfile = open(outfileName, 'w')
    para = infile.read().splitlines()
    for line in para:
        ss = s = e = 0
        print(line)
        while ss < len(line):
            searchObj = re.search(r'———|——|[，。“”？！：《》、；·‘’（）●／—]|[０-９]|[0-9]{4}年|[0-9]+\.?[0-9]*[百千万亿月日时分秒%]?|[零一二三四五六七八九十○]{4}年|[零一二三四五六七八九十○]+[月日时分秒]', line[ss:])
            if searchObj:
                (s, e) = searchObj.span()
                print(f'find obj s={s+ss} e={e+ss} obj={line[s+ss:e+ss]}')
                if s > 0: # before searchObj
                    s += ss
                    print(f'predicting {line[ss:s]} [{ss}:{s}]')
                    pred = seg.calc(line[ss:s])
                    for i in range(ss, s):
                        print(f'parsing i={i}')
                        outfile.write(line[i])
                        if pred[i-ss] == 'S' or pred[i-ss] == 'E':
                            outfile.write('  ')
                outfile.write(searchObj.group() + '  ') # write searchObj
                ss += e
            else:
                print(f'predicting {line[ss:len(line)]} [{ss}:{len(line)}]')
                pred = seg.calc(line[ss:len(line)])
                for i in range(ss, len(line)):
                    print(f'parsing i={i} ss={ss}')
                    outfile.write(line[i])
                    if pred[i-ss] == 'S' or pred[i-ss] == 'E':
                        outfile.write('  ')
                break
        outfile.write('\n')
    infile.close()
    outfile.close()

if __name__ == '__main__':
#    corpus = parse_trainset() 
#    f = open("output", "w")
#    for l in corpus:
#        f.write('\n'.join('{} {}'.format(x[0],x[1]) for x in l))
#    f.close()
    seg = HmmSeg()
    seg.train()
    gen_outfile(seg)
#    print(seg.calc('共同创造美好的新世纪——二○○一年新年贺词'))
