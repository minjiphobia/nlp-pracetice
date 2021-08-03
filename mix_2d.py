#!/usr/bin/python3

import re

path2trainset = 'datasets/training.txt'
path2testset = 'datasets/test.txt'
path2mix = 'datasets/result_mix_2d.txt'
path2dict = 'datasets/training_vocab.txt'

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

def gen_outfile(seg, infileName = path2testset, outfileName = path2mix):
    infile = open(infileName, 'r')
    outfile = open(outfileName, 'w')
    with open(path2dict, 'r') as f:
        user_dict = set(f.read().splitlines())
    print(user_dict)
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
                    #(mmlst, remaining, forward) = bimm(user_dict, line[ss:s]) 
                    (fmmlst, fremaining) = mm_parse(user_dict, line[ss:s], True)
                    toadd = 0
                    for fmmword in fmmlst:
                        toadd += len(fmmword)
                        outfile.write(fmmword + '  ')
                    if fremaining:
                        (rmmlst, rremaining) = mm_parse(user_dict, line[ss+toadd:s], False)
                        if rremaining:
                            print(f'predicting {rremaining} [{ss+toadd}:{ss+toadd+len(rremaining)}]')
                            pred = seg.calc(rremaining)
                            for i in range(ss+toadd, ss+toadd+len(rremaining)):
                                print(f'parsing i={i}')
                                outfile.write(line[i])
                                if pred[i-ss-toadd] == 'S' or pred[i-ss-toadd] == 'E':
                                    outfile.write('  ')
                        for rmmword in rmmlst:
                            outfile.write(rmmword + '  ')

                outfile.write(searchObj.group() + '  ') # write searchObj
                ss += e
            else:
                #(mmlst, remaining, forward) = bimm(user_dict, line[ss:len(line)]) 
                (fmmlst, fremaining) = mm_parse(user_dict, line[ss:len(line)], True)
                toadd = 0
                for fmmword in fmmlst:
                    toadd += len(fmmword)
                    outfile.write(fmmword + '  ')
                if fremaining:
                    (rmmlst, rremaining) = mm_parse(user_dict, line[ss+toadd:len(line)], False)
                    if rremaining:
                        print(f'predicting {rremaining} [{ss+toadd}:{ss+toadd+len(rremaining)}]')
                        pred = seg.calc(rremaining)
                        for i in range(ss+toadd, ss+toadd+len(rremaining)):
                            print(f'parsing i={i}')
                            outfile.write(line[i])
                            if pred[i-ss-toadd] == 'S' or pred[i-ss-toadd] == 'E':
                                outfile.write('  ')
                    for rmmword in rmmlst:
                        outfile.write(rmmword + '  ')
                break
        outfile.write('\n')
    infile.close()
    outfile.close()

def bimm(user_dict, line):
    fmm, frem = mm_parse(user_dict, line, True)
    rmm, rrem = mm_parse(user_dict, line, False)
    lenf = len(fmm)
    lenr = len(rmm)

    # choose between fmm and rmm
    if lenf < lenr:
        return fmm, frem, True
    elif lenf > lenr:
        return rmm, rrem, False
    else:
        countf = 0
        for w in fmm:
            if len(w) == 1:
                countf += 1
        countr = 0
        for w in rmm:
            if len(w) == 1:
                countr += 1
        if countf < countr:
            return fmm, frem, True
        else:
            return rmm, rrem, False


def mm_parse(user_dict, sentence, forward):
    max_len = max([len(w) for w in user_dict])
    result = [] 
    idx = 0
    while idx < len(sentence):
        lenw = min(max_len, len(sentence)-idx)  
        while True:
            if forward:
                word = sentence[idx: idx+lenw]
            else: # backward
                word = sentence[len(sentence)-lenw-idx: len(sentence)-idx] # len(sentence)-idx corresponds to idx backwards 
            if word in user_dict:
                result.append(word)
                idx += lenw 
                break
            elif lenw == 1:
                if forward:
                    return result, sentence[idx:]
                else:
                    return result, sentence[: len(sentence)-idx]
            else:
                lenw -= 1
    if not forward:
        result.reverse()
    return result, ''

if __name__ == '__main__':
    seg = HmmSeg()
    seg.train()
    gen_outfile(seg)

