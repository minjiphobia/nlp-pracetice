#!/usr/bin/python3

import sys

path2dict = 'datasets/training_vocab.txt'
path2testset = 'datasets/test.txt'
path2fmm = 'datasets/result_fmm.txt'
path2rmm = 'datasets/result_rmm.txt'
path2bimm = 'datasets/result_bimm.txt'

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
            if word in user_dict or lenw == 1:
                result.append(word)
                idx += lenw 
                break
            else:
                lenw -= 1

    if not forward:
        result.reverse()

    return result

def get_outfile(mode): # fmm rmm bimm
    # generate user dictionary
    with open(path2dict, 'r') as f:
        user_dict = set(f.read().splitlines())

    # fetch test set
    with open(path2testset, 'r') as f:
        lines = f.read().splitlines()
    
    if mode == 'fmm':
        outfile = open(path2fmm, 'w')
    elif mode == 'rmm':
        outfile = open(path2rmm, 'w')
    else: # mode == 'bimm'
        outfile = open(path2bimm, 'w')

    for line in lines:
        #print(f'parsing {line}')
        if mode == 'fmm':
            result = mm_parse(user_dict, line, True)
        elif mode == 'rmm':
            result = mm_parse(user_dict, line, False)
        else: # mode == 'bimm'
            fmm = mm_parse(user_dict, line, True)
            rmm = mm_parse(user_dict, line, False)
            lenf = len(fmm)
            lenr = len(rmm)

            # choose between fmm and rmm
            if lenf < lenr:
                result = fmm
            elif lenf > lenr:
                result = rmm 
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
                    result = fmm
                else:
                    result = rmm

        outfile.write('  '.join(result) + '\n')

    outfile.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('no mode specified. using bimm as default')
        mode = 'bimm' # default
    elif len(sys.argv) > 2:
        print('only one arg is permitted')
        exit()
    else:
        if sys.argv[1] in {'fmm', 'rmm', 'bimm'}:
            mode = sys.argv[1]
        else:
            print('wrong arg')
            exit()
    get_outfile(mode)
