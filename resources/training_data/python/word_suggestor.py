import re
import threading
from collections import Counter

regex = "[a-zA-ZçÇãÃõÕáÁéÉíÍóÓúÚâÂêÊîÎôÔûÛàÀüÜ']+"

data = open('shakespeare.txt').read()

tokens = re.findall(regex, data.lower())
tokens = tokens[:(len(tokens)//12+1)]

tokens_count = Counter(tokens)

unique_tokens = list(tokens_count)

print("Token amount: {}, unique Tokens: {}\nAmount of necessary iterations per word: {}".format(len(tokens), len(unique_tokens), len(tokens)*len(unique_tokens)))
    
def count_bigram(w1, w2):
    count_w1w2 = 0
    for i in range(len(tokens)-1):
        if tokens[i] == w1 and tokens[i+1] == w2:
            count_w1w2 += 1
    return count_w1w2

def count_trigram(w1, w2, w3):
    count_w1w2w3 = 0
    for i in range(len(tokens)-2):
        if tokens[i] == w1 and tokens[i+1] == w2 and tokens[i+2] == w3:
            count_w1w2w3 += 1
    return count_w1w2w3

def prob_bigram(w1, w2):
    count_w1 = tokens_count[w1]
    count_w1w2 = count_bigram(w1, w2)
    try:
        return count_w1w2/count_w1
    except ZeroDivisionError:
        return 0

def prob_trigram(w1, w2, w3):
    count_w1w2w3 = count_trigram(w1, w2, w3)
    count_w1w2 = count_bigram(w1, w2)
    try:
        return count_w1w2w3/count_w1w2
    except ZeroDivisionError:
        return 0

def get_best_words(words, arg, word_list, skip=1):
    shadow1_index = 0
    shadow2_index = 0
    
    highest_prob = 0
    highest_prob_index = 0
    print("Iteration size: " + str(len(word_list)))
    for i in range(0, len(word_list), skip):
        prob = 0
        if arg == 'bigram':
            prob = prob_bigram(words[-1], word_list[i])
        elif arg == 'trigram':
            prob = prob_trigram(words[-2], words[-1], word_list[i])
            
        if prob >= highest_prob:
           shadow1_index = shadow2_index
           shadow2_index = highest_prob_index
           
           highest_prob = prob
           highest_prob_index = i
           if prob > 0:
               print("Highest prob word at the moment: {} with prob: {}".format(word_list[i], prob))

    return [word_list[highest_prob_index], word_list[shadow2_index], word_list[shadow1_index]]

def shannon(words, arg, num):
    word_list = unique_tokens
    arg = arg.lower()
    suggestions = []
    suggestions = get_best_words(words, arg, word_list)
##    word = None
##    for i in range(num):
##        if word is not None:
##            word_list.remove(word)
##            
##        word = get_best_word(words, arg, word_list)
##
##        if word is not None:
##            suggestions.append(word)

    return suggestions
                
        
