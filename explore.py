import csv
import glob
import json
import os
import re
from collections import Counter
from os import path

from functional import seq
from nltk.collocations import ngrams
from nltk.corpus import gazetteers
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk.tag import StanfordNERTagger
from nltk.tokenize import RegexpTokenizer
from nltk.util import everygrams

from restaurantOriginalData import Restaurant
from nationalityToCountry import convert

foodExtractor = re.compile('^(\d+)\t(.+)\t(.+)$')
featureExtractor = re.compile('^(\d+)\t(.+)$')

engStop = stopwords.words('english')
eng = EnglishStemmer()


def cleanFeatures(line):
    return featureExtractor.match(line.rstrip('\n')).groups()


def cleanFood(line):
    return foodExtractor.match(line.rstrip('\n')).groups()


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''


def explore1():
    restaurants = []
    features = {}
    print(glob.glob('data/*.txt'))
    with open('data/features.txt', 'r') as featureIn:
        for line in map(cleanFeatures, featureIn):
            features[line[0]] = line[1]

    for file in glob.glob('data/*.txt'):
        if not file == 'data/features.txt':
            with open(file, 'r') as fin:
                for food in map(cleanFood, fin):
                    restaurants.append(Restaurant(file, food, features))

    fr = seq(restaurants)  # type: seq

    grouped = fr.flat_map(lambda f: list(map(lambda v: (v, f), f.featureVector))) \
        .group_by(lambda fv: fv[0]) \
        .map(lambda item: (item[0], list(map(lambda it: it[1], item[1])))) \
        .to_dict()
    for k, v in grouped.items():
        print(k, v)
        print('==========================\n\n')


def word_grams(words, min=1, max=4):
    s = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
    return s


def append_elements(n_gram):
    for element in range(len(n_gram)):
        phrase = ''
        for sub_element in n_gram[element]:
            phrase += sub_element + ' '
        n_gram[element] = phrase.strip().lower()
    return n_gram


def compare(n_gram1, n_gram2):
    n_gram1 = append_elements(n_gram1)
    n_gram2 = append_elements(n_gram2)
    common = []
    for phrase in n_gram1:
        if phrase in n_gram2:
            common.append(phrase)
    if not common:
        print("Nothing in common between")
        # or you could print a message saying no commonality was found
    else:
        for i in common:
            print(i)


def firstPassGrouping():
    words = []

    stemmed = []
    features = {}
    tokenizer = RegexpTokenizer('\s+', gaps=True)
    clean = re.compile("[()\/']")
    split = re.compile("[/]")
    grams = []
    with open('data/features.txt', 'r') as featureIn:
        for line in map(cleanFeatures, featureIn):
            ws = []
            for w in tokenizer.tokenize(clean.sub(' ', line[1])):
                if w not in engStop:
                    stemmed.append((eng.stem(w).lower(), line[1]))
                    words.append((w.lower(), line[1]))
                    ws.append(w.lower())

            grams.append((list(everygrams(ws, min_len=2, max_len=2)), line[1]))
            features[line[0]] = line[1]


    # cuisine, style, price, atmosphere, and occasion


    noGrams = set(map(lambda x: x[1], filter(lambda x: len(x[0]) == 0, grams)))

    grams = list(filter(lambda x: len(x[0]) > 0, grams))
    groupedw = seq(grams) \
        .flat_map(lambda x: set([(w, x[1]) for w in seq(x[0]).flat_map(lambda y: list(y)).to_list()])) \
        .group_by(lambda w: w[0]) \
        .map(lambda x: (x[0], list(map(lambda y: y[1], x[1])))) \
        .to_dict()

    noGramsId = {}
    for g in noGrams:
        noGramsId[g] = g
    simGrouped = {}
    simular = set()
    for k, v in sorted(groupedw.items(), key=lambda x: x[0]):
        # print(k, v)
        nl = v.copy()
        match = noGramsId.get(k, None)
        for nk in noGramsId.keys():
            if len(nk) > 1:
                if nk in v:
                    nl.append(nk)
                    simular.add(nk)
                for vv in v:
                    if nk in vv:
                        nl.append(nk)
                        simular.add(nk)

        if match is not None:
            nl.append(match)
            simGrouped[k] = list(set(nl))
            simular.add(match)
        else:
            if len(k) > 1:
                simGrouped[k] = v

    noSim = noGrams - simular
    #
    nationalities = gazetteers.words()

    featureNationality = []
    for nosim in noSim:
        didConvert = convert(nosim)
        if didConvert is not None:
            if didConvert in nationalities:
                featureNationality.append(nosim)
        else:
            if nosim in nationalities:
                featureNationality.append(nosim)
            else:
                split = nosim.split('-')
                for sp in split:
                    if sp in nationalities:
                        featureNationality.append(nosim)

    # print("-----------------")


    noSim = noSim - set(featureNationality)
    # occasions = ['monday']
    # # cuisine, style, price, atmosphere, and occasion
    for k, v in sorted(simGrouped.items(), key=lambda x: x[0]):
        # print(k,v)
        if k in nationalities:
            featureNationality.append(k)
            featureNationality.extend(v)
            simGrouped.pop(k)
        didConvert = convert(k)
        if didConvert is not None:
            if didConvert in nationalities:
                simGrouped.pop(k)
                featureNationality.append(k)
                featureNationality.extend(v)

    with open('q1/noSim.json', 'w+') as nsOut:
        nsOut.write(json.dumps(list(noSim), indent=2, sort_keys=True))

    with open('q1/featureNationality.json', 'w+') as nsOut:
        nsOut.write(json.dumps(featureNationality, indent=2, sort_keys=True))

    with open('q1/grouped.json', 'w+') as nsOut:
        nsOut.write(json.dumps(simGrouped, indent=2, sort_keys=True))


def useOtherDataSet():
    knownCuisine = set()
    cuisineCounter = Counter()
    with open('chefmozcuisine.csv', 'r') as cin:
        for row in csv.DictReader(cin):
            cuisineCounter[row['Rcuisine']] += 1
            knownCuisine.add(row['Rcuisine'])

    with open('usercuisine.csv', 'r') as cin:
        for row in csv.DictReader(cin):
            cuisineCounter[row['Rcuisine']] += 1
            knownCuisine.add(row['Rcuisine'])

    q1Cuisine = set()
    with open('data/features.txt', 'r') as featureIn:
        for line in map(cleanFeatures, featureIn):
            if line[1] in knownCuisine:
                q1Cuisine.add(line[1])
                print(line[1])
    #
    with open('q1/q1Labels.json', 'r') as cin:
        labels = json.load(cin)
        labels["cuisine"] = sorted(list(q1Cuisine))
        print(labels)
        with open('q1/q1Labels.json', 'w+') as cout:
            json.dump(labels, cout, indent=2, sort_keys=1)


def cleanUp():
    split = re.compile("[\-\_]")
    knownCuisine = []
    cuisineCounter = Counter()
    with open('chefmozcuisine.csv', 'r') as cin:
        for row in csv.DictReader(cin):
            cuisineCounter[row['Rcuisine']] += 1
            knownCuisine.append(row['Rcuisine'])

            knownCuisine.extend(split.sub(' ', row['Rcuisine']).split(' '))
            knownCuisine.extend(split.sub(' ', row['Rcuisine']))

    with open('usercuisine.csv', 'r') as cin:
        for row in csv.DictReader(cin):
            cuisineCounter[row['Rcuisine']] += 1
            knownCuisine.append(row['Rcuisine'])
            knownCuisine.extend(split.sub(' ', row['Rcuisine']).split(' '))
            knownCuisine.extend(split.sub(' ', row['Rcuisine']))
    labels = []
    with open('q1/q1Labels.json', 'r') as cin:
        ls = json.load(cin)

    labelsSorted = {}

    for k, v in ls.items():
        print(k)
        labelsSorted[k] = sorted(list(set(v)))

    with open('q1/q1Labels.json', 'w+') as cout:
        json.dump(labelsSorted, cout, indent=2, sort_keys=1)

    for v in labelsSorted.values():
        labels.extend(v)
    labels = set(labels)
    grouped = {}
    test = {}
    with open('q1/grouped.json', 'r') as nsOut:
        g = json.load(nsOut)
        for k, v in g.items():
            if len(v) != 0:
                test[k] = v

    for k, v in test.items():
        nv = []
        for vv in v:
            if vv not in labels:
                nv.append(vv)
        if len(nv) != 0:
            grouped[k] = nv
    with open('q1/grouped.json', 'w+') as nsOut:
        nsOut.write(json.dumps(grouped, indent=2, sort_keys=True))
    print(grouped)


if __name__ == '__main__':
    print("hi")

    features = {}
    restaurants = []

    with open('data/features.txt', 'r') as featureIn:
        for line in map(cleanFeatures, featureIn):
            features[line[0]] = line[1]
            print(line[1])

    for file in glob.glob('data/*.txt'):
        if not file == 'data/features.txt':
            with open(file, 'r') as fin:
                for food in map(cleanFood, fin):
                    restaurants.append(Restaurant(file, food, features))

    with open('q1/q1Labels.json', 'r') as cin:
        ls = json.load(cin)

    # for r in restaurants:
    #     print(r)
    #     for feature in r.featureVector:
    #         for k,v in ls.items():
    #             for vv in v:
    #                 if feature == vv:
    #                     print(k,feature)
    with open('allCities.csv', 'w+') as cout:
        cout.write('restaurant,city,features\n')
        for r in restaurants:
            print(r)
            cout.write(r.dump_csv())
            # with open('q1/q1Labels.json', 'w+') as cout:
            #     json.dump(labels, cout, indent=2, sort_keys=1)
            # #

            # southern
            # american
            # asian
            # italian
            #
            # french



            #
            # print(sorted(knownCuisine))
            # print(sorted(cuisineCounter.keys()))
            # print(cuisineCounter)
            # print(len(knownCuisine),len(list(cuisineCounter.keys())))
