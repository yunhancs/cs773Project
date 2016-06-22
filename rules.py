import csv
import glob
import json
import os
from os import path
import re
import types
import string
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import defaultdict, Counter

from sklearn.tree import DecisionTreeClassifier

from restaurantOriginalData import readResturants, featureMap
from resturant import Restaurant, Label, getResturants
from functional import seq
from sklearn.datasets import load_iris, load_boston
from sklearn import tree



from nltk.classify.util import names_demo, binary_names_demo_features
#
# def readResturants():
#     restaurants = []
#     features = featureMap()
#     for file in glob.glob('data/*.txt'):
#         if not file == 'data/features.txt':
#             with open(file, 'r') as fin:
#                 for food in map(cleanFood, fin):
#                     restaurants.append(Restaurant(file, food, features))
#     return restaurants

def clean_up():
    restaurants = []
    labels = []
    labelsMapped = {}

    with open('projectData/labels.json', 'r') as cin:
        ls = json.load(cin)  # type: dict[str,list]
        for k, v in ls.items():
            for vv in v:
                labelsMapped[vv] = k

    with open('projectData/uniqueFeatures.json', 'r') as cin:
        uf = json.load(cin)  # type: dict[str,list]

    featuresMapped = {}
    features = featureMap()

    for k, v in uf.copy().items():
        featuresMapped[k] = v
        featuresMapped[v] = k

    # for k, v in featuresMapped.items():
    #     print(k, v)

    restaurants = readResturants()  # type: list

    nr = []
    for r in restaurants:
        nrf = []
        nff = []
        for rf in r.rawFeatureVector:
            raw = featuresMapped[features[rf]]
            nrf.append(raw)
            nff.append(featuresMapped[raw])
            r.mapped.append({"num": raw, "f": featuresMapped[raw]})
        r.rawFeatureVector = nrf
        r.featureVector = nff
        nr.append(r)

    with open('projectData/cleanedCityData.json', 'w+') as cc:
        json.dump(nr, cc, indent=2, sort_keys=True, default=lambda x: x.for_json())


def finalize_resturants_json():
    with open('projectData/cleanedCityData.json', 'r') as cc:
        cleaned = json.load(cc)

    labelsMapped = {}
    unique = {}
    with open('projectData/labels.json', 'r') as cin:
        ls = json.load(cin)  # type: dict[str,list]
        for k, v in ls.items():
            for vv in v:
                labelsMapped[vv] = k

    with open('projectData/uniqueFeatures.json', 'r') as cin:
        uf = json.load(cin)  # type: dict[str,list]
        for k, v in uf.items():
            unique[v] = k

    restaurants = []
    for it in cleaned:
        labels = []
        for m in it['mapped']:
            l = m['f']
            labels.append(Label(m['num'], labelsMapped[l], l))
        restaurants.append(Restaurant(it['id'], it['city'], it['name'], labels))

    with open('projectData/restaurants.json', 'w+') as cc:
        json.dump(restaurants, cc, indent=2, sort_keys=True, default=lambda x: x.for_json())


def final2():
    labelsMapped = {}
    unique = {}
    with open('projectData/labels.json', 'r') as cin:
        ls = json.load(cin)  # type: dict[str,list]
        for k, v in ls.items():
            for vv in v:
                labelsMapped[vv] = k

    reduce = ['Indian', 'Mexican', 'Italian', 'French', 'American', 'Mex']
    newUnique = set()
    newcuisine = set()
    reduceMapping = {}
    for c in ls['cuisine']:
        wasIn = False
        it = None
        for r in reduce:
            if r in c:
                if c != 'French-Japanese':
                    it = 'Mexican' if r == 'Mex' or r == 'Mexican' else r
                    wasIn = True
                    newcuisine.add(it)
        if not wasIn:
            newcuisine.add(c)
    ls['cuisine'] = sorted(list(newcuisine))
    for k, v in ls.items():
        for vv in v:
            newUnique.add(vv)
    out = {}
    for idx, it in enumerate(sorted(newUnique)):
        out[idx] = it

    with open('projectData/reducedFeatures.json', 'w+') as cin:
        json.dump(out, cin, indent=2, sort_keys=True)

    with open('projectData/labels2.json', 'w+') as cin:
        json.dump(ls, cin, indent=2, sort_keys=True)

    bway = {}
    with open('projectData/reducedFeatures.json', 'r') as cin:
        uf = json.load(cin)  # type: dict[str,list]
        for k, v in uf.items():
            bway[v] = k
            bway[k] = v

    rs = getResturants()  # type: list[Restaurant]
    reduce = ['Indian', 'Mexican', 'Italian', 'French', 'American', 'Mex']
    for r in rs:
        newlabels = []
        for l in r.labels:
            if l.label == 'cuisine':
                for r in reduce:
                    if r in l.val:
                        # print('before change', l)
                        it = 'Mexican' if r == 'Mex' or r == 'Mexican' else r
                        l.val = it
            l.num = bway[l.val]
            # print('after change', l)

    for r in rs:
        for l in r.labels:
            if l.label == 'cuisine':
                print(l, l.num)

    with open('projectData/restaurants2.json', 'w+') as cc:
        json.dump(rs, cc, indent=2, sort_keys=True, default=lambda x: x.for_json())


def indexCityId():
    cspace = re.compile('\s')
    cities = set()
    rCity = list()
    rs = getResturants()  # type: list[Restaurant]

    with open('projectData/labels2.json', 'r+') as cin:
        labels = json.load(cin)

    for r in rs:
        rCity.append(r.city + r.id)

    indexCitiesName = {}

    for idx, c in enumerate(rCity):
        indexCitiesName[c] = str(idx)
        indexCitiesName[str(idx)] = c

    for r in rs:
        r.cid = indexCitiesName[r.city + r.id]

    with open('projectData/restaurants2.json', 'w+') as cc:
        json.dump(rs, cc, indent=2, sort_keys=True, default=lambda x: x.for_json())

    with open('projectData/indexCityName.json', 'w+') as cc:
        json.dump(indexCitiesName, cc, indent=2)


def write_arff(fname, features, datas):
    with open('projectData/%s.arff' % fname, 'w+') as arff:
        arff.write('%%\n')
        arff.write('@relation %s\n' % (fname))
        arff.write('@attribute restaurant numeric\n')
        arff.write('@attribute features relational\n')

        arff.write('\t@attribute f numeric\n')
        arff.write('@end features\n')
        arff.write('@data\n')
        for r in datas:
            arff.write(r.for_arff())
        arff.write('%%\n%s' % os.linesep)


if __name__ == '__main__':
    print("hi")

    # with open('projectData/ii.csv','w+') as o:
    #     o.write('restaurant,features%s'%os.linesep)
    #     rs = getResturants() # type: list[Restaurant]
    #
    #     for r in filter(lambda x: x.hasLabelValue(('cuisine',['Indian', 'Mexican', 'Italian', 'French',
    # 'American'])),rs):
    #         o.write(r.for_csv())

    # with open('projectData/reducedFeatures.json','r') as rin:
    #     rfs = json.load(rin)
    #
    # with open('projectData/labels2.json', 'r') as rin:
    #         labs = json.load(rin)
    #
    # print(labs)
    nums = []
    rs = getResturants()  # type: list[Restaurant]

    rests = []
    labs = []


    for r in filter(lambda x: x.hasLabelValue(('cuisine', ['Indian', 'Mexican', 'Italian', 'French', 'American'])), rs):
        re,lab = r.for_np()
        print(r.for_json())
        rests.append(re)
        labs.append(lab)

    np_resturants = np.array(rests)
    # np_labels = np.array(labs)
    # dt = DecisionTreeClassifier()#min_samples_split=20, random_state=99)
    # dt.fit(np_resturants,np_labels)

