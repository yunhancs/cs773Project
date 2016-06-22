import glob
import re
from collections import defaultdict
from os import path
import json

cspace = re.compile('\s')

foodExtractor = re.compile('^(\d+)\t(.+)\t(.+)$')
featureExtractor = re.compile('^(\d+)\t(.+)$')


def cleanFeatures(line):
    return featureExtractor.match(line.rstrip('\n')).groups()


def cleanFood(line):
    return foodExtractor.match(line.rstrip('\n')).groups()


def readFeatures():
    with open('data/features.txt', 'r') as featureIn:
        return list(map(cleanFeatures, featureIn))

def featureMap():
    fs = readFeatures()
    features = {}
    for num,f in fs:
        features[num] = f
    return features

def readResturants():
    restaurants = []
    features = featureMap()
    for file in glob.glob('data/*.txt'):
        if not file == 'data/features.txt':
            with open(file, 'r') as fin:
                for food in map(cleanFood, fin):
                    restaurants.append(Restaurant(file, food, features))
    return restaurants






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




class FeatureMap:
    def __init__(self):
        self.fmap = defaultdict(list)
        self.rmap = defaultdict(list)

    def __setitem__(self, key, value):
        self.fmap[key].append(value)
        self.rmap[value].append(key)

    def __getitem__(self, item):
        fi = self.fmap[item]
        ri = self.rmap[item]
        if len(fi) == 0 and len(ri) == 0:
            return None
        elif len(fi) != 0 and len(ri) == 0:
            return fi if len(fi) > 1 else fi[0]
        else:
            return ri if len(ri) > 1 else ri[0]

    def get(self, k, d=None):
        fi = self.fmap[k]
        ri = self.rmap[k]
        if len(fi) == 0 and len(ri) == 0:
            return d
        elif len(fi) != 0 and len(ri) == 0:
            return fi if len(fi) > 1 else fi[0]
        else:
            return ri if len(ri) > 1 else ri[0]

    def keys(self):
        ks = []
        ks.extend(list(self.fmap.keys()))
        ks.extend(list(self.rmap.keys()))
        return ks

    def items(self):
        itemss = []
        for k, v in self.fmap.items():
            itemss.append((k, v))
        for k, v in self.rmap.items():
            itemss.append((k, v))
        return itemss

class Feature:
    def __init__(self, group):
        self.num = group[0]
        self.feature = group[1]

    def __repr__(self):
        return self.feature


class Restaurant:
    def __init__(self, cityfile, groups, features):
        self.city_file = path.splitext(path.basename(cityfile))[0]
        self.rid = groups[0]
        self.name = groups[1]
        self.rawFeatureVector = groups[2].split(' ')
        self.featureVector = list(map(lambda rf:  features[rf], self.rawFeatureVector))

    def for_json(self):
        return {"name": self.name, "city": self.city_file, "id": self.rid, "labels": self.featureVector}

    def dump_csv(self):
        return '"%s", "%s", "%s"\n' % (self.name, self.city_file, ' '.join(self.featureVector))

    def __repr__(self):
        return "%s:%s" % (self.name, self.city_file)




