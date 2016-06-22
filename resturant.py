import json
import os


def getResturants():
    with open('projectData/restaurants2.json', 'r') as cc:
        rs = json.load(cc)
    restaurants = []
    for r in rs:  # num, label, val
        restaurants.append(Restaurant(r['id'], r['city'], r['name'],
                                      list(map(lambda l: Label(l['num'], l['label'], l['val']), r['labels'])),r['cid']))
    return restaurants

class Label:
    def __init__(self, num, label, val):
        self.num = num # type: str
        self.label = label # type: str
        self.val = val # type: str


    def __eq__(self, other):
        if isinstance(other, str):
            return self.val == other or self.label == other or self.num == other
        elif isinstance(other, tuple):
            return self.label == other[0] and self.val == other[1]
        elif isinstance(other, Label):
            return self.num == other.num and self.label == other.label and self.val == other.val

    def __repr__(self):
        return '{ %s %s }' % (self.label, self.val)

    def __str__(self):
        return self.__repr__()

    def for_csv(self):
        return self.num

    def for_json(self):
        return {'num': self.num, 'label': self.label, 'val': self.val}


class Restaurant:
    def __init__(self, id, city, name, labels,cid=None):
        self.id = id # type: str
        self.city = city # type: str
        self.name = name # type: str
        self.labels = labels # type: list[Label]
        self.cid = None
        if cid is not None:
            self.cid = cid

    def for_np(self):
        labs = []
        for l in self.labels:
            labs.append(l)
        return [self.id,self.city,self.name], labs

    def hasLabelValue(self,l):
        for v in l[1]:
            if self.hasLabel((l[0],v)):
                return True
        return False

    def hasLabel(self,l):
        return l in self.labels

    def for_csv(self):
        labs = [l.for_csv() for l in self.labels]
        return '%s, %s%s'%(self.cid,' '.join(labs),os.linesep)

    def for_json(self):
        if self.cid is not None:
            return {'id': self.id, 'city': self.city, 'name': self.name, 'labels': self.labels,"cid": self.cid}
        else:
            return {'id': self.id, 'city': self.city, 'name': self.name, 'labels': self.labels}

    def for_arff(self):
        labs = [l.for_csv()+'\\n' for l in self.labels]
        return '%s,"%s"\n'%(self.cid,''.join(labs))

    def __repr__(self):
        return "%s:%s %s" % (self.name, self.city, ' '.join(map(lambda l: str(l),self.labels)))

    def __eq__(self, other):
        if isinstance(other, Restaurant):
            return self.id == other.id and self.city == other.city and self.name == other.name and self.labels == \
                                                                                                   other.labels
        else:
            return False

    def __hash__(self):
        return hash(repr(self))