# python libs
import os, multiprocessing, linecache
from random import shuffle
import string
printable = set(string.printable)

import numpy as np

# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

'''
doc2vec on flat file of articles 
stored at data/articles in format:
<uid> <article text>
...
<uid> <article text>
'''

class TaggedDocuments(object):
    ids = []

    def __init__(self, source, cleaningfns=None):
        self.source = source

        if cleaningfns: self.cleaningfns = cleaningfns
        else: self.cleaningfns = [lambda x: x]

        # make sure that keys are unique
        with utils.smart_open(self.source) as fin:
            for line in fin:
                # split '<id> <text>' to get id
                idd = line.split(' ', 1)[0]
                self.ids.append(self.gen_id(idd))
        # assert all ids are unique
        assert len(set(self.ids)) == len(self.ids), 'prefixes non-unique'
        self.numdocs = len(self.ids)

        self.indices = xrange(self.numdocs)
    
    def __iter__(self):
        for idx in self.indices:
            lineno = idx + 1
            line = linecache.getline(self.source, lineno)
            #linecache.clearcache() # uncomment if storing file in memory isn't feasible
            yield self.tagged_sentence(line)

    def permute(self):
        '''randomly order how documents are iterated'''
        self.indices = np.random.permutation(self.numdocs)

    def tagged_sentence(self, line):
        # split '<id> <text>'
        idd, text = line.split(' ', 1)
        # clean text
        for fn in self.cleaningfns:
            text = fn(text)
        # split on spaces
        text = utils.to_unicode(text).split()
        return TaggedDocument(words=text, tags=[self.gen_id(idd)])

    def docs_perm(self):
        shuffle(self.docs)
        return self.docs

    def gen_id(self, idd):
        return 'DOC_%s' % idd


if __name__ == "__main__":
    from pprint import pprint
    import argparse

    parser = argparse.ArgumentParser(description='Train doc2vec on a corpus')
    # required
    parser.add_argument('-c','--corpus', required=True, help='path to the corpus file on which to train') 
    parser.add_argument('-o','--output', required=True, help='file path to output trained model')
    # doc2vec training parameters - not required.
    # NOTE: all defaults match gensims, except --sample.
    parser.add_argument('--dm',        type=int,   default=1,    help='defines training algorithm. 0: distributed bag-of-words, 1: distributed-memory.')
    parser.add_argument('--min_count', type=int,   default=5,    help='ignore all words with total frequency lower than this.')
    parser.add_argument('--window',    type=int,   default=8,    help='the maximum distance between the predicted word and context words used for prediction within a document.')
    parser.add_argument('--size',      type=int,   default=300,  help='is the dimensionality of the feature vectors.')
    parser.add_argument('--sample',    type=float, default=1e-5, help='threshold for configuring which higher-frequency words are randomly downsampled. 0 is off.')
    parser.add_argument('--negative',  type=int,   default=0,    help='if > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drawn (usually between 5-20).')
    # convert Namespace to dict
    arg = vars(parser.parse_args())

    # defines model parameters
    params = { k: arg[k] for k in ['dm','min_count','window','size','sample','negative'] }
    params.update({'workers': multiprocessing.cpu_count() })
    pprint('model parameters:')
    pprint(params)
    model = Doc2Vec(**params)

    # strip punctuation and ensure lower case
    strip_punct = lambda text: filter(lambda x: x in printable, text)
    lower = lambda text: text.lower()

    # builds the vocabulary
    print 'instantiating TaggedDocuments'
    articles = TaggedDocuments(arg['corpus'], [strip_punct, lower])
    print 'building vocabulary'
    model.build_vocab(articles)

    # trains the model
    for epoch in range(10):
        print 'epoch:', epoch
        articles.permute()
        model.train(articles)

    modelfile = "{output}-dm{dm}-mincount{min_count}-window{window}-size{size}-sample{sample}-neg{negative}.d2v".format(arg)
    model.save(modelfile)
