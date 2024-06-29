#!/usr/bin/env python3
import sys
import re
import os
from nltk.stem import PorterStemmer
from collections import defaultdict
from array import array
import math

porter = PorterStemmer()


class CreateIndex:

    def __init__(self):
        self.titleIndexFile = "titleIndex.dat"
        self.indexFile = "testIndex.dat"
        self.sw = {}
        self.stopwordsFile = "stopwords.dat"
        self.index = defaultdict(list)  # the inverted index
        self.titleIndex = {}
        self.tf = defaultdict(list)  # term frequencies of terms in documents
        self.ldict = {}  # documents in the same order as in the main index
        self.df = defaultdict(int)  # document frequencies of terms in the files
        self.numDocuments = 0

    def getStopwords(self):
        """get stopwords from the stopwords file"""
        with open(self.stopwordsFile, 'r', errors="ignore") as f:
            stopwords = [line.rstrip() for line in f]
            self.sw = dict.fromkeys(stopwords)

    def getTerms(self, line):
        '''given a stream of text, get the terms from the text'''
        line = line.lower()
        line = re.sub(r'[^a-z0-9 ]', ' ', line)  # put spaces instead of non-alphanumeric characters
        line = line.split()
        line = [x for x in line if x not in self.sw]  # eliminate the stopwords
        line = [porter.stem(word, True) for word in line]
        return line

    def writelines(self):
        with open("lines.dat", 'w', errors="ignore") as f:
            for term, postings in self.ldict.items():
                f.write('%s|' % term)

                for posting in postings:
                    f.write('%s:' % posting[0])
                    f.write(','.join(str(pos) for pos in posting[1]))
                    f.write(';')

                f.write('\n')

    def writeIndexToFile(self):
        '''write the index to the file'''
        # write main inverted index
        with open(self.indexFile, 'w', errors="ignore") as f:
            # first line is the number of documents
            f.write(str(self.numDocuments) + '\n')
            self.numDocuments = float(self.numDocuments)
            for term in self.index.keys():
                postinglist = []
                for p in self.index[term]:
                    docID = p[0]
                    positions = p[1]
                    postinglist.append(':'.join([str(docID), ','.join(map(str, positions))]))
                # print data
                postingData = ';'.join(postinglist)
                tfData = ','.join(map(str, self.tf[term]))
                idfData = '%.4f' % (self.numDocuments / self.df[term])
                f.write('|'.join((term, postingData, tfData, idfData)) + '\n')

        # write title index
        with open(self.titleIndexFile, 'w', errors="ignore") as f:
            for pageid, title in self.titleIndex.items():
                f.write(str(pageid) + ' ' + title + '\n')

    def createIndex(self):
        '''main of the program, creates the index'''
        self.getStopwords()
        for name, filename in enumerate(os.listdir('Files/')):
            if filename.endswith(".txt"):
                #print(filename)
                with open('Files/' + filename, 'r', errors="ignore") as file:
                    text = file.read()

                pageid = name
                linedict = {}
                ct = 1
                with open('Files/' + filename, 'r', errors="ignore") as file:
                    for line in file:
                        x = line.lower()
                        x = re.sub(r'[^a-z0-9 ]', ' ', x)

                        for word in x.split():
                            if word in self.sw:
                                continue
                            y = porter.stem(word, True)

                            if y in linedict.keys():
                                linedict[y][1].append(ct)
                            else:
                                linedict[y] = [pageid, array('I', [ct])]
                        ct += 1

                for term, positions in linedict.items():
                    try:
                        self.ldict[term].append(positions)
                    except:
                        self.ldict[term] = [positions]

                terms = self.getTerms(text)
                self.titleIndex[pageid] = filename
                self.numDocuments += 1
                termdictPage = {}
                for position, term in enumerate(terms):
                    try:
                        termdictPage[term][1].append(position)
                    except:
                        termdictPage[term] = [pageid, array('I', [position])]

                # normalize the document vector
                norm = 0
                for term, posting in termdictPage.items():
                    norm += len(posting[1]) ** 2
                    norm = math.sqrt(norm)

                # calculate the tf and df weights
                for term, posting in termdictPage.items():
                    self.tf[term].append('%.4f' % (len(posting[1]) / norm))
                    self.df[term] += 1

                # merge the current page index with the main index
                for termPage, postingPage in termdictPage.items():
                    self.index[termPage].append(postingPage)
            else:
                continue

        self.writelines()
        self.writeIndexToFile()


if __name__ == "__main__":
    c = CreateIndex()
    c.createIndex()

