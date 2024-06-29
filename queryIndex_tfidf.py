from time import perf_counter
import re

from tkinter import *
import customtkinter
import CTkListbox
from functools import reduce
from nltk.stem import PorterStemmer
from collections import defaultdict
import copy

porter = PorterStemmer()


class QueryIndex:
    def __init__(self):
        self.stopwordsFile = "stopwords.dat"
        self.titleIndexFile = "titleIndex.dat"
        self.indexFile = "testIndex.dat"
        self.numDocuments = 0
        self.sw = {}
        self.index = {}
        self.titleIndex = {}
        self.ldict = {}

        self.tf = {}  # term frequencies
        self.idf = {}  # inverse document frequencies
        self.readIndex()
        self.getStopwords()
        self.linenumbers()

    def intersectLists(self, lists):
        if len(lists) == 0:
            return []
        # start intersecting from the smaller list
        lists.sort(key=len)
        return list(reduce(lambda x, y: set(x) & set(y), lists))

    def getStopwords(self):
        with open(self.stopwordsFile, 'r', errors="ignore") as f:
            stopwords = [line.rstrip() for line in f]
            self.sw = dict.fromkeys(stopwords)

    def getTerms(self, line):
        line = line.lower()
        line = re.sub(r'[^a-z0-9 ]', ' ', line)  # put spaces instead of non-alphanumeric characters
        line = line.split()
        line = [x for x in line if x not in self.sw]
        line = [porter.stem(word) for word in line]
        return line

    def getPostings(self, terms):
        # all terms in the list are guaranteed to be in the index
        return [self.index[term] for term in terms]

    def getDocsFromPostings(self, postings):
        # no empty list in postings
        return [[x[0] for x in p] for p in postings]

    def linenumbers(self):
        with open("lines.dat", 'r', errors="ignore") as f:
            for line in f:
                line = line.rstrip()
                term, postings = line.split('|')
                postings = postings.split(';')
                postings = [x.split(':') for x in postings]
                postings = [[int(x[0]), list(map(int, x[1].split(',')))] for x in postings if x[0] and x[1]]
                self.ldict[term] = postings

    def readIndex(self):
        # read main index
        with open(self.indexFile, 'r', errors="ignore") as f:
            # first read the number of documents
            self.numDocuments = int(f.readline().rstrip())
            for line in f:
                line = line.rstrip()
                term, postings, tf, idf = line.split('|')  # term='termID', postings='docID1:pos1,pos2;docID2:pos1,pos2'
                postings = postings.split(';')  # postings=['docId1:pos1,pos2','docID2:pos1,pos2']
                postings = [x.split(':') for x in postings]  # postings=[['docId1', 'pos1,pos2'], ['docID2', 'pos1,pos2']]
                postings = [[int(x[0]), list(map(int, x[1].split(',')))] for x in postings]  # final postings list
                self.index[term] = postings
                # read term frequencies
                tf = tf.split(',')
                self.tf[term] = list(map(float, tf))
                # read inverse document frequency
                self.idf[term] = float(idf)

        # read title index
        with open(self.titleIndexFile, 'r', errors="ignore") as f:
            for line in f:
                pageid, title = line.rstrip().split(' ', 1)
                self.titleIndex[int(pageid)] = title

    def dotProduct(self, vec1, vec2):
        if len(vec1) != len(vec2):
            return 0
        return sum([x * y for x, y in zip(vec1, vec2)])

    def rankDocuments(self, terms, docs):
        # term at a time evaluation
        docVectors = defaultdict(lambda: [0] * len(terms))
        queryVector = [0] * len(terms)
        for termIndex, term in enumerate(terms):
            if term not in self.index:
                continue
            queryVector[termIndex] = self.idf[term]
            for docIndex, (doc, postings) in enumerate(self.index[term]):
                if doc in docs:
                    docVectors[doc][termIndex] = self.tf[term][docIndex]

        # calculate the score of each doc
        docScores = [[self.dotProduct(curDocVec, queryVector), doc] for doc, curDocVec in docVectors.items()]
        docScores.sort(reverse=True)
        resultDocs = [x[1] for x in docScores][:10]

        result = []
        for i in resultDocs:
            pageid = i
            for j in terms:
                for id, postings in self.ldict[j]:
                    if id == pageid:
                        result.append((i, postings))

        return result

    def queryType(self, q):
        if '"' in q:
            return 'PQ'
        elif len(q.split()) > 1:
            return 'FTQ'
        else:
            return 'OWQ'

    def owq(self, q):
        '''One Word Query'''
        originalQuery = q
        q = self.getTerms(q)
        if len(q) == 0:
            return ''
        elif len(q) > 1:
            return self.ftq(originalQuery)

        # q contains only 1 term
        term = q[0]
        if term not in self.index:
            return ''
        else:
            postings = self.index[term]
            docs = [x[0] for x in postings]
            return self.rankDocuments(q, docs)

    def ftq(self, q):
        """Free Text Query"""
        q = self.getTerms(q)
        if len(q) == 0:
            return ''
        li = set()
        for term in q:
            try:
                postings = self.index[term]
                docs = [x[0] for x in postings]
                li = li | set(docs)
            except:
                # term not in index
                pass
        li = list(li)
        return self.rankDocuments(q, li)

    def pq(self, q):
        '''Phrase Query'''
        originalQuery = q
        q = self.getTerms(q)
        if len(q) == 0:
            return ''
        elif len(q) == 1:
            return self.owq(originalQuery)

        phraseDocs = self.pqDocs(q)
        return self.rankDocuments(q, phraseDocs)

    def pqDocs(self, q):
        """ here q is not the query, it is the list of terms """
        phraseDocs = []
        length = len(q)
        # first find matching docs
        for term in q:
            if term not in self.index:
                # if a term doesn't appear in the index
                # there can't be any document matching it
                return []

        postings = self.getPostings(q)  # all the terms in q are in the index
        docs = self.getDocsFromPostings(postings)
        # docs are the documents that contain every term in the query
        docs = self.intersectLists(docs)
        # postings are the postings list of the terms in the documents docs only
        for i in range(len(postings)):
            postings[i] = [x for x in postings[i] if x[0] in docs]

        # check whether the term ordering in the docs is like in the phrase query
        # subtract i from the ith terms location in the docs
        postings = copy.deepcopy(postings)  # this is important since we are going to modify the postings list
        for i in range(len(postings)):
            for j in range(len(postings[i])):
                postings[i][j][1] = [x - i for x in postings[i][j][1]]

        # intersect the locations
        result = []
        for i in range(len(postings[0])):
            li = self.intersectLists([x[i][1] for x in postings])
            if li == []:
                continue
            else:
                result.append(postings[0][i][0])  # append the docid to the result

        return result

    def queryIndex(self, q):
        qt = self.queryType(q)
        if qt == 'OWQ':
            return self.owq(q)
        elif qt == 'FTQ':
            return self.ftq(q)
        elif qt == 'PQ':
            return self.pq(q)


# find button callback
def search():
    # get start directory and file ending
    startDir = entryQuery.get()
    t_start = perf_counter()
    result = q.queryIndex(startDir)
    # clear the listbox
    fileList.delete(0, END)

    name = ""
    q1 = q.getTerms(startDir)
    i = 0

    try:
        for doc, line in result:
            pageid = doc
            if name == doc:
                i = i + 1
                pos = q.ldict[q1[i]]
                pos = [b for a, b in pos if a == pageid]
                if len(pos) == 0:
                    while 1:
                        i = i + 1
                        pos = q.ldict[q1[i]]
                        pos = [b for a, b in pos if a == pageid]
                        if len(pos) != 0:
                            break
                fileList.insert(END, "Line Numbers : " + "'" + q1[i] + "'" + " " + str(line))
                t_end = perf_counter()
            else:
                i = 0
                fileList.insert(END, q.titleIndex[doc])
                fileList.insert(END, "\n")
                pos = q.ldict[q1[i]]
                pos = [b for a, b in pos if a == pageid]
                if len(pos) == 0:
                    while 1:
                        i = i + 1
                        pos = q.ldict[q1[i]]
                        pos = [b for a, b in pos if a == pageid]
                        if len(pos) != 0:
                            break
                        else:
                            pos = pos[0]
                else:
                    pos = pos[0]

                with open("Files/"+q.titleIndex[pageid], "r", errors="ignore") as file:
                    content = file.readlines()
                    for line_no in line:
                        fileList.insert(END,   f"[{str(line_no)}] {content[line_no -2]} {content[line_no - 1]} {content[line_no]}")
                fileList.insert(END, "\n")
                t_end = perf_counter()
                #fileList.insert(END, "Line Numbers : " + "'" + q1[i] + "'" + " " + str(line))
                name = doc
    except:
        fileList.insert(END, "Not present")


win = customtkinter.CTk()
win.title("  File Search")
win.geometry("640x480")
win.iconbitmap('Assets\search.ico')
customtkinter.set_appearance_mode("")

frame = customtkinter.CTkFrame(win)
frame.grid(row=0, column=0, padx=10, pady=10)
frame.place(relx=0.2, rely=0.1)

entryQuery = customtkinter.CTkEntry(frame, width=250, corner_radius=25, placeholder_text="Type a query...")
entryQuery.grid(row=0, column=1, padx=(20, 5))

btnSearch = customtkinter.CTkButton(frame, text="Search", width=10, command=search, corner_radius=25)
btnSearch.grid(row=0, column=2, padx=(0, 20), pady=20)


fileList = CTkListbox.CTkListbox(win, width=500, height=200)
fileList.grid(row=2, column=0, padx=20, pady=20)
fileList.place(relx=0.1, rely=0.3)

q = QueryIndex()
win.mainloop()
