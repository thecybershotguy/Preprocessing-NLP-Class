import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

class PreProcessing:

    def LoadCsv(self, path, encoding = "latin-1"):
        ''' Loads the CSV file'''
        
        #    Add if the path string is empty or null
        #    Check if the path exists

        self.corpus = pd.read_csv(path,encoding)


    def CheckIfCorpusDefined(self):
        ''' Checks if the corpus is defined , if not insists users to load the corpus '''
        if self.corpus is None:
            print("Load the dataset by using the function name - Loadcsv")
            raise Exception()


    def RemoveBlankColumns(self,columnNames):
        self.CheckIfCorpusDefined()
        for column in columnNames:
            if column in self.corpus:
                self.corpus[column].dropna(inplace=True)


    def __checkIfColumnExists__(self, columnNames):
        ''' Checks if columns exists in corpus''' 
        columnsNotFound = []
        for column in columnNames:
            if column not in self.corpus:
                columnsNotFound.append(column)

        if len(columnsNotFound) != 0:
            raise Exception("Columns not found" + str(columnsNotFound))


    def LowerTexts(self, columns):
        self.CheckIfCorpusDefined()
        self.__checkIfColumnExists__(columns)
        for column in columns:
            self.corpus[column] = [entry.lower() for entry in self.corpus[column]]


    def WordNetLemmatizer(self):
        # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
        self.tag_map = defaultdict(lambda : wn.NOUN)
        self.tag_map['J'] = wn.ADJ
        self.tag_map['V'] = wn.VERB
        self.tag_map['R'] = wn.ADV

        print(self.tag_map)

if __name__ == "__main__":
    print("Yet to implement")
 








        

    

        


