%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import numpy as np
import pandas as pd
import pickle
import csv
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import isri
from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score

def readPositiveData(path,txtFileCount):
    readData=[]
    for i in range(txtFileCount):
        data = pd.read_csv(path+str(i)+".txt",
                          encoding="utf-8",quoting=csv.QUOTE_NONE, header=None)
        readData.append([data,1])
    posData=pd.DataFrame(readData,columns=["tweets","sentiment"])
    print(posData.tweets)
    return posData
    
def readnegativeData(path,txtFileCount):
    readData=[]
    for i in range(txtFileCount):
        try:
            data = pd.read_csv(path+str(i)+".txt",
                          encoding="utf-8",quoting=csv.QUOTE_NONE, header=None)
        except:
            print(i)
        readData.append([data,0])
    negData=pd.DataFrame(readData,columns=["tweets","sentiment"])
    print(negData.tweets)
    return negData
    
    
def concatenateData(posData,negData):
    Data=pd.concat([posData,negData])
    Data=Data.sample(frac=1)
    Data
    return Data
    
    
def beautiful(Data):
    Data['tweets'] = Data['tweets'].apply(lambda x: BeautifulSoup(str(x), "lxml").text)
    Data['tweets'] = Data['tweets'].map(lambda x: x.replace('0\n0',''))
    Data_clean = Data.loc[:, ['tweets', 'sentiment']]
#     Data_clean.tweets
#     Data
    return Data_clean
# for i in range(len(Data)) : 
#     BeautifulSoup(str(Data.loc[i, "tweets"]), "lxml").text

def trainTestSplit(Data_clean,test_size=0.2, random_state=1):
    train, test = train_test_split(Data_clean, test_size=test_size, random_state=random_state)
    X_train = train['tweets'].values
    X_test = test['tweets'].values
    y_train = train['sentiment']
    y_test = test['sentiment']
    return X_train, X_test, y_train, y_test
    
 
 def tokenize(text): 
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)

def stem(doc):
    return (isri.stem(w) for w in analyzer(doc))

def vectorize():
    en_stopwords = set(stopwords.words("arabic")) 

    vectorizer = CountVectorizer(
        analyzer = 'word',
        tokenizer = tokenize,
        lowercase = True,
        ngram_range=(1, 1),
        stop_words = en_stopwords)
    return vectorizer
    
    
def train(vectorizer,X_train, X_test, y_train, y_test):
    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    np.random.seed(1)

    pipeline_svm = make_pipeline(vectorizer, 
                                SVC(probability=True, class_weight="balanced"))

    grid_svm = GridSearchCV(pipeline_svm,
                        {'svc__C': [3.5],'svc__kernel':["rbf"]}, 
                        cv = kfolds,
                        scoring="roc_auc",
                        verbose=1,   
                        n_jobs=-1) 

    grid_svm.fit(X_train, y_train)
    grid_svm.score(X_test, y_test)
    return grid_svm
    
    
def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)        

    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    result = {'auc': auc, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
    return result
    
    
def get_roc_curve(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_proba)
    return fpr, tpr
    
    
fpr, tpr = roc_svm
plt.figure(figsize=(14,8))
plt.plot(fpr, tpr, color="red")
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc curve')
plt.show()


def testYourSelf(query):
    return grid_svm.predict([query])
    
    
def main(path1,path2):
    posPath=path1
    posData=readPositiveData(posPath,29849)
    negPath=path2
    negData=readPositiveData(negPath,28902)
    s=negData.tweets
    print(s[5][0][0])
    Data=concatenateData(posData,negData)
    Data_clean=beautiful(Data)
    X_train, X_test, y_train, y_test= trainTestSplit(Data_clean,test_size=0.2, random_state=1)
    vectorizer = vectorize()
    grid_svm= train(vectorizer,X_train, X_test, y_train, y_test)
    grid_svm.best_params_
    grid_svm.best_score_
    print(report_results(grid_svm.best_estimator_, X_test, y_test))
    roc_svm = get_roc_curve(grid_svm.best_estimator_, X_test, y_test)
    fpr, tpr = roc_svm
    plt.figure(figsize=(14,8))
    plt.plot(fpr, tpr, color="red")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc curve')
    plt.show()
    print(testYourSelf("الحمد لله دائما وابدا"))
    print(testYourSelf("شئ لا يصدق ,, حقا رائع"))
    print(testYourSelf("الجو حر اوي النهاردة"))
    print(testYourSelf("يارب الاهلي يكسب النهاردة"))
    print(testYourSelf("كوتشي جديد وجميل"))
    print(testYourSelf("انا حزين جدا لان حذائي قد قطع"))
    

main("/home/ahmed/intern work/sentiment analysis/Arabic sentiment/164704_1375839_bundle_archive/arabic_tweets/pos/",
    "/home/ahmed/intern work/sentiment analysis/Arabic sentiment/164704_1375839_bundle_archive/arabic_tweets/neg/")


#load model part
    
def loadModel(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
    
    
def predict(model,query):
    return model.predict([query])
    
    
model= loadModel('arabic sentiment analysis.sav')

print(predict(model,"اخي مريض جدا"))
