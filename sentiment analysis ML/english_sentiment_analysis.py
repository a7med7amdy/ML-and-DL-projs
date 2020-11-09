%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
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

def readData(path):
    data = pd.read_csv(path)
    return data
    
def evaluate(value):
    if(value=="negative"):
        return 1
#     elif(value=="neutral"):
#         return 0
#     else:
#         return -1
    else:
        return 0
    
def dataClean(data):
    data_clean = data.copy()
    data_clean = data_clean[data_clean['airline_sentiment_confidence'] > 0.65]
    data_clean['sentiment'] = data_clean['airline_sentiment'].\
        apply(lambda x: evaluate(x))

    data_clean['text_clean'] = data_clean['text'].apply(lambda x: BeautifulSoup(x, "lxml").text)
    data_clean = data_clean.loc[:, ['text_clean', 'sentiment']]
#     print(data_clean)
    return data_clean
    
    
def trainTestSplit(data_clean,test_size=0.2,random_state=1):
    train, test = train_test_split(data_clean, test_size=test_size, random_state=random_state)
    X_train = train['text_clean'].values
    X_test = test['text_clean'].values
    y_train = train['sentiment']
    y_test = test['sentiment']
    return X_train,X_test,y_train,y_test
    
    
def tokenize(text): 
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)

def stem(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

def vectorize():
    en_stopwords = set(stopwords.words("english")) 

    vectorizer = CountVectorizer(
        analyzer = 'word',
        tokenizer = tokenize,
        lowercase = True,
        ngram_range=(1, 1),
        stop_words = en_stopwords)
    return vectorizer
    
def train(vectorizer,X_train, y_train,X_test, y_test):
    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    np.random.seed(1)

    pipeline_svm = make_pipeline(vectorizer, 
                                SVC(probability=True, class_weight="balanced"))

    grid_svm = GridSearchCV(pipeline_svm,
                        {'svc__C': [3.5,4],'svc__kernel':["rbf"]}, 
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
    
    
def testYourSelf(query):
    return grid_svm.predict([query])
    
def main(path):
    data=readData(path)
    data_clean=dataClean(data)
    X_train,X_test,y_train,y_test=trainTestSplit(data_clean,test_size=0.2,random_state=1)
    vectorizer=vectorize()
    grid_svm=train(vectorizer,X_train, y_train,X_test, y_test)
    print(grid_svm.best_params_)
    print(grid_svm.best_score_)
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
    print(testYourSelf("flying with @united is always a great experience"))
    print(testYourSelf("I love @united. haha, it was a joke!"))
    print(testYourSelf("@united very bad experience!"))
    
path="/home/ahmed/intern work/sentiment analysis/17_742210_bundle_archive/Tweets.csv"
main(path)

#load model
def loadModel(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def predict(model,query):
    return model.predict([query])

model= loadModel('english sentiment analysis.sav')

print(predict(model,"it is the best thing I found today"))

print(predict(model,"it is the worst thing I found today"))