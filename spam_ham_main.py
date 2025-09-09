import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

from sklearn.preprocessing import MinMaxScaler

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import wordcloud

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from collections import Counter


def print_validation_report(y_true, y_pred):
    print("Classification Report")
    print(classification_report(y_true, y_pred))
    acc_sc = accuracy_score(y_true, y_pred)
    print("Accuracy : "+ str(acc_sc))
    
    return acc_sc

def plot_confusion_matrix(y_true, y_pred):
    mtx = confusion_matrix(y_true, y_pred)
    #fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  
                cmap="Blues", cbar=False, ax=ax)
    #  square=True,
    plt.ylabel('true label')
    plt.xlabel('predicted label')

data = pd.read_csv("spam.csv",encoding='latin-1')
print(data.head())

data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":"label", "v2":"text"})

print(data.describe())


print(data.groupby("label").describe())

print(data.label.value_counts())

data.label.value_counts().plot.bar()
plt.show()

data['spam'] = data['label'].map( {'spam': 1, 'ham': 0} ).astype(int)
print(data.head(15))

data['length'] = data['text'].apply(len)

data.hist(column='length',by='label',bins=60,figsize=(12,4));
plt.xlim(-40,950)
plt.show()

data_ham  = data[data['spam'] == 0].copy()
data_spam = data[data['spam'] == 1].copy()

def show_wordcloud(data_spam_or_ham, title):
    text = ' '.join(data_spam_or_ham['text'].astype(str).tolist())
    stopwords = set(wordcloud.STOPWORDS)
    
    fig_wordcloud = wordcloud.WordCloud(stopwords=stopwords,background_color='lightgrey',
                    colormap='viridis', width=800, height=600).generate(text)
    
    plt.figure(figsize=(10,7), frameon=True)
    plt.imshow(fig_wordcloud)  
    plt.axis('off')
    plt.title(title, fontsize=20 )
    plt.show()

show_wordcloud(data_ham, "Ham messages") 

show_wordcloud(data_spam, "Spam messages")

print(string.punctuation)


from nltk.corpus import stopwords
print(stopwords.words("english")[100:110])

def remove_punctuation_and_stopwords(sms):
    
    sms_no_punctuation = [ch for ch in sms if ch not in string.punctuation]
    sms_no_punctuation = "".join(sms_no_punctuation).split()
    
    sms_no_punctuation_no_stopwords = \
        [word.lower() for word in sms_no_punctuation if word.lower() not in stopwords.words("english")]
        
    return sms_no_punctuation_no_stopwords

print(data['text'].apply(remove_punctuation_and_stopwords).head())



data_ham.loc[:, 'text'] = data_ham['text'].apply(remove_punctuation_and_stopwords)
words_data_ham = data_ham['text'].tolist()
data_spam.loc[:, 'text'] = data_spam['text'].apply(remove_punctuation_and_stopwords)
words_data_spam = data_spam['text'].tolist()

list_ham_words = []
for sublist in words_data_ham:
    for item in sublist:
        list_ham_words.append(item)

list_spam_words = []
for sublist in words_data_spam:
    for item in sublist:
        list_spam_words.append(item)


c_ham  = Counter(list_ham_words)
c_spam = Counter(list_spam_words)
df_hamwords_top30  = pd.DataFrame(c_ham.most_common(30),  columns=['word', 'count'])
df_spamwords_top30 = pd.DataFrame(c_spam.most_common(30), columns=['word', 'count'])

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='word', y='count', 
            data=df_hamwords_top30, ax=ax)
plt.title("Top 30 Ham words")
plt.xticks(rotation='vertical')
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='word', y='count', 
            data=df_spamwords_top30, ax=ax)
plt.title("Top 30 Spam words")
plt.xticks(rotation='vertical')
plt.show()

fdist_ham  = nltk.FreqDist(list_ham_words)
fdist_spam = nltk.FreqDist(list_spam_words)

df_hamwords_top30_nltk  = pd.DataFrame(fdist_ham.most_common(30),  columns=['word', 'count'])
df_spamwords_top30_nltk = pd.DataFrame(fdist_spam.most_common(30), columns=['word', 'count'])

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='word', y='count', 
            data=df_hamwords_top30_nltk, ax=ax)
plt.title("Top 30 Ham words")
plt.xticks(rotation='vertical')
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='word', y='count', 
            data=df_spamwords_top30_nltk, ax=ax)
plt.title("Top 30 Spam words")
plt.xticks(rotation='vertical')
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer = remove_punctuation_and_stopwords).fit(data['text'])
print(len(bow_transformer.vocabulary_))

sample_spam = data['text'][8]
bow_sample_spam = bow_transformer.transform([sample_spam])
print(sample_spam)
print(bow_sample_spam)

rows, cols = bow_sample_spam.nonzero()
for col in cols: 
    print(bow_transformer.get_feature_names_out()[col])


print(np.shape(bow_sample_spam))

sample_ham = data['text'][4]
bow_sample_ham = bow_transformer.transform([sample_ham])
print(sample_ham)
print(bow_sample_ham)

rows, cols = bow_sample_ham.nonzero()
for col in cols: 
    print(bow_transformer.get_feature_names_out()[col])

bow_data = bow_transformer.transform(data['text'])
print(bow_data.shape)
print(bow_data.nnz)
print( bow_data.nnz / (bow_data.shape[0] * bow_data.shape[1]) *100 )

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(bow_data)

tfidf_sample_ham = tfidf_transformer.transform(bow_sample_ham)
print(tfidf_sample_ham)

tfidf_sample_spam = tfidf_transformer.transform(bow_sample_spam)
print(tfidf_sample_spam)


data_tfidf = tfidf_transformer.transform(bow_data)

print(np.shape(data_tfidf))


from sklearn.model_selection import train_test_split

data_tfidf_train, data_tfidf_test, label_train, label_test = train_test_split(data_tfidf, data["spam"], test_size=0.3, random_state=5)

from scipy.sparse import  hstack
X2 = hstack((data_tfidf ,np.array(data['length'])[:,None])).A

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, data["spam"], test_size=0.3, random_state=5)

data_tfidf_train = data_tfidf_train.A
data_tfidf_test = data_tfidf_test.A

spam_detect_model = MultinomialNB().fit(data_tfidf_train, label_train)
pred_test_MNB = spam_detect_model.predict(data_tfidf_test)
acc_MNB = accuracy_score(label_test, pred_test_MNB)
print(acc_MNB)

scaler = MinMaxScaler()
data_tfidf_train_sc = scaler.fit_transform(data_tfidf_train)
data_tfidf_test_sc  = scaler.transform(data_tfidf_test)

spam_detect_model_minmax = MultinomialNB().fit(data_tfidf_train_sc, label_train)
pred_test_MNB = spam_detect_model_minmax.predict(data_tfidf_test_sc)
acc_MNB = accuracy_score(label_test, pred_test_MNB)
print(acc_MNB)

spam_detect_model_2 = MultinomialNB().fit(X2_train, y2_train)
pred_test_MNB_2 = spam_detect_model_2.predict(X2_test)
acc_MNB_2 = accuracy_score(y2_test, pred_test_MNB_2)
print(acc_MNB_2)

X2_tfidf_train = X2_train[:,0:9431]
X2_tfidf_test  = X2_test[:,0:9431]
X2_length_train = X2_train[:,9431]
X2_length_test  = X2_test[:,9431]

scaler = MinMaxScaler()
X2_tfidf_train = scaler.fit_transform(X2_tfidf_train)
X2_tfidf_test  = scaler.transform(X2_tfidf_test)
scaler = MinMaxScaler()
X2_length_train = scaler.fit_transform(X2_length_train.reshape(-1, 1))
X2_length_test  = scaler.transform(X2_length_test.reshape(-1, 1))
X2_train = np.hstack((X2_tfidf_train, X2_length_train))
X2_test  = np.hstack((X2_tfidf_test,  X2_length_test))

spam_detect_model_3 = MultinomialNB().fit(X2_train, y2_train)
pred_test_MNB_3 = spam_detect_model_3.predict(X2_test)
acc_MNB_3 = accuracy_score(y2_test, pred_test_MNB_3)
print(acc_MNB_3)

parameters_KNN = {'n_neighbors': (10,15,17), }

grid_KNN = GridSearchCV( KNeighborsClassifier(), parameters_KNN, cv=5,
                        n_jobs=-1, verbose=1)

grid_KNN.fit(data_tfidf_train, label_train)

print(grid_KNN.best_params_)
print(grid_KNN.best_score_)

parameters_KNN = {'n_neighbors': (6,8,10), }
grid_KNN = GridSearchCV( KNeighborsClassifier(), parameters_KNN, cv=5,
                        n_jobs=-1, verbose=1)
grid_KNN.fit(data_tfidf_train_sc, label_train)

print(grid_KNN.best_params_)
print(grid_KNN.best_score_)

from sklearn.model_selection import train_test_split

sms_train, sms_test, label_train, label_test = train_test_split(data["text"], data["spam"], test_size=0.3, random_state=5)

pipe_MNB = Pipeline([ ('bow'  , CountVectorizer(analyzer = remove_punctuation_and_stopwords) ),
                   ('tfidf'   , TfidfTransformer()),
                   ('clf_MNB' , MultinomialNB()),
                    ])

pipe_MNB.fit(X=sms_train, y=label_train)
pred_test_MNB = pipe_MNB.predict(sms_test)
acc_MNB = accuracy_score(label_test, pred_test_MNB)
print(acc_MNB)
print(pipe_MNB.score(sms_test, label_test))

from sklearn.feature_extraction.text import TfidfVectorizer
pipe_MNB_tfidfvec = Pipeline([ ('tfidf_vec' , TfidfVectorizer(analyzer = remove_punctuation_and_stopwords)),
                               ('clf_MNB'   , MultinomialNB()),
                            ])
pipe_MNB_tfidfvec.fit(X=sms_train, y=label_train)
pred_test_MNB_tfidfvec = pipe_MNB_tfidfvec.predict(sms_test)
acc_MNB_tfidfvec = accuracy_score(label_test, pred_test_MNB_tfidfvec)
print(acc_MNB_tfidfvec)
print(pipe_MNB_tfidfvec.score(sms_test, label_test))

pipe_KNN = Pipeline([ ('bow'  , CountVectorizer(analyzer = remove_punctuation_and_stopwords) ),
                   ('tfidf'   , TfidfTransformer()),
                   ('clf_KNN' , KNeighborsClassifier() )
                    ])

parameters_KNN = {'clf_KNN__n_neighbors': (8,15,20), }

grid_KNN = GridSearchCV(pipe_KNN, parameters_KNN, cv=5,
                        n_jobs=-1, verbose=1)

grid_KNN.fit(X=sms_train, y=label_train)

pred_test_grid_KNN = grid_KNN.predict(sms_test)
acc_KNN = accuracy_score(label_test, pred_test_grid_KNN)
print(acc_KNN)
print(grid_KNN.score(sms_test, label_test))

pipe_SVC = Pipeline([ ('bow'  , CountVectorizer(analyzer = remove_punctuation_and_stopwords) ),
                   ('tfidf'   , TfidfTransformer()),
                   ('clf_SVC' , SVC(gamma='auto', C=1000)),
                    ])


parameters_SVC = dict(tfidf=[None, TfidfTransformer()],
                      clf_SVC__C=[500, 1000,1500]
                      )
#parameters = {'tfidf__use_idf': (True, False),    }

grid_SVC = GridSearchCV(pipe_SVC, parameters_SVC, 
                        cv=5, n_jobs=-1, verbose=1)

grid_SVC.fit(X=sms_train, y=label_train)

pred_test_grid_SVC = grid_SVC.predict(sms_test)
acc_SVC = accuracy_score(label_test, pred_test_grid_SVC)
print(acc_SVC)
print(grid_SVC.score(sms_test, label_test))

pipe_SGD = Pipeline([ ('bow'  , CountVectorizer(analyzer = remove_punctuation_and_stopwords) ),
                   ('tfidf'   , TfidfTransformer()),
                   ('clf_SGD' , SGDClassifier(random_state=5)),
                    ])

parameters_SGD = {
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    #'clf_SGD__max_iter': (5,10),
    'clf_SGD__alpha': (1e-05, 1e-04),
}

grid_SGD = GridSearchCV(pipe_SGD, parameters_SGD, cv=5,
                               n_jobs=-1, verbose=1)

grid_SGD.fit(X=sms_train, y=label_train)

pred_test_grid_SGD = grid_SGD.predict(sms_test)
acc_SGD = accuracy_score(label_test, pred_test_grid_SGD)
print(acc_SGD)
print(grid_SGD.score(sms_test, label_test))

pipe_GBC = Pipeline([ ('bow'  , CountVectorizer(analyzer = remove_punctuation_and_stopwords) ),
                      ('tfidf'   , TfidfTransformer() ),
                      ('clf_GBC' , GradientBoostingClassifier(random_state=5) ),
                    ])

parameters_GBC = { 'tfidf__use_idf': (True, False), 
                   'clf_GBC__learning_rate': (0.1, 0.2),
                   #'clf_GBC__min_samples_split': (3,5), 
                 }

grid_GBC = GridSearchCV(pipe_GBC, parameters_GBC, 
                        cv=5, n_jobs=-1, verbose=1)

grid_GBC.fit(X=sms_train, y=label_train)

pred_test_grid_GBC = grid_GBC.predict(sms_test)
acc_GBC = accuracy_score(label_test, pred_test_grid_GBC)
print(acc_GBC)
print(grid_GBC.score(sms_test, label_test))

import xgboost as xgb

# Set params['eval_metric'] = ...
pipe_XGB = Pipeline([ ('bow'  , CountVectorizer(analyzer = remove_punctuation_and_stopwords) ),
                      ('tfidf'   , TfidfTransformer() ),
                      ('clf_XGB' , xgb.XGBClassifier(random_state=5) ),
                    ])

parameters_XGB = { 'tfidf__use_idf': (True, False), 
                   'clf_XGB__eta': (0.01, 0.02),
                   'clf_XGB__max_depth': (5,6), 
                 }

grid_XGB = GridSearchCV(pipe_XGB, parameters_XGB, 
                        cv=5, n_jobs=-1, verbose=1)

grid_XGB.fit(X=sms_train, y=label_train)

pred_test_grid_XGB = grid_XGB.predict(sms_test)
acc_XGB = accuracy_score(label_test, pred_test_grid_XGB)
print(acc_XGB)
print(grid_XGB.score(sms_test, label_test))

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

def plot_confusion_matrix(y_true, y_pred):
    mtx = confusion_matrix(y_true, y_pred)
    #fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  
                cmap="Blues", square=True, cbar=False)
    #  
    plt.ylabel('true label')
    plt.xlabel('predicted label')

list_clf = ["MNB", "KNN", "SVC", "SGD", "GBC", "XGB"]

list_pred = [pred_test_MNB, pred_test_grid_KNN, 
             pred_test_grid_SVC, pred_test_grid_SGD,
             pred_test_grid_GBC, pred_test_grid_XGB]

dict_pred = dict(zip(list_clf, list_pred))

def plot_all_confusion_matrices(y_true, dict_all_pred, str_title):
    
    list_classifiers = list(dict_all_pred.keys())
    plt.figure(figsize=(10,7.5))
    plt.suptitle(str_title, fontsize=20, fontweight='bold')
    n=231

    for clf in list_classifiers : 
        plt.subplot(n)
        plot_confusion_matrix(y_true, dict_all_pred[clf])
        plt.title(clf, fontweight='bold')
        n+=1

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
plot_all_confusion_matrices(label_test, dict_pred, "Pipelines v1, scoring=accuracy")

dict_acc = {}
for clf in list_clf :
    dict_acc[clf] = accuracy_score(label_test, dict_pred[clf])
for clf in list_clf :
    print(clf, " " , dict_acc[clf])


for clf in list_clf :
    print(clf, " ", precision_score(label_test, dict_pred[clf]))

for clf in list_clf :
    print(clf, " ", precision_score(label_test, dict_pred[clf], average=None, labels=[0,1]))

for clf in list_clf :
    print(clf, " ", recall_score(label_test, dict_pred[clf]))

for clf in list_clf :
    print(clf, " ", recall_score(label_test, dict_pred[clf], average=None, labels=[0,1] ))

for clf in list_clf :
    print(clf, " ", f1_score(label_test, dict_pred[clf]))

for clf in list_clf :
    print(clf, " ", f1_score(label_test, dict_pred[clf], average=None, labels=[0,1] ))

print(classification_report(label_test, pred_test_MNB))

for clf in list_clf :
    print(clf, " ", precision_recall_fscore_support(label_test, dict_pred[clf], average=None, labels=[0,1] ))

for clf in list_clf :
    print(clf, " ", roc_auc_score(label_test, dict_pred[clf] ))

import sklearn.metrics
sklearn.metrics.SCORERS.keys()

scoring = 'precision'

grid_KNN_2 = GridSearchCV(pipe_KNN, parameters_KNN, cv=5,
                          scoring=scoring, n_jobs=-1, verbose=1)

grid_KNN_2.fit(X=sms_train, y=label_train)
pred_test_grid_KNN_2 = grid_KNN_2.predict(sms_test)

grid_SVC_2 = GridSearchCV(pipe_SVC, parameters_SVC, cv=5,
                          scoring=scoring, n_jobs=-1, verbose=1)

grid_SVC_2.fit(X=sms_train, y=label_train)
pred_test_grid_SVC_2 = grid_SVC_2.predict(sms_test)

grid_SGD_2 = GridSearchCV(pipe_SGD, parameters_SGD, cv=5,
                          scoring=scoring, n_jobs=-1, verbose=1)

grid_SGD_2.fit(X=sms_train, y=label_train)
pred_test_grid_SGD_2 = grid_SGD_2.predict(sms_test)

grid_GBC_2 = GridSearchCV(pipe_GBC, parameters_GBC, cv=5,
                          scoring=scoring, n_jobs=-1, verbose=1)

grid_GBC_2.fit(X=sms_train, y=label_train)
pred_test_grid_GBC_2 = grid_GBC_2.predict(sms_test)

grid_XGB_2 = GridSearchCV(pipe_XGB, parameters_XGB, cv=5,
                          scoring=scoring, n_jobs=-1, verbose=1)

grid_XGB_2.fit(X=sms_train, y=label_train)
pred_test_grid_XGB_2 = grid_XGB_2.predict(sms_test)

# Confusion matrices for scoring by precision
list_clf = ["MNB", "KNN_2", "SVC_2", "SGD_2", "GBC_2", "XGB_2"]

list_pred = [pred_test_MNB, pred_test_grid_KNN_2, 
             pred_test_grid_SVC_2, pred_test_grid_SGD_2,
             pred_test_grid_GBC_2, pred_test_grid_XGB_2]

dict_pred_2 = dict(zip(list_clf, list_pred))
plot_all_confusion_matrices(label_test, dict_pred_2, "Pipelines v2, scoring=precision")

scoring = 'recall'

grid_KNN_3 = GridSearchCV(pipe_KNN, parameters_KNN, cv=5,
                          scoring=scoring, n_jobs=-1, verbose=1)

grid_KNN_3.fit(X=sms_train, y=label_train)
pred_test_grid_KNN_3 = grid_KNN_3.predict(sms_test)

grid_SVC_3 = GridSearchCV(pipe_SVC, parameters_SVC, cv=5,
                          scoring=scoring, n_jobs=-1, verbose=1)

grid_SVC_3.fit(X=sms_train, y=label_train)
pred_test_grid_SVC_3 = grid_SVC_3.predict(sms_test)

grid_SGD_3 = GridSearchCV(pipe_SGD, parameters_SGD, cv=5,
                          scoring=scoring, n_jobs=-1, verbose=1)

grid_SGD_3.fit(X=sms_train, y=label_train)
pred_test_grid_SGD_3 = grid_SGD_3.predict(sms_test)

grid_GBC_3 = GridSearchCV(pipe_GBC, parameters_GBC, cv=5,
                          scoring=scoring, n_jobs=-1, verbose=1)

grid_GBC_3.fit(X=sms_train, y=label_train)
pred_test_grid_GBC_3 = grid_GBC_3.predict(sms_test)

grid_XGB_3 = GridSearchCV(pipe_XGB, parameters_XGB, cv=5,
                          scoring=scoring, n_jobs=-1, verbose=1)

grid_XGB_3.fit(X=sms_train, y=label_train)
pred_test_grid_XGB_3 = grid_XGB_3.predict(sms_test)

# Confusion matrices for scoring by recall¶
list_clf = ["MNB", "KNN_3", "SVC_3", "SGD_3", "GBC_3", "XGB_3"]

list_pred = [pred_test_MNB, pred_test_grid_KNN_3, 
             pred_test_grid_SVC_3, pred_test_grid_SGD_3,
             pred_test_grid_GBC_3, pred_test_grid_XGB_3]

dict_pred_3 = dict(zip(list_clf, list_pred))
plot_all_confusion_matrices(label_test, dict_pred_3, "Pipelines v3, scoring=recall")

scoring = 'roc_auc'

grid_KNN_4 = GridSearchCV(pipe_KNN, parameters_KNN, cv=5,
                          scoring=scoring, n_jobs=-1, verbose=1)

grid_KNN_4.fit(X=sms_train, y=label_train)
pred_test_grid_KNN_4 = grid_KNN_4.predict(sms_test)

from sklearn.metrics import roc_curve, auc

fpr, tpr, thr = roc_curve(label_test, grid_KNN_4.predict_proba(sms_test)[:,1])
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Plot')
auc_knn4 = auc(fpr, tpr) * 100
plt.legend(["AUC {0:.3f}".format(auc_knn4)])
plt.show()


grid_SVC_4 = GridSearchCV(pipe_SVC, parameters_SVC, cv=5,
                          scoring=scoring, n_jobs=-1, verbose=1)

grid_SVC_4.fit(X=sms_train, y=label_train)
pred_test_grid_SVC_4 = grid_SVC_4.predict(sms_test)

grid_SGD_4 = GridSearchCV(pipe_SGD, parameters_SGD, cv=5,
                          scoring=scoring, n_jobs=-1, verbose=1)

grid_SGD_4.fit(X=sms_train, y=label_train)
pred_test_grid_SGD_4 = grid_SGD_4.predict(sms_test)

grid_GBC_4 = GridSearchCV(pipe_GBC, parameters_GBC, cv=5,
                          scoring=scoring, n_jobs=-1, verbose=1)

grid_GBC_4.fit(X=sms_train, y=label_train)
pred_test_grid_GBC_4 = grid_GBC_4.predict(sms_test)

from sklearn.metrics import roc_curve, auc
fpr, tpr, thr = roc_curve(label_test, grid_GBC_4.predict_proba(sms_test)[:,1])
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Plot')
auc_gbc4 = auc(fpr, tpr) * 100
plt.legend(["AUC {0:.3f}".format(auc_gbc4)])
plt.show()

grid_XGB_4 = GridSearchCV(pipe_XGB, parameters_XGB, cv=5,
                          scoring=scoring, n_jobs=-1, verbose=1)

grid_XGB_4.fit(X=sms_train, y=label_train)
pred_test_grid_XGB_4 = grid_XGB_4.predict(sms_test)

# Confusion matrices for scoring by roc auc
list_clf = ["MNB", "KNN_4", "SVC_4", "SGD_4", "GBC_4", "XGB_4"]

list_pred = [pred_test_MNB, pred_test_grid_KNN_4, 
             pred_test_grid_SVC_4, pred_test_grid_SGD_4,
             pred_test_grid_GBC_4, pred_test_grid_XGB_4]

dict_pred_4 = dict(zip(list_clf, list_pred))
plot_all_confusion_matrices(label_test, dict_pred_4, "Pipelines v4, scoring=roc auc")

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
print(data['text'][7])

print(sent_tokenize(data['text'][7]))

print(word_tokenize(data['text'][7]))

stopWords = set(stopwords.words('english'))
words = word_tokenize(data['text'][7])
wordsFiltered = []

for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)

print(wordsFiltered)

