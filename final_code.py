#load modules
import numpy as np
import sys
import pickle
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import xgboost as xgb
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import spacy
import re
from pytictoc import TicToc
import matplotlib.pyplot as plt

# we first set the seed to get deterministically shuffled results.
random.seed(0)
np.random.seed(0)

# load the various text-preprocessing parsers and lists
re_tokenizer = RegexpTokenizer(r'\w+') # remove punctuation
nlp = spacy.load('en', disable=['parser', 'ner'])
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# A simple helper function to generate train, validate, and test
def train_valid_test_split(data_x, data_y, train_ratio, valid_ratio, test_ratio, rand_seed=0):
    if (train_ratio + valid_ratio + test_ratio) > 1:
        return None, None, None, None, None, None
    else:
        train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=(valid_ratio + test_ratio),
                                                              random_state=rand_seed)
        if not valid_ratio == 0:
            valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y,
                                                                test_size=(test_ratio / (test_ratio + valid_ratio)),
                                                                random_state=rand_seed)
        else:
            test_x = valid_x
            test_y = valid_y
            valid_x = None
            valid_y = None
        return train_x, valid_x, test_x, train_y, valid_y, test_y

# a simple helper function that takes a model and finds accuracy and the Confusion Matrix
def predict_scores(clf_name, mod, test_x, test_y):
    predict_y = mod.predict(test_x)
    cmatrix = confusion_matrix(test_y, predict_y, labels=[1, 0])
    score = accuracy_score(test_y, predict_y)
    cm_key = np.array([['TP', 'TN'], ['FP', 'FN']])
    print(clf_name, "confusion matrix:")
    print(np.c_[cm_key, cmatrix])
    print(clf_name, 'accuracy percentage: ', score * 100)

#Regular Expression 'stop words' - sequences of text that we might want to remove to improve performance
usernames = re.compile("@\\w+ *") # regex to remove usernames
odd_chars = {'&amp;', '&quot;', '&lt;3', '♥', '&gt;', '�', '&gt;', '♫'} # various odd characters found in training_data.csv
emoticons = {':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}', ':^)', ':-D', ':D', '8-D',
                   '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D', '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P',
                   ':-P', ':P', 'X-P', 'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)', '<3',
                   ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
                   ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
                   ':c', ':{', '>:\\', ';('} # might be beneficial to keep for sentiment

# more regex to remove links (as seen in training_data.csv) as well as emojis (in case they do appear)
remove_links = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
remove_emojis = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

'''
Remove usernames, blank words, odd characters and emoticons
Allow for removing stop words
Allow for lemmatization
'''
def parse_words(parse_str, lower_flag = False, lemma_flag = False, stop_flag = False):
    # remove usernames
    parse_list = parse_str.split(' ')
    parse_list = [word for word in parse_list if len(word) > 0 and '@' not in word]
    parse_list = [word for word in parse_list if word not in odd_chars and word not in emoticons]
    parse_str = ' '.join(parse_list)
    parse_str = remove_links.sub('', parse_str)
    parse_str = remove_emojis.sub('', parse_str)
    if lower_flag:
        tokenized_words = re_tokenizer.tokenize(parse_str.lower())
    else:
        tokenized_words = re_tokenizer.tokenize(parse_str)
    if stop_flag:
        stripped_words = [stem for stem in tokenized_words if not stem in stop_words]
    else:
        stripped_words = tokenized_words
    if lemma_flag:
        lemmatized_words = nlp(' '.join(stripped_words))
        return ' '.join([token.lemma_ for token in lemmatized_words])
    else:
        return ' '.join(stripped_words)

# Performs xgboost using either Grid Search or Randomized Search
def xgb_clf(param_dict, cv_func, train_x, train_y, num_folds=10, gpu=None, rand_seed=0, num_iter=100):
    if gpu:
        clf = xgb.XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')
        if cv_func == GridSearchCV:
            cv_model = cv_func(clf, param_dict, scoring='accuracy', cv=num_folds)
        elif cv_func == RandomizedSearchCV:
            cv_model = cv_func(clf, param_dict, scoring='accuracy', cv=num_folds, n_iter=num_iter)
    else:
        clf = xgb.XGBClassifier()
        if cv_func == GridSearchCV:
            cv_model = cv_func(clf, param_dict, n_jobs=-1, scoring='accuracy', cv=num_folds)
        elif cv_func == RandomizedSearchCV:
            cv_model = cv_func(clf, param_dict, n_jobs=-1, scoring='accuracy', cv=num_folds, random_state=rand_seed, n_iter=num_iter, verbose=1)
    cv_model.fit(train_x, train_y)
    print('Best validation params', cv_model.best_params_)
    print('Best validation score:', cv_model.best_score_)

    clf.set_params(**cv_model.best_params_)
    clf.fit(train_x, train_y)
    return clf

try:
    if sys.argv[1] == 'test':
        training_data = sys.argv[2]
        testing_data = sys.argv[3]
        cv_or_test = 'test'
    elif sys.argv[1]  == 'cv':
        training_data = sys.argv[2]
        cv_or_test = 'cv'
    elif sys.argv[1] == 'load':
        cv_or_test = 'load'
    else:
        raise ValueError
except:
    print('USAGE (train on training and test):', sys.argv[0], 'test', 'FILENAMETRAINING FILENAMETESTING')
    print('OR')
    print('USAGE (cv):', sys.argv[0], 'cv', 'FILENAMETRAINING')
    print('OR')
    print('USAGE (to load pickle files):', sys.argv[0], 'load')
    exit(1)

features = []
labels = []

with open(training_data, 'r') as f:
    next(f) # skip first line
    g = f.readlines()
    random.shuffle(g)  # get better training
    for line in g:
        if line[0] == '#':
            continue
        line = line.strip()
        loc = line.replace(',', '?', 1).find(',')
        text = parse_words(line[loc:-2])
        features.append(text)
        labels.append(int(line[-1]))

labels = np.array(labels)

# summary of features and labels
print(features[:10])
print(labels[:10])

vectorizer = CountVectorizer(binary=False, lowercase=False,max_features=1000000)  # ngram_range, max features are things to tweak
model = SGDClassifier(loss='log', random_state=0, verbose=10, n_jobs=-1)  #LogisticRegression(max_iter=1000, verbose=1)  # LogisticRegressionCV(max_iter=1000, n_jobs=-1, verbose=1)
# SGDClassifier(loss='log', random_state=0, verbose=1, n_jobs=-1,tol=1e-4)  #

if cv_or_test == 'cv':
    #split data
    trainX, validX, testX, trainY, validY, testY = train_valid_test_split(features, labels, 0.9, 0, 0.1)

    encoded_trainX = vectorizer.fit_transform(trainX)
    encoded_testX = vectorizer.transform(testX)
    pickle.dump(encoded_trainX, open('encoded_trainX.pkl', 'wb'))
    pickle.dump(encoded_testX, open('encoded_testX.pkl', 'wb'))
    pickle.dump(trainX, open('trainX.pkl', 'wb'))
    pickle.dump(testX, open('testX.pkl', 'wb'))
    pickle.dump(trainY, open('trainY.pkl', 'wb'))
    pickle.dump(testY, open('testY.pkl', 'wb'))


    # dummy baseline class to return all ones
    class Baseline:
        def __init__(self):
            pass

        def predict(self, test):
            test = np.array(test)
            return np.ones((test.shape[0], 1))

    bmodel = Baseline()
    predict_scores("Baseline, all 1's", bmodel, testX, testY)

    print('Ready to train')
    t = TicToc()
    t.tic()
    model.fit(encoded_trainX, trainY)
    t.toc()
    pickle.dump(model, open('logistic.pkl', 'wb'))
    predict_scores("Logistic Regression", model, encoded_testX, testY)
    t.toc()

    # took too long to finish so ultimately abandoned it
    # xgb_params = {
    #         #"booster": ['dart'],
    #         "objective": ["binary:logistic", "reg:tweedie", "reg:gamma", "rank:pairwise"], #try various objectives
    #         #"nthread": [-1], #use as many threads as available
    #         "eta": list(np.arange(0,10,0.2)),
    #         #"max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    #         #"min_child_weight": [1, 3, 5, 7],
    #         "lambda": list(np.arange(0, 10, 0.2)), # [4.7]
    #         "gamma": list(np.arange(0, 10, 0.2)),
    #         #"colsample_bytree": [0.25, 0.5, 0.75, 1]
    # }

    # t=TicToc()
    # t.tic()
    # model = xgb_clf(xgb_params, RandomizedSearchCV, encoded_trainX, trainY)
    # xgb_model = model.fit(encoded_trainX, trainY)
    # t.toc()
    # predict_scores("xgb", xgb_model, encoded_testX, testY)
    # pickle.dump(xgb_model, open('xgboost.pkl','wb'))
    # xgb.plot_importance(xgb_model,importance_type='gain')
    # plt.show()
elif cv_or_test == 'load':
    encoded_trainX = pickle.load(open('encoded_trainX.pkl', 'rb'))
    encoded_testX = pickle.load(open('encoded_testX.pkl', 'rb'))
    trainX = pickle.load(open('trainX.pkl', 'rb'))
    testX = pickle.load(open('testX.pkl', 'rb'))
    trainY = pickle.load(open('trainY.pkl', 'rb'))
    testY = pickle.load(open('testY.pkl', 'rb'))
    cv_or_test == 'test' #now we test the model
if cv_or_test == 'test':
    trainX = features
    trainY = labels
    testX = []
    testing_file = None

    with open(testing_data, 'r') as f:
        next(f)
        testing_file = f.readlines()
        #remove newlines
        for i in range(len(testing_file)):
            testing_file[i] = testing_file[i].strip()
        for line in testing_file:
            loc = line.replace(',', '?', 1).find(',')
            text = parse_words(line[loc:])
            testX.append(text)

    encoded_trainX = vectorizer.fit_transform(trainX)
    encoded_testX = vectorizer.transform(testX)
    pickle.dump(encoded_trainX, open('encoded_trainX.pkl', 'wb'))
    pickle.dump(encoded_testX, open('encoded_testX.pkl', 'wb'))
    pickle.dump(trainX, open('trainX.pkl', 'wb'))
    pickle.dump(testX, open('testX.pkl', 'wb'))
    pickle.dump(trainY, open('trainY.pkl', 'wb'))

    print('Ready to train')
    t = TicToc()
    t.tic()
    # SGDClassifier(loss='log', random_state=0, verbose=1, n_jobs=-1,tol=1e-4)  #
    model.fit(encoded_trainX, trainY)
    t.toc()
    pickle.dump(model, open('logistic.pkl', 'wb'))
    encoded_testX = vectorizer.transform(testX)
    predictY = model.predict(encoded_testX)

    with open('OUTPUT_contest_judgement.csv', 'w+') as f:
        for i in range(len(testing_file)):
            f.write(testing_file[i] + ',' + str(predictY[i]) + '\n')


