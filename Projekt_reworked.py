#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xlrd
from xlrd import open_workbook
import nltk
from nltk.corpus import stopwords
import re


# In[2]:


# open .xls files as xlrd workbooks and save them into sheets
en_wb = open_workbook(r"file.xls")
en_sheet = en_wb.sheet_by_index(0)
de_wb = open_workbook(r"file.xls")
de_sheet = de_wb.sheet_by_index(0)
fr_wb = open_workbook(r"file.xls")
fr_sheet = fr_wb.sheet_by_index(0)
pt_wb = open_workbook(r"file.xls")
pt_sheet = pt_wb.sheet_by_index(0)


# In[3]:


# a list of emojis
emojis = [":)", ":-)", "(:", ":))", ":-]", ":]", ";)", ";-)", ":-3", ":3",  ":->",  ":>",  ":o)",  "=]",  "=)", "(=", ":D", ":-D", ";D", "xD", "^^", "^-^", "^_^", "*-*", "*_*", "*__*", "*___*", "♥", ":(", ":((", ":-(", "):", ":'(", ":'((", ";-(" ":/", ":-/", ":-|", ":|", "D:", "O.o", "Oo", "o.O", "oO", ">.<", "-.-", "._.", "-_-", "-.-'"]


# In[4]:


stop_en = stopwords.words('english')
stop_de = stopwords.words('german')
stop_fr = stopwords.words('french')
stop_pt = stopwords.words('portuguese')


# In[5]:


# define a function which extracts the data from the excel sheet to a list
def sheet_extractor(sheet):
    lang_list = []
    # for each row the data is saved as a triple consisting of sentiment (String), tweet-ID (int) and the tweet (String) and put into a list
    for x in range(1,sheet.nrows):
        sentiment = (sheet.cell_value(x,0))
        tweetID = int((sheet.cell_value(x,1)))
        tweet = (sheet.cell_value(x,2))
        triple = (sentiment, tweetID, tweet)
        lang_list.append(triple)
    return lang_list


# In[6]:


# define a function which: 
# inserts whitespace before and after punctuation
# deletes double/multiple whitespaces
# transforms the tweets into lowercase and splits each tweet into a list of strings
def punctuation_splitter(tweetlist):
    new_tweetlist = [(a,b,(re.sub('([.,;!?&])', r' \1 ', c))) for (a,b,c) in tweetlist]
    new_tweetlist = [(a,b,(re.sub('\s{2,}', ' ', c))) for (a,b,c) in new_tweetlist]
    new_tweetlist = [(a,b,c.lower().split()) for (a,b,c) in new_tweetlist]
    return new_tweetlist


# In[7]:


# just_tweet_words creates a list with just the tweet words/vocabulary stripped from sentiment and tweet (list of list of strings)
def just_tweet_words(wordlist):
    tweets = [c for (a,b,c) in wordlist]
    return tweets


# In[8]:


# creates a flat list of words (not divided by tweet!) (list of strings)
def clean_tweets(wordlist):
    tweets = just_tweet_words(wordlist)
    tweets_clean = [word for sent in tweets for word in sent if word.isalpha()]
    return tweets_clean


# In[9]:


#  creates list without stopwords and non-alphabetic (i.e. numbers, punctuation) words (list of strings)
def stopwords(wordlist, stopwordlist):
    tweets_stopw = [word for sent in wordlist for word in sent if word not in stopwordlist and word.isalpha()]
    return tweets_stopw


# In[10]:


# finds out the 300 most frequently used words and saves just the words in a list (list of strings)
def topwords(wordlist):
    tweet_topwords = nltk.FreqDist(wordlist).most_common(300)
    tweet_topwords = [word for word,freq in tweet_topwords]
    return tweet_topwords


# In[11]:


# extracts the hashtags
def hashtags(wordlist):
    tweet_hashtags = [word for sent in wordlist for word in sent if "#" in word]
    return tweet_hashtags


# In[12]:


# creates bigrams and trigrams of the tweets (list of tuples/triples of strings)
def bigrams(wordlist):
    tweet_bigrams = list(nltk.bigrams(wordlist))
    top_bigrams = nltk.FreqDist(tweet_bigrams).most_common(50)
    top_bigrams = [bigram for bigram,freq in top_bigrams]
    return top_bigrams


# In[13]:


def trigrams(wordlist):
    tweet_trigrams = list(nltk.trigrams(wordlist))
    top_trigrams = nltk.FreqDist(tweet_trigrams).most_common(50)
    top_trigrams = [trigram for trigram,freq in top_trigrams]
    return top_trigrams


# In[14]:


# creates a list of tuples which contain tweets as list of Strings and sentiment
def tweet_sent(wordlist):
    tweet_sent_list = [(tweet, sentiment) for (sentiment, tweetID, tweet) in wordlist]
    return tweet_sent_list


# In[15]:


def extract_tweet_features(tweet, all_featurelist):
    features = {}
    tweet_bigrams = list(nltk.bigrams(tweet))
    tweet_trigrams = []
    if len(tweet) >= 3:
        tweet_trigrams = list(nltk.trigrams(tweet))
        # all_featurelist contains topwords, hashtags, bigrams and trigrams
    for featurelist in all_featurelist:
        for element in featurelist:
            features[element] = (element in tweet)
    return features


# In[16]:


# main method which calls every functionand applies the classifier to the data
def main(sheet, stopw):
    tweetlist = sheet_extractor(sheet)
    new_tweetlist = punctuation_splitter(tweetlist)
    tweet_words = just_tweet_words(new_tweetlist)
    clean_tweetlist = clean_tweets(new_tweetlist)
    tweet_stopw = stopwords(tweet_words, stopw)
    tweet_topw = topwords(tweet_stopw)
    tweet_hasht = hashtags(tweet_words)
    tweet_bigram = bigrams(clean_tweetlist)
    tweet_trigram = trigrams(clean_tweetlist)
    t_s = tweet_sent(new_tweetlist)
    # all lists as one so that it can be passed as one argument
    all_list = []
    all_list.append(tweet_topw)
    all_list.append(tweet_hasht)
    all_list.append(tweet_bigram)
    all_list.append(tweet_trigram)
    all_list.append(emojis) 
    
    # apply the feature extractor to the tweets
    featuresets = [(extract_tweet_features(tweet, all_list),sent) for (tweet, sent) in t_s]  
    # divide the data into a train_set (90% of the data) and a devtest_set (10% of the data) as the test_set is separate
    train_set = featuresets[:(int(len(featuresets) * 0.9))]
    devtest_set = featuresets[(int(len(featuresets) * 0.9)):] 
    # train the classifier with the train_set
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print("accuracy: " + str(nltk.classify.accuracy(classifier, devtest_set)))
    classifier.show_most_informative_features(20)
    


# In[17]:


main(en_sheet, stop_en)


# In[18]:


main(de_sheet, stop_de)


# In[19]:


main(fr_sheet, stop_fr)


# In[20]:


main(pt_sheet, stop_pt)

