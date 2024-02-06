import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, f1_score

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
nltk.download('vader_lexicon')


df = pd.read_csv('reduced_rotten_tomatoes_movie_reviews.csv')
#print(df.columns)

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

def vader_sentiment(review):
    score = sia.polarity_scores(review)
    #print(score)
    if score['compound'] >= 0.05:
        print("positive")
        return 'POSITIVE'
    elif score['compound'] <= -0.05:
        print("negative")
        return 'NEGATIVE'
    else:
        print("negative")
        return 'NEUTRAL' 





def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def sentiwordnet_sentiment(review):
    pos_tagged = pos_tag(word_tokenize(review))
    scores = []
    for word, tag in pos_tagged:
        
        wordnet_tag = get_wordnet_pos(tag)
        
        if wordnet_tag not in (wordnet.NOUN, wordnet.ADJ, wordnet.ADV, wordnet.VERB):
            continue
        lemma = lemmatizer.lemmatize(word, pos=wordnet_tag)
        synsets = list(sentiwordnet.senti_synsets(lemma, wordnet_tag))
        if not synsets:
            continue
        # Take the first sense, the most common
        synset = synsets[0]
        score = synset.pos_score() - synset.neg_score()
        scores.append(score)
    # Decision on '0' score could be based on domain knowledge or threshold tuning
    sentiment = 'POSITIVE' if np.mean(scores) > 0 else 'NEGATIVE'
    return sentiment



df['VADER_sentiment'] = df['reviewText'].apply(vader_sentiment) #.apply is super sick 
df['SWN_sentiment'] = df['reviewText'].apply(sentiwordnet_sentiment)




# Assuming 'scoreSentiment' is your ground truth and it's already compatible with your model outputs

# Evaluation for VADER Sentiment Analysis
vader_accuracy = accuracy_score(df['scoreSentiment'], df['VADER_sentiment'])
vader_f1_micro = f1_score(df['scoreSentiment'], df['VADER_sentiment'], average='micro')
vader_f1_macro = f1_score(df['scoreSentiment'], df['VADER_sentiment'], average='macro')

print("VADER Sentiment Analysis Evaluation")
print("Accuracy:", vader_accuracy)
print("F1-Score (Micro):", vader_f1_micro)
print("F1-Score (Macro):", vader_f1_macro)

# Evaluation for SentiWordNet Sentiment Analysis
swn_accuracy = accuracy_score(df['scoreSentiment'], df['SWN_sentiment'])
swn_f1_micro = f1_score(df['scoreSentiment'], df['SWN_sentiment'], average='micro')
swn_f1_macro = f1_score(df['scoreSentiment'], df['SWN_sentiment'], average='macro')

print("\nSentiWordNet Sentiment Analysis Evaluation")
print("Accuracy:", swn_accuracy)
print("F1-Score (Micro):", swn_f1_micro)
print("F1-Score (Macro):", swn_f1_macro)
