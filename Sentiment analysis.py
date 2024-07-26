import pandas as pd

data = pd.read_csv("data.csv")

data.drop(["Sentiment"], axis=1, inplace=True)

print(data.isna().sum())

data = data.dropna()

print(data.isna().sum())

import nltk

#download vader from nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#creating an object of sentiment intensity analyzer
sia= SentimentIntensityAnalyzer()

data['scores']=data['Sentence'].apply(lambda body: sia.polarity_scores(str(body)))

data['compound']=data['scores'].apply(lambda score_dict:score_dict['compound'])
data.head()
data['pos']=data['scores'].apply(lambda pos_dict:pos_dict['pos'])
data.head()
data['neg']=data['scores'].apply(lambda neg_dict:neg_dict['neg'])

data['type']=''
data.loc[data.compound>0,'type']='POS'
data.loc[data.compound==0,'type']='NEUTRAL'
data.loc[data.compound<0,'type']='NEG'


print(data.head()
      )

data.to_csv("Final-result.csv")