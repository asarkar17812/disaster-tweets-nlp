import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pprint
import seaborn as sns

pp = pprint.PrettyPrinter()
df_train = pd.read_csv('F:\\disaster_tweets\\source\\train.csv')

X = df_train.text
y = df_train.target

fig = plt.figure(figsize=(6,4))
colors=["#00FF2A","#DF1616"]
pos=y[y == 1]
neg=y[y == 0]
ck=[pos.count(),neg.count()]
legpie=plt.pie(ck,labels=["True Label","False Label"],
                 autopct ='%1.1f%%', 
                 colors = colors,
                 startangle=90)
plt.title("Pi Chart of Training Disaster Tweets by Label")
plt.savefig(r'F:\disaster_tweets\plots\eda\tweet_label_piChart.png', dpi=1200, bbox_inches='tight')
plt.show()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
tweet_len=df_train[df_train['target']==1]['text'].str.split().map(lambda x: len(x))
ax1.hist(tweet_len,color='blue')
ax1.set_title('disaster tweets')
tweet_len=df_train[df_train['target']==0]['text'].str.split().map(lambda x: len(x))
ax2.hist(tweet_len,color='yellow')
ax2.set_title('Non disaster tweets')
fig.suptitle('Words in a processed tweet')
plt.savefig(r'F:\disaster_tweets\plots\eda\tweet_word_count_histogram.png', dpi=1200, bbox_inches='tight')
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Disaster tweets
word = df_train[df_train['target'] == 1]['text'].str.split().apply(lambda x: [len(i) for i in x])
sns.histplot(word.map(lambda x: np.mean(x)), kde=True, ax=ax1, color='purple')
ax1.set_title('Disaster tweets')

# Non-disaster tweets
word = df_train[df_train['target'] == 0]['text'].str.split().apply(lambda x: [len(i) for i in x])
sns.histplot(word.map(lambda x: np.mean(x)), kde=True, ax=ax2, color='orange')
ax2.set_title('Non-disaster tweets')

fig.suptitle('Average word length in each processed tweet')
plt.savefig(r'F:\disaster_tweets\plots\eda\tweet_word_count_histogram_pdf.png', dpi=1200, bbox_inches='tight')
plt.show()