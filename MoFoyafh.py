import snscrape.modules.twitter as sntwitter
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import matplotlib.pyplot as plt
import dataframe_image as dfi
import markovify
from pycaret.anomaly import *
from pycaret.clustering import *
from ctgan import CTGANSynthesizer
from sklearn import preprocessing
import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from PIL import ImageDraw, Image, ImageFont

s=[]

def twitter_pestel():
    global s
    l=['politics','economy','legal','environmental','technological','social']
    # Creating list to append tweet data
    for x in l:
        tweets_list1 = []
        # Using TwitterSearchScraper to scrape data and append tweets to list
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(x).get_items()): #declare a username 
            if i>300: #number of tweets you want to scrape
                break
            tweets_list1.append([tweet.date, tweet.id, tweet.content]) #declare the attributes to be returned
            
        # Creating a dataframe from the tweets list above 
        tweets_df1 = pd.DataFrame(tweets_list1, columns=['Datetime', 'Tweet Id', 'Text'])
        #topic modelling
        from sklearn.feature_extraction.text import CountVectorizer

        count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
        doc_term_matrix = count_vect.fit_transform(tweets_df1['Text'].values.astype('U'))
        from sklearn.decomposition import LatentDirichletAllocation

        LDA = LatentDirichletAllocation(n_components=5, random_state=42)
        LDA.fit(doc_term_matrix)
        first_topic = LDA.components_[0]
        top_topic_words = first_topic.argsort()[-10:]
        for i,topic in enumerate(LDA.components_):
            print(f'Top 10 words for topic #{i}:')
            print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
            print('\n')
        topic_values = LDA.transform(doc_term_matrix)
        topic_values.shape
        tweets_df1['Topic'] = topic_values.argmax(axis=1)
        r=[]
        for i in tweets_df1.Text:
            sid_obj = SentimentIntensityAnalyzer()
            r.append(sid_obj.polarity_scores(i)['compound'])
        tweets_df1['sentiment']=r
        tweets_df1.to_csv(x+'.csv')
        suma=tweets_df1[["Topic","Text"]].groupby(by="Topic").count()
        suma=suma.reset_index()
        suma.to_csv('suma.csv')
        plt.bar(suma.Topic, suma.Text, color ='maroon',
        width = 0.4)
        plt.bar(suma.Topic, suma.Text, color ='maroon',
        width = 0.4)
        plt.xlabel("Topic Number")
        plt.ylabel("count")
        plt.title("Distribution of Topics")
        plt.savefig('foo.png')
        dfi.export(suma, 'dataframe.png')
        text= " ".join(tweets_df1['Text'].values.tolist()).replace('\n','')
        # Build the model.
        text_model = markovify.Text(text)
        # Print five randomly-generated sentences
        print('Topic summary')
        for i in range(5):
            s.append(text_model.make_sentence())
        df=pd.DataFrame(s)
        df.to_csv('twitter_data.csv')
    return 0

def segment():
    fil=pd.read_csv('cust.csv')
    #intialize the setup
    exp_clu = setup(fil)
    # create a model
    kmeans = create_model('kmeans')
    # assign labels using trained model
    kmeans_df = assign_model(kmeans)
    sm1=kmeans_df[["p1","Label"]].groupby(by="Label").count() #output df
    sm1=sm1.reset_index()
    sm1.to_csv('customer_segments.csv')
    return 0

def newprod ():
    df=pd.read_csv('file.csv')
    data2 = df.drop(columns=['c1','c2','c3','c4','c5'])
    clusters=['c1','c2','c3','c4','c5']
    # Names of the columns that are discrete
    discrete_columns = data2.columns
    ctgan = CTGANSynthesizer(epochs=10)
    ctgan.fit(data2, discrete_columns)
    # Synthetic copy
    samples = ctgan.sample(1000)
    le = preprocessing.LabelEncoder()
    for r in data2.columns:
      if r not in clusters:
        le.fit(data2[r].values.tolist())
        data2[r]=le.transform(data2[r].values.tolist())
    X=data2
    y=df[clusters]
    clf = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
    for p in samples.columns:
        le.fit(samples[p].values.tolist())
        samples[p]=le.transform(samples[p].values.tolist())
    preds=clf.predict(samples)
    samples['l']=preds[0][:]
    return 0

def genimg(sen1='example code',is1='Telkom'):
  for i in range(len(g)):
    if is1=='Telkom':
      fty='telkom.ttf'
      log='telkom.png'
    siz=int(500/len(sen1))
    img = Image.new('RGB', (500, 300), color ='blue')
    d = ImageDraw.Draw(img)
    fnt = ImageFont.truetype(fty,siz)
    d.text((20,50), sen1,font=fnt, fill='White')
    img.save('pil_text_font.png')
    im1 = Image.open('pil_text_font.png')
    im2 = Image.open(log)
    new_image = im2.resize((150, 150))
    back_im = im1.copy()
    back_im.paste(new_image, (330, 120))
    back_im.save('generated'+str(i)+'.png', quality=95) #output
    return 0
    
 