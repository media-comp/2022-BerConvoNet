import sys
sys.path.append('bert/')

import re
import numpy as np
import tensorflow as tf
import modeling
import tokenization
import numpy as np
import pandas as pd
import re, string, unicodedata

import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# NLP Libs
import nltk
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from string import punctuation

# PyTorch
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

strategy = tf.distribute.MirroredStrategy()

df = pd.read_csv('fake_or_real_news.csv')
df.head()
df = df[['title','text','label']]
null_df = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending = False) * 100

null_df = pd.concat([null_df, percent], axis = 1, keys = ['Counts', '% Missing'])
null_df.head()

true = df[df['label'] == 'REAL'][:2000]
fake = df[df['label'] == 'FAKE'][:2000]

true.head()

true.shape, fake.shape # relative same size good for training model

true['label'] = 1
fake['label'] = 0

df = pd.concat([true, fake],ignore_index= True).drop_duplicates()
df.sample(10)

df['text'] = df['title'] + "" + df['text']
del df['title']

nltk.download('stopwords')
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

def strip_html(text):
  # get a BeautifulSoup object that follows standard structure
  soup = BeautifulSoup(text, "html.parser")
  return soup.get_text() # extract all the text from a page

# removing the square brackets
def remove_between_square_brackets(text):
  # '' replace patterns occured in left of text like \[[^]*\]
  return re.sub(r'[[^]*\]', '', text)

# removing URL's
def remove_urls(text):
  return re.sub(r'http\S+', '', text)

# removing the stopwords from text
def remove_stopwords(text):
  final_text = []
  for i in text.split():
    if i.strip().lower() not in stop:
      final_text.append(i.strip()) # strip(None) remove blank char in default
  return " ".join(final_text)
  
# removing the noisy text
def denoise_text(text):
  text = strip_html(text)
  text = remove_between_square_brackets(text)
  text = remove_stopwords(text)
  return text

# Apply function on review column
df['text'] = df['text'].apply(denoise_text)

df.head()

# word clouds for true news
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000, width = 1600, height = 800, stopwords = STOPWORDS).generate(" ".join(df[df.label == 1].text))
plt.imshow(wc, interpolation = 'bilinear')
# show the most prominent or frequent words in a body of true news

# word clouds for fake news
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000, width = 1600, height = 800, stopwords = STOPWORDS).generate(" ".join(df[df.label == 0].text))
plt.imshow(wc, interpolation = 'bilinear')

nltk.download('wordnet')
plt.figure(figsize = (20,20))
def clean(text: str) -> list:
  'A simple function to cleanup text data'
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  text = (text.encode('ascii', 'ignore')
           .decode('utf-8', 'ignore')
           .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]

corpus = clean(' '.join(df[df.label == 1].text))

def listToString(s):
  str1 = " "
  return (str1.join(s))

corpus_str = listToString(corpus)
#d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
def transform_format(val):
  if val.any() == 0:
    return 255
  else:
    return val

#coloring = np.array(Image.open(path.join(d, "/content/gdrive/MyDrive/Daco_115064.png")))
coloring = np.array(Image.open("REAL_Word_Mark_Green_Logo.jpg"))

stoplist = set(STOPWORDS)
wc = WordCloud(background_color = "white", max_words = 2000, width = 1600, height = 800,mask = coloring, stopwords = stoplist, max_font_size = 100)

wc.generate(corpus_str)

image_colors = ImageColorGenerator(coloring)
plt.imshow(wc, interpolation = 'bilinear')
plt.axis("off")

fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig("real_new_word_cloud.png", bbox_inches = 'tight', dpi = 600)

nltk.download('wordnet')
plt.figure(figsize = (20,20))
def clean(text: str) -> list:
  'A simple function to cleanup text data'
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  text = (text.encode('ascii', 'ignore')
           .decode('utf-8', 'ignore')
           .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]

corpus = clean(' '.join(df[df.label == 0].text))

def listToString(s):
  str1 = " "
  return (str1.join(s))

corpus_str = listToString(corpus)
#d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
def transform_format(val):
  if val.any() == 0:
    return 255
  else:
    return val

#coloring = np.array(Image.open(path.join(d, "/content/gdrive/MyDrive/Daco_115064.png")))
coloring = np.array(Image.open("tumblr_mkfgk4gtzO1qajmbto1_1280.jpg"))

stoplist = set(STOPWORDS)
wc = WordCloud(background_color = "white", max_words = 2000, width = 1600, height = 800,mask = coloring, stopwords = stoplist, max_font_size = 100)

wc.generate(corpus_str)

image_colors = ImageColorGenerator(coloring)
plt.imshow(wc, interpolation = 'bilinear')
plt.axis("off")

fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig("real_new_word_cloud.png", bbox_inches = 'tight', dpi = 600)

# Word Embedding using BERT
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], stratify = df['label'])

max_len =  128
train_batch_size = 16
valid_batch_size = 16
epochs = 3
learning_rate = 1e-5
filters = 50
filter_sizes = [2, 3 ,4 ,5]
hidden_size = 768
dropout = 0.1

from transformers import BertTokenizer, BertModel, BertConfig
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
x_train_tokens = tokenizer.batch_encode_plus(
    x_train.tolist(),
    add_special_tokens = True,
    max_length = max_len,
    padding = 'max_length',
    truncation = True,
    return_attention_mask = True,
    return_tensors = 'pt'
)

x_test_tokens = tokenizer.batch_encode_plus(
    x_test.tolist(),
    add_special_tokens = True,
    max_length = max_len,
    padding = 'max_length',
    truncation = True,
    return_attention_mask = True,
    return_tensors = 'pt'
)




x_train_tokens, x_test_tokens
y_train_2d = [[x] for x in y_train.tolist()]
y_test_2d = [[x] for x in y_test.tolist()]

y_train_tensor = torch.tensor(y_train_2d)
y_test_tensor = torch.tensor(y_test_2d)

#DataLoaders -> data become iteral
batch_size = 16

train_data = TensorDataset(x_train_tokens['input_ids'], x_train_tokens['token_type_ids'], x_train_tokens['attention_mask'], y_train_tensor)
train_data_sampler = RandomSampler(train_data)
train_data_loader = DataLoader(train_data, sampler = train_data_sampler, batch_size = batch_size)

test_data = TensorDataset(x_test_tokens['input_ids'], x_test_tokens['token_type_ids'], x_test_tokens['attention_mask'], y_test_tensor)
test_data_sampler = RandomSampler(test_data)
test_data_loader = DataLoader(test_data, sampler = test_data_sampler, batch_size = batch_size)

class bert_cnn_model(nn.Module):
 def __init__(self, model):
   super(bert_cnn_model, self).__init__()
   model_config = BertConfig.from_pretrained('bert-base-uncased', return_dict = False, output_hidden_states=True)
   self.bert = BertModel.from_pretrained('bert-base-uncased', config = model_config)
   for param in self.bert.parameters():
     param.requires_grad = True #update data
   self.convs = nn.ModuleList([nn.Conv1d(1, filters, (k, hidden_size)) for k in filter_sizes])
   self.dropout = nn.Dropout(dropout)
   self.fc = nn.Linear(filters * len(filter_sizes), 2)

 def conv_and_pool(self, x, conv):
   x = conv(x) #[batch_size,channel_num,pad_size,embedding_size（1）]
   x = nn.ReLU(x) 
   x = x.squeeze(3) #[batch_size,channel_num,pad_size]
   x = nn.MaxPool1d(x, x.size(2))
   x = x.squeeze(2) # [batch_size,channel_num]
   return x

 def forward(self, x):
   context = x[0]
   mask = x[2]

   encoder_out, pooled = self.bert(context, attention_mask = mask, output_all_encoded_layers=False)
   out = torch.cat([self.conv_and_pool(encoder_out,conv) for conv in self.convs],1)
   out = self.fc(out)
   return out
   
bc_model = bert_cnn_model(nn.Module)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bc_model.to(device)

#loss function
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes = np.unique(y_train), y = y_train)
weights = torch.tensor(class_weights, dtype = torch.float)
weights = weights.to(device)

cross_entropy = nn.NLLLoss(weight = weights)

# class_weights
#optimizer
optimizer = torch.optim.Adam(params = bc_model.parameters(), lr = learning_rate)

#Train model

def train_model(
    n_epochs,
    training_loader,
    validation_loader,
    model, 
    optimizer
  ):

  valid_loss_min = np.Inf

  for epoch in range(1, n_epochs + 1):
      print("Epoch {:} / {:}".format(epoch, epochs))
      train_loss, valid_loss = 0, 0
      avg_train_accuracy, avg_valid_accuracy = 0, 0
      accuracy = 0
      model.train()
      train_predictions = []
      valid_predictions = []

      #training loop
      for idx, batch in enumerate(training_loader):
          batch = [r.to(device) for r in batch]
          input_ids, attention_masks, labels = batch
          outputs = model(batch)
          model.zero_grad()  #clear previously calculated gradients
          optimizer.zero_grad()
          loss = cross_entropy(outputs, labels.squeeze(1))
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          train_loss = train_loss + (1/(idx + 1) * (loss.item() - train_loss))
          outs = outputs.detach().cpu().numpy()
          train_predictions.append(outs)
          predictions = torch.argmax(outputs, dim = 1)
          accuracy += sum([1 if predictions[i] == labels[i] else 0 for i in range(len(labels))])
          
      avg_train_accuracy = accuracy / len(training_loader.dataset)
      print(f'\Training Accuracy: {avg_train_accuracy}')
      print(f'\Training Loss: {train_loss: .f}')

      #validation loop
      accuracy = 0
      model.eval()
      with torch.no_grad():
        for idx, bacth in enumerate(validation_loader):
            batch = [r.to(device) for r in batch]
            input_ids, attention_masks, labels = batch
            outputs = model(batch)
            loss = cross_entropy(outputs, labels.squeeze(1))
            valid_loss = valid_loss + (1/(idx + 1) * (loss.item() - valid_loss))
            outs = outputs.detach().cpu().numpy()
            valid_predictions.append(outs)
            predictions = torch.argmax(outputs, dim = 1)
            accuracy += sum([1 if predictions[i] == labels[i] else 0 for i in range(len(labels))])
            
      if valid_loss < valid_loss_min:
          valid_loss_min = valid_loss
          torch.save(model.state_dict(), 'saved_wrights.pt')

      avg_valid_accuracy = accuracy / len(validation_loader.dataset)
      print(f'\Validation Accuracy: {avg_valid_accuracy}')
      print(f'\Validation Loss: {valid_loss: .f}')
      #print(f'roc_auc score: {roc_auc_score(y_train_tensor, train_predictions)}')
      #print(f'F1 score: {f1_score(y_train_tensor, train_predictions)}')

  return bert_model

train_model = train_model(epochs, train_data_loader, test_data_loader, bc_model, optimizer)
