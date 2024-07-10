import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import classification_report, accuracy_score

# Initial System Implementation
# fetch_20newsgroups includes news articles where the articles have been classified.
# We are going to treat the News Article like Emails, the principles in terms of Multi Class Classification are the same
# First we will map the individual labels into top level categories
# We then train a classifier to predict/assign a single top-level Category

# This provides us with a fully working example of the initial multi-class classification system suggested in the CA Brief

# The second step will be to implement Chained Multi-outputs multi-label classification system:

# Define category mapping

def train_multi_class():
  category_mapping = {
      'comp.graphics': 'technology',
      'comp.os.ms-windows.misc': 'technology',
      'comp.sys.ibm.pc.hardware': 'technology',
      'comp.sys.mac.hardware': 'technology',
      'comp.windows.x': 'technology',
      'sci.crypt': 'science',
      'sci.electronics': 'science',
      'sci.med': 'science',
      'sci.space': 'science',
      'rec.autos': 'sport',
      'rec.motorcycles': 'sport',
      'rec.sport.baseball': 'sport',
      'rec.sport.hockey': 'sport',
      'talk.politics.guns': 'politics',
      'talk.politics.mideast': 'politics',
      'talk.politics.misc': 'politics',
      'alt.atheism': 'religion',
      'soc.religion.christian': 'religion',
      'talk.religion.misc': 'religion'
  }

  # Load dataset
  data = fetch_20newsgroups(subset='all', categories=category_mapping.keys(), remove=('headers', 'footers', 'quotes'))

  # Map the original target labels to new labels
  new_targets = [category_mapping[data.target_names[target]] for target in data.target]

  # Split dataset
  X_train, X_test, y_train, y_test = train_test_split(data.data, new_targets, test_size=0.25, random_state=42)

  # Vectorize text data
  vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
  X_train_transformed = vectorizer.fit_transform(X_train)
  X_test_transformed = vectorizer.transform(X_test)

  # Train classifier
  model = MultinomialNB()
  model.fit(X_train_transformed, y_train)

  # Predict and evaluate
  y_pred = model.predict(X_test_transformed)
  print(classification_report(y_test, y_pred))
  print(confusion_matrix(y_test, y_pred))

  # Map the original target labels to new labels
  mapped_labels = [category_mapping[data.target_names[target]] for target in data.target]
  original_labels = [data.target_names[target] for target in data.target]  # Get original labels

  # Create a DataFrame
  df = pd.DataFrame({
      'text': data.data,
      'original_category': original_labels,
      'mapped_category': mapped_labels
  })

  # Save the DataFrame to a CSV file
  df.to_csv('mapped_newsgroups_with_original.csv', index=False)

  # Optionally print the head of the DataFrame to verify
  print(df.head())



def train_chained_multi():
  # Load the DataFrame from the CSV file
  df = pd.read_csv('mapped_newsgroups_with_original.csv')

  # Drop rows with missing values in the 'text' column
  df.dropna(subset=['text'], inplace=True)

  # Preprocess the text data
  vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
  X = vectorizer.fit_transform(df['text'])

  # Encode original categories into binary labels
  mlb = MultiLabelBinarizer()
  y = mlb.fit_transform(df['original_category'].apply(lambda x: [x]))

  # Split the data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

  # Initialize the base classifier
  base_lr = LogisticRegression(solver='lbfgs', max_iter=1000)

  # Initialize the ClassifierChain
  chain = ClassifierChain(base_lr, order='random', random_state=42)

  # Train the classifier
  chain.fit(X_train, y_train)

  # Predict using the classifier
  y_pred = chain.predict(X_test)

  # Evaluate the model
  print(classification_report(y_test, y_pred, target_names=mlb.classes_))

  # Evaluate the performance of each individual chain
  X_test_chained = X_test
  for i, classifier in enumerate(chain.estimators_):
      y_pred_individual = classifier.predict(X_test_chained)
      accuracy = accuracy_score(y_test[:, i], y_pred_individual)
      print(f"Performance of chain {i+1} ({mlb.classes_[i]}):")
      print(f"Accuracy: {accuracy:.4f}")
      print(classification_report(y_test[:, i], y_pred_individual, target_names=[f'not_{mlb.classes_[i]}', mlb.classes_[i]]))
      if isinstance(X_test_chained, np.ndarray):
          X_test_chained = np.hstack((X_test_chained, y_pred_individual.reshape(-1, 1)))
      else:
          X_test_chained = hstack((X_test_chained, y_pred_individual.reshape(-1, 1)))

  # Save the predictions to a DataFrame
  df_pred = pd.DataFrame(y_pred, columns=mlb.classes_)
  df_pred.to_csv('chained_multi_label_predictions.csv', index=False)


train_multi_class()
train_chained_multi()
