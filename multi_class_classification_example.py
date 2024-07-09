from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Initial System Implementation
# fetch_20newsgroups includes news articles where the articles have been classified.
# We are going to treat the News Article like Emails, the principles in terms of Multi Class Classification are the same
# First we will map the individual labels into top level categories
# We then train a classifier to predict/assign a single top-level Category

# This provides us with a fully working example of the initial multi-class classification system suggested in the CA Brief

# The second step will be to implement Chained Multi-outputs multi-label classification system:

# Define category mapping
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
