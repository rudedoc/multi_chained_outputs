import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import classification_report

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

# Save the predictions to a DataFrame
df_pred = pd.DataFrame(y_pred, columns=mlb.classes_)
df_pred.to_csv('chained_multi_label_predictions.csv', index=False)