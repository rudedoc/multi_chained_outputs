import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import classification_report, accuracy_score

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
