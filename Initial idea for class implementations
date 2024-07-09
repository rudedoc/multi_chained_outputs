# Initial idea for class implementations
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

# Define category mapping
CATEGORY_MAPPING = {
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

class NewsGroupProcessor:
    """
    A class to handle data loading and preprocessing steps.
    It includes methods for loading the dataset, mapping categories,
    splitting the data, transforming targets into a multi-label format,
    and vectorizing the text data.
    """
    def __init__(self, category_mapping):
        self.category_mapping = category_mapping
        self.mlb = MultiLabelBinarizer()
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

    def load_and_process_data(self):
        # Load dataset and map original target labels to new labels
        data = fetch_20newsgroups(subset='all', categories=self.category_mapping.keys(), remove=('headers', 'footers', 'quotes'))
        new_targets = [self.category_mapping[data.target_names[target]] for target in data.target]

        # Split dataset into training and testing sets
        X_train, X_test, y_train_single, y_test_single = train_test_split(data.data, new_targets, test_size=0.25, random_state=42)

        # Convert single-label to multi-label
        y_train = self.mlb.fit_transform([[label] for label in y_train_single])
        y_test = self.mlb.transform([[label] for label in y_test_single])

        # Vectorize text data
        X_train_transformed = self.vectorizer.fit_transform(X_train)
        X_test_transformed = self.vectorizer.transform(X_test)

        return X_train_transformed, X_test_transformed, y_train, y_test, data.data, new_targets, data.target

class ModelInterface:
    """
    A class that acts as a wrapper around the classifier chain.
    It includes methods for training the model, making predictions,
    and evaluating the model.
    """
    def __init__(self, base_model):
        self.base_model = base_model
        self.chain = ClassifierChain(base_model, order='random', random_state=42)

    def train(self, X_train, y_train):
        # Train the classifier chain
        self.chain.fit(X_train, y_train)

    def predict(self, X_test):
        # Predict using the classifier chain
        return self.chain.predict(X_test)

    def evaluate(self, y_test, y_pred, classes):
        # Print classification report
        print(classification_report(y_test, y_pred, target_names=classes))

def save_data_to_csv(data, original_labels, new_targets, file_name='mapped_newsgroups_with_original.csv'):
    """
    Save the dataset with original and mapped categories to a CSV file.
    """
    df = pd.DataFrame({
        'text': data,
        'original_category': original_labels,
        'mapped_category': new_targets
    })
    df.to_csv(file_name, index=False)
    print(df.head())

def main():
    # Preprocess data
    processor = NewsGroupProcessor(CATEGORY_MAPPING)
    X_train, X_test, y_train, y_test, data, new_targets, original_labels = processor.load_and_process_data()

    # Initialize and train model
    model = ModelInterface(MultinomialNB())
    model.train(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    model.evaluate(y_test, y_pred, processor.mlb.classes_)

    # Save data to CSV
    save_data_to_csv(data, original_labels, new_targets)

if __name__ == "__main__":
    main()
