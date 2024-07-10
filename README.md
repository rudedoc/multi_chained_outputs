# Chained Multi Outputs System

MSc/PG Diploma in Artificial Intelligence (MSCAI_JAN24, PGDAI_JAN24)
Release Date: 12th June 2024
Submission Date: 10th July 2024 @ 23.55 hrs
Lecturers: Jaswinder Singh, Muslim Jameel Syed

# Description
We are addressing the problem of updating a Multi-Class Classification system to a Chained Multi-outputs multi-label classification system.

The idea proposed is to use the fetch_20newsgroups dataset as a stand-in replacement for a dataset of emails.

The fetch_20newsgroups dataset resembles email data and has an initial set of labels annotated to it. We first mapped the individual labels into a set of super categories:

```
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
```
In the `multi_class_classification_example.py` script, we trained a model that would map the email dataset to the super categories above. This implements the initial Multi-Class Classification system as outlined in the CA.

In the `multi_label_classification.py` script, we implemented a Chained Multi-outputs multi-label classification system, which predicts and maps the data back to the original set of labels.

We acknowledge that this is a circular approach, but it achieves the requirements of the CA.

The next step is to reimplement/refactor this into a set of classes that define the various interfaces between different parts of the system. This will help modularize the code and make it more maintainable and scalable.

## playground.ipynb
This is a playground notebook for exploring ideas and does not form part of the final implementation.

Develop a full working example using Design Choice 1 (Chained Multi-outputs). Implement the Chained Multi-outputs approach, ensuring that the code is clean, well-documented, and functional. Use a version control system (like Git) to track your progress. Add the instructor as a collaborator to your repository.

# Running this code
There are 2 steps split into to separate files:
- multi_class_classification_example.py
- multi_label_classification.py

## Step 1:
Initialize the environment:
`source .venv/bin/activate`

Install dependencies:
`pip install -r requirements.txt`

## Step 2:
run `python multi_class_classification_example.py`
This script will output 1 file:
- mapped_newsgroups_with_original.csv

This file contains the articles with the original and mapped singluar classifications

## Step 3:
run `python multi_label_classification.py`
This will output file: `chained_multi_label_predictions.csv`
This file contains the a vector mapping on the different labels per data point in the original dataset

# UML component diagram
```
  +---------------------------+
  | Data Preparation Component|
  |---------------------------|
  | - CSV File                |
  | - Pandas                  |
  | - DataFrame               |
  | - Train-Test Split        |
  +---------------------------+
            |
            v
  +---------------------------+
  | Text Vectorization        |
  |---------------------------|
  | - TfidfVectorizer         |
  +---------------------------+
            |
            v
  +---------------------------+
  | Multi-label Encoding      |
  |---------------------------|
  | - MultiLabelBinarizer     |
  +---------------------------+
            |
            v
  +---------------------------+
  | Model Training and        |
  | Prediction Component      |
  |---------------------------|
  | - LogisticRegression      |
  | - ClassifierChain         |
  +---------------------------+
            |
            v
  +---------------------------+
  | Evaluation                |
  |---------------------------|
  | - Classification Report   |
  +---------------------------+
            |
            v
  +---------------------------+
  | Output                    |
  |---------------------------|
  | - Predictions DataFrame   |
  | - CSV File                |
  +---------------------------+
```

# Performance
```
                          precision    recall  f1-score   support

             alt.atheism       0.86      0.12      0.21       211
           comp.graphics       0.89      0.29      0.43       255
 comp.os.ms-windows.misc       0.76      0.49      0.60       249
comp.sys.ibm.pc.hardware       0.79      0.34      0.48       236
   comp.sys.mac.hardware       0.83      0.59      0.69       219
          comp.windows.x       0.94      0.39      0.55       247
               rec.autos       0.10      0.99      0.19       253
         rec.motorcycles       0.98      0.52      0.68       244
      rec.sport.baseball       0.97      0.38      0.54       243
        rec.sport.hockey       0.96      0.64      0.77       234
               sci.crypt       0.90      0.64      0.74       254
         sci.electronics       1.00      0.13      0.23       238
                 sci.med       0.99      0.53      0.69       230
               sci.space       0.98      0.38      0.55       230
  soc.religion.christian       0.59      0.79      0.67       243
      talk.politics.guns       0.80      0.26      0.39       230
   talk.politics.mideast       0.97      0.49      0.65       203
      talk.politics.misc       0.93      0.13      0.23       197
      talk.religion.misc       0.62      0.05      0.09       159

               micro avg       0.44      0.44      0.44      4375
               macro avg       0.83      0.43      0.49      4375
            weighted avg       0.83      0.44      0.50      4375
             samples avg       0.44      0.44      0.44      4375
```

```
Performance of chain 1 (alt.atheism):
Accuracy: 0.9566
                 precision    recall  f1-score   support

not_alt.atheism       0.96      1.00      0.98      4164
    alt.atheism       0.86      0.12      0.21       211

       accuracy                           0.96      4375
      macro avg       0.91      0.56      0.59      4375
   weighted avg       0.95      0.96      0.94      4375
```
```
Performance of chain 2 (comp.graphics):
Accuracy: 0.9200
                   precision    recall  f1-score   support

not_comp.graphics       0.94      0.98      0.96      4120
    comp.graphics       0.04      0.02      0.02       255

         accuracy                           0.92      4375
        macro avg       0.49      0.50      0.49      4375
     weighted avg       0.89      0.92      0.90      4375
```
```
Performance of chain 3 (comp.os.ms-windows.misc):
Accuracy: 0.9360
                             precision    recall  f1-score   support

not_comp.os.ms-windows.misc       0.94      0.99      0.97      4126
    comp.os.ms-windows.misc       0.00      0.00      0.00       249

                   accuracy                           0.94      4375
                  macro avg       0.47      0.50      0.48      4375
               weighted avg       0.89      0.94      0.91      4375
```
```
Performance of chain 4 (comp.sys.ibm.pc.hardware):
Accuracy: 0.9278
                              precision    recall  f1-score   support

not_comp.sys.ibm.pc.hardware       0.95      0.98      0.96      4139
    comp.sys.ibm.pc.hardware       0.01      0.00      0.01       236

                    accuracy                           0.93      4375
                   macro avg       0.48      0.49      0.48      4375
                weighted avg       0.89      0.93      0.91      4375
```
```
Performance of chain 5 (comp.sys.mac.hardware):
Accuracy: 0.9282
                           precision    recall  f1-score   support

not_comp.sys.mac.hardware       0.95      0.98      0.96      4156
    comp.sys.mac.hardware       0.00      0.00      0.00       219

                 accuracy                           0.93      4375
                macro avg       0.47      0.49      0.48      4375
             weighted avg       0.90      0.93      0.91      4375
```
```
Performance of chain 6 (comp.windows.x):
Accuracy: 0.9202
                    precision    recall  f1-score   support

not_comp.windows.x       0.94      0.98      0.96      4128
    comp.windows.x       0.00      0.00      0.00       247

          accuracy                           0.92      4375
         macro avg       0.47      0.49      0.48      4375
      weighted avg       0.89      0.92      0.90      4375
```
```
Performance of chain 7 (rec.autos):
Accuracy: 0.9186
               precision    recall  f1-score   support

not_rec.autos       0.94      0.98      0.96      4122
    rec.autos       0.00      0.00      0.00       253

     accuracy                           0.92      4375
    macro avg       0.47      0.49      0.48      4375
 weighted avg       0.89      0.92      0.90      4375
```
```
Performance of chain 8 (rec.motorcycles):
Accuracy: 0.9239
                     precision    recall  f1-score   support

not_rec.motorcycles       0.94      0.98      0.96      4131
    rec.motorcycles       0.00      0.00      0.00       244

           accuracy                           0.92      4375
          macro avg       0.47      0.49      0.48      4375
       weighted avg       0.89      0.92      0.91      4375
```
```
Performance of chain 9 (rec.sport.baseball):
Accuracy: 0.9273
                        precision    recall  f1-score   support

not_rec.sport.baseball       0.94      0.98      0.96      4132
    rec.sport.baseball       0.00      0.00      0.00       243

              accuracy                           0.93      4375
             macro avg       0.47      0.49      0.48      4375
          weighted avg       0.89      0.93      0.91      4375
```
```
Performance of chain 10 (rec.sport.hockey):
Accuracy: 0.9401
                      precision    recall  f1-score   support

not_rec.sport.hockey       0.95      0.99      0.97      4141
    rec.sport.hockey       0.00      0.00      0.00       234

            accuracy                           0.94      4375
           macro avg       0.47      0.50      0.48      4375
        weighted avg       0.90      0.94      0.92      4375
```
```
Performance of chain 11 (sci.crypt):
Accuracy: 0.9056
               precision    recall  f1-score   support

not_sci.crypt       0.94      0.96      0.95      4121
    sci.crypt       0.01      0.00      0.00       254

     accuracy                           0.91      4375
    macro avg       0.47      0.48      0.48      4375
 weighted avg       0.89      0.91      0.90      4375
```
```
Performance of chain 12 (sci.electronics):
Accuracy: 0.9099
                     precision    recall  f1-score   support

not_sci.electronics       0.94      0.96      0.95      4137
    sci.electronics       0.00      0.00      0.00       238

           accuracy                           0.91      4375
          macro avg       0.47      0.48      0.48      4375
       weighted avg       0.89      0.91      0.90      4375
```
```
Performance of chain 13 (sci.med):
Accuracy: 0.9445
              precision    recall  f1-score   support

 not_sci.med       0.95      1.00      0.97      4145
     sci.med       0.00      0.00      0.00       230

    accuracy                           0.94      4375
   macro avg       0.47      0.50      0.49      4375
weighted avg       0.90      0.94      0.92      4375
```
```
Performance of chain 14 (sci.space):
Accuracy: 0.9122
               precision    recall  f1-score   support

not_sci.space       0.95      0.96      0.95      4145
    sci.space       0.01      0.00      0.01       230

     accuracy                           0.91      4375
    macro avg       0.48      0.48      0.48      4375
 weighted avg       0.90      0.91      0.90      4375
```
```
Performance of chain 15 (soc.religion.christian):
Accuracy: 0.9163
                            precision    recall  f1-score   support

not_soc.religion.christian       0.94      0.97      0.96      4132
    soc.religion.christian       0.00      0.00      0.00       243

                  accuracy                           0.92      4375
                 macro avg       0.47      0.49      0.48      4375
              weighted avg       0.89      0.92      0.90      4375
```
```
Performance of chain 16 (talk.politics.guns):
Accuracy: 0.9182
                        precision    recall  f1-score   support

not_talk.politics.guns       0.95      0.97      0.96      4145
    talk.politics.guns       0.00      0.00      0.00       230

              accuracy                           0.92      4375
             macro avg       0.47      0.48      0.48      4375
          weighted avg       0.90      0.92      0.91      4375
```
```
Performance of chain 17 (talk.politics.mideast):
Accuracy: 0.9127
                           precision    recall  f1-score   support

not_talk.politics.mideast       0.95      0.96      0.95      4172
    talk.politics.mideast       0.01      0.00      0.01       203

                 accuracy                           0.91      4375
                macro avg       0.48      0.48      0.48      4375
             weighted avg       0.91      0.91      0.91      4375
```
```
Performance of chain 18 (talk.politics.misc):
Accuracy: 0.8816
                        precision    recall  f1-score   support

not_talk.politics.misc       0.95      0.92      0.94      4178
    talk.politics.misc       0.01      0.02      0.02       197

              accuracy                           0.88      4375
             macro avg       0.48      0.47      0.48      4375
          weighted avg       0.91      0.88      0.90      4375
```
```
Performance of chain 19 (talk.religion.misc):
Accuracy: 0.4526
                        precision    recall  f1-score   support

not_talk.religion.misc       0.96      0.45      0.61      4216
    talk.religion.misc       0.03      0.49      0.06       159

              accuracy                           0.45      4375
             macro avg       0.50      0.47      0.34      4375
          weighted avg       0.93      0.45      0.59      4375
```