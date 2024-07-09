# Chained Multi Outputs System for Email

MSc/PG Diploma in Artificial Intelligence (MSCAI_JAN24, PGDAI_JAN24)
Release Date: 12th June 2024
Submission Date: 10th July 2024 @ 23.55 hrs
Lecturers: Jaswinder Singh, Muslim Jameel Syed

# playground.ipynb
This is a playground notebook for exploring ideas and does not form part of the final implementation

**Develop a full working example using Design Choice 1 (Chained Multi-outputs). Implement the Chained Multi-outputs approach, ensuring that the code is clean, well-documented, and functional. Use a version control system (like Git) to track your progress. Add the instructor as a collaborator to your repository.

# Running this code
There are 2 steps split into to separate files:
- multi_class_classification_example.py
- multi_label_classification.py

Step 1:
Initialize the environment:
`source venv/bin/activate`

Install dependencies:
`pip install -r requirements.txt`


Step 2:

run `python multi_class_classification_example.py`
This script will output 1 file:
- mapped_newsgroups_with_original.csv

This file contains the articles with the original and mapped singluar classifications

Step 3:
run `python multi_label_classification.p`
This will output file: `chained_multi_label_predictions.csv`
This file contains the a vector mapping on the different labels per data point in the original dataset

The script also outputs the following:

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


