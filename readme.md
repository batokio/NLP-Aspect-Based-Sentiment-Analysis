# Aspect-Based Sentiment Analysis

## 1.  Authors

-   Bryan ATOK A KIKI bryan.atok-a-kiki\@student-cs.fr
-   Ofir BERGER ofir.berger\@student-cs.fr
-   Sauvik CHATTERJEE sauvik.chatterjee\@student-cs.fr
-   Abhinav SINGH abhinav.singh\@student-cs.fr

## 2.  Settings

In addition to the common Python libraries, our model requires the
installation of the following packages: 
- `Transformers`
- `PyTorch`

## 3.  Preprocessing

As the BERT model performs well on a wide variety of syntax, we decided
to reduce the number of preprocessing to the following steps: 
- We first converted the list of sentiment to integer as such : 
  - 0: Negative
  - 1: Neutral 
  - 2: Positive 
- Then, based on the paper from Sun et
al. \[1\] (2019), we converted the aspect category column to meaningful
questions and concatenated them with the target term. 
Ex: 'AMBIENCE\#GENERAL' --\> "What do you think of the ambience ? seating"

On average, this preprocessing step increased the accuracy of our model
by 4%.

## 4.  Model

Our classifier model is built on the BERT pretrained model from the
Transformers library. We used the based 'bert-base-uncased' as this
model led on average to an accuracy 2% higher than the
'bert-base-cased'. This means that the case sensitivity is not relevant
for this task.

This model uses 12 layers of transformers block with a hidden size of
768 and number of self-attention heads as 12.

We replaced the last layer by a fully connected layer with 3 for the
size the output which is equal to the number of classes.

We set the initial learning rate at 2e-5 and used the Adam optimizer
with a linear learning rate scheduler. The training batch size is 32 and
we trained the model over 10 epochs.

In addition, we also used the BERTtokenizer encoding the sentence and
the processed aspect. It outputs the attention masks and the input ids
which are then used to train the model as well as the sentence.

## 5.  Performance

The accuracy of the model is 86.70% (with a standard deviation of 0.37
on 5 tests). 
Full results: \[86.17, 86.44, 86.7, 87.23, 86.97\]

## 6.  Reference

\[1\] Chi Sun, Luyao Huang, Xipeng Qiu. 2019. Utilizing BERT for Aspect-Based Sentiment Analysis via
Constructing Auxiliary Sentence. NAACL
