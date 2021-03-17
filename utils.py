import transformers
import pandas as pd
import numpy as np
import sklearn
import csv
import matplotlib.pyplot as plt
import seaborn as sn
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
import torch
import re

# read tsv files and separate the labels from the text inputs
# the different parameters account for slight differences in how the files are structured
def read_tsv(file_path, titles=True, annotated=False, questions=False):
    labels = []
    texts = []
    labeldict = dict(zip(['DEMO', 'DISE', 'FAML', 'GOAL', 'PREG', 'SOCL', 'TRMT'], [0,1,2,3,4,5,6]))
    with open(file_path, encoding='utf8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        if questions:
            for line in tsv_reader:
                labels.append(0)
                texts.append(line[0])
        else:   
            next(tsv_reader)
            for line in tsv_reader:
                labels.append(labeldict[line[0]])
                if annotated:
                    texts.append(line[1])
                elif titles:
                    texts.append(line[1]+' '+line[2])
                else:
                    texts.append(line[2])
    return texts, labels
    
class ICHIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

#prepare data for use with PubMedBERT  
def prepare_datasets(filepaths):
    
    #split all data into texts and labels
    pre_train_texts, pre_train_labels = read_tsv(filepaths[0])
    test_texts, test_labels = read_tsv(filepaths[1])
    COVID_annotated_texts, COVID_annotated_labels = read_tsv(filepaths[2], annotated=True)
    icliniq_annotated_texts, icliniq_annotated_labels = read_tsv(filepaths[3], annotated=True)
    COVID_texts, COVID_labels = read_tsv(filepaths[4], questions=True)
    icliniq_texts, icliniq_labels = read_tsv(filepaths[5], questions=True)
    
    # create a validation dataset from the training dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(pre_train_texts, pre_train_labels, test_size=.2)
    
    # tokenize the texts with PubMedBERT's corresponding tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
  
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
    COVID_annotated_encodings = tokenizer(COVID_annotated_texts, truncation=True, padding=True, max_length=512)
    icliniq_annotated_encodings = tokenizer(icliniq_annotated_texts, truncation=True, padding=True, max_length=512)
    COVID_encodings = tokenizer(COVID_texts, truncation=True, padding=True, max_length=512)
    icliniq_encodings = tokenizer(icliniq_texts, truncation=True, padding=True, max_length=512)
    
    # create torch datasets from data
    train_dataset = ICHIDataset(train_encodings, train_labels)
    val_dataset = ICHIDataset(val_encodings, val_labels)
    test_dataset = ICHIDataset(test_encodings, test_labels)
    COVID_annotated_dataset = ICHIDataset(COVID_annotated_encodings, COVID_annotated_labels)
    icliniq_annotated_dataset = ICHIDataset(icliniq_annotated_encodings, icliniq_annotated_labels)
    COVID_dataset = ICHIDataset(COVID_encodings, COVID_labels)
    icliniq_dataset = ICHIDataset(icliniq_encodings, icliniq_labels)
    
    return train_dataset, val_dataset, test_dataset, COVID_annotated_dataset, icliniq_annotated_dataset, COVID_dataset, icliniq_dataset
 
# compute the precision, recall, and f1 for each class, as well as the mean f1 and the global accuracy   
def compute_metrics(pred):
    label_names = ['DEMO', 'DISE', 'FAML', 'GOAL', 'PREG', 'SOCL', 'TRMT']
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, labels=[0,1,2,3,4,5,6])
    acc = accuracy_score(labels, preds)
    returndict = {'accuracy': acc}
    for i in range(len(label_names)):
        returndict[label_names[i]+' f1']= f1[i]
        returndict[label_names[i]+' precision']= precision[i]
        returndict[label_names[i]+' recall']= recall[i]
    return returndict
    
    
def split_on_dialogue(data_path):
    """
    Returns list with conversations
    Format conversatoins: [[conversation1], [conversation2], ..., [conversation_n]]
    """
    
    with open(data_path) as f:
        lines = f.readlines()
        f.close()

    i = 0
    j = 0
    dialogue_i = 0
    convo=[]
    conversations=[]

    for line in lines:
        i += 1
        tokens = word_tokenize(line)


        if line[:8] == 'Dialogue':
            dialogue_i = i+1

        if i == dialogue_i+j:
            convo.append(line)
            j +=1
            if len(tokens) == 0:
                conversations.append(convo)
                convo = []
                j = 0
                continue
    return conversations
    
def save(df, save_preprocessed_dataframe_path, name):
    """
    Function that saves the created dataframe as a csv.
    """
    
    df.to_csv(save_preprocessed_dataframe_path+name+'.tsv', sep='\t', index=False)


def get_patient_questions(df):
    """
    Function that splits the dialogue between that of the patient and doctor.
    """
    df_copy = df.copy()
    questions = []

    for index in df.index:
        raw_dialogue = df_copy.loc[index, 'dialogue']
        question = re.split('Patient:|Doctor:', ' '.join(raw_dialogue))[1]
        questions.append(question)


    
    return pd.DataFrame(questions, columns=['questions'])

def preprocess_to_tsv(data_path, save_to):
    
    # Split on dialogue 
    conversations = split_on_dialogue(data_path)
    
    # Make dataframe
    df_raw = pd.DataFrame(np.array(conversations), columns=['dialogue'])
    df_questions = get_patient_questions(df_raw).drop_duplicates()
    # Save
    name = os.path.basename(data_path)[:-4]+'_questions'
    save(df_questions, save_to, name)
    
    print(name, 'done')


def make_grouped_bar_chart(covid_pred_labels, icliniq_pred_labels):

    label_names = ['DEMO', 'DISE', 'FAML', 'GOAL', 'PREG', 'SOCL', 'TRMT']
    covid_label_counts = [i[1]/len(covid_pred_labels) for i in sorted(Counter(covid_pred_labels).items())]
    icliniq_label_counts = [i[1]/len(icliniq_pred_labels) for i in sorted(Counter(icliniq_pred_labels).items())]

    x = np.arange(len(label_names))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, covid_label_counts, width, label='COVID')
    rects2 = ax.bar(x + width/2, icliniq_label_counts, width, label='icliniq')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title('percentage presence of each label by dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(label_names)
    ax.legend()

    for rects in [rects1, rects2]:
        for rect in rects:
            height = round(rect.get_height(), 2)
            ax.annotate('{}'.format(height),
               xy=(rect.get_x() + rect.get_width() / 2, height),
               xytext=(0, 3),  # 3 points vertical offset
               textcoords="offset points",
               ha='center', va='bottom')

    fig.tight_layout()

    plt.show()
    

def make_misclassification_heatmap(true, pred):
    
    label_names = ['DEMO', 'DISE', 'FAML', 'GOAL', 'PREG', 'SOCL', 'TRMT']
    named_true_labels = [label_names[i] for i in true]
    named_pred_labels =  [label_names[i] for i in pred]

    df = pd.DataFrame(np.array([named_true_labels, named_pred_labels]).T, columns=['True', 'Predicted'])

    sn.heatmap(df.value_counts(sort=False).unstack().fillna(0), annot=True, fmt='.0f')
    plt.show()
