# PubMedBERT-Question-Classification

This repository contains code to finetune a PubMedBERT Model on a medical question classification task, and classify a number of new questions with this model. It exists as part of a project for the course "Language, Speech, and Dialogue Processing" (LSDP) from the Bachelor Kunstmatige Intelligentie at the University of Amsterdam.

## Datasets
The data used in this project is:

for finetuning:
* ICHI 2016 dataset, as found in Medical Forum Question Classification Using Deep Learning (Jalan et al., 2018). (Their code and the dataset can be found at https://tinyurl.com/medCat18)

To classify:
The first parts of the dialogues from:
* COVID-19 Dialogue Datase (during/after covid) https://www.kaggle.com/xuehaihe/covid-dialogue-dataset?select=COVID-Dialogue-Dataset-English.txt
* MedDialog Dataset (English) (before covid) https://github.com/UCSD-AI4H/Medical-Dialogue-System

## Model

To use PubMedBERT I used Huggingface's transformers library, and the version of PubMedBERT available here, on huggingface's modelhub https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

## Files
* question_analysis.ipynb: main file to finetune the model
* utils.py: various utility functions to eg. load data in a usable format, make graphs
* ds_config.json: config file for using deepspeed while finetuning
