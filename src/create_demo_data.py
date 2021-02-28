import pandas as pd
from nltk.tokenize import word_tokenize
from random import random

###########################################################################
# This file was only used to create some dummy data for the dashboard demo
###########################################################################

###########################################################################
# Entries were taken from my real data file, all sensitive information (name, number, etc.) was removed, and the text message contents
# were replaced with random words taken from a list of common English words.
# Any additional editing, such as removing extra long texts, or inserting extra records, was done directly in the CSV file that
# this file produced.
# All modification of the call data was done directly in the xml file. The following are the regex expressions used to select the number,
# and subscription_id columns:
# (number="\+?\d+")
# (subscription_id="\+?\d+")
###########################################################################

word_bank = []
with open('data/common_words.txt', 'r') as words:
    # ignore the comment at the start of the file, and take every fifth word to limit the number of words
    next(words)
    word_bank = word_tokenize(words.read())[::5]

print(word_bank)


def create_text_demo_data():
    """
    Put this in a function so the changes I made in the new demo data file don't get overwritten each time I run this file.
    """

    # read in data file to be turned into demo data
    df_texts = pd.read_csv('data/texts_demo_data.csv', usecols=['_readable_date', '_body', '_type'],
                           parse_dates=['_readable_date'])

    # replace each text body with random words, but keep the same length
    for index, row in df_texts.iterrows():
        length = row['_body']
        new_body = ""
        for i in range(len(row['_body'])):
            new_body += word_bank[int(random()*len(word_bank))]
            new_body += " "
        df_texts.at[index, '_body'] = new_body

    df_texts.to_csv("data/texts_demo_temp.csv")
