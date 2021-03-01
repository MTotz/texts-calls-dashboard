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

# For the call data, only the calls with contact "John" were kept, as well as the first and last lines of the XML tree.
# The following regex expressions were used to find and change all phone numbers and subscription ids.
# (number="\+?\d+")
# (subscription_id="\+?\d+")
###########################################################################


def create_text_demo_data():
    """
    Put this in a function so the changes I made in the new demo data file don't get overwritten each time I run this file.
    """
    word_bank = []

    with open('data/common_words.txt', 'r') as words:
        # ignore the comment at the start of the file, and take every fifth word to limit the number of words
        next(words)
        word_bank = word_tokenize(words.read())[::5]

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


def create_call_demo_data():
    with open('../data/calls_demo_data.xml', 'r') as file, open('../data/calls_demo_temp.xml', 'w') as temp_file:
        for line in file:
            # line that contains "archive" is the firs line of the XML tree, "John" are the line for the call data
            # "</calls>" is the last line of the file
            if "archive" in line or "John" in line or "</calls>" in line:
                temp_file.write(line)


create_call_demo_data()
