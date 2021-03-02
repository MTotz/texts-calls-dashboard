#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on sat May 02 16:42 2020

@author: Michael
"""

# text files are saved using SMS backup app on my phone as xml file, then converted to csv file via
# http://www.convertcsv.com/xml-to-csv.htm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as spec
from nltk.tokenize import word_tokenize, regexp_tokenize
from nltk.corpus import stopwords
from collections import Counter

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LabelSet, Title, Label, LinearAxis, Range1d, Line
from bokeh.models.widgets import Tabs, Panel, Select, DataTable, TableColumn
from bokeh.layouts import column, row, gridplot
from bokeh.transform import cumsum

from math import pi

person1 = "Mary"
person2 = "John"


def text_messages(filename, resample):
    """
    Converts text message csv file to DataFrame and creates the following plots: line
    plot of text messages sent vs. date, line plot of average text message length vs.
    date, pie chart of number of texts sent by each person, table listing 10 most common
    words.
    Input: Absolute path file of the csv file; the resampling time interval string.
    Returns: Bokeh Panel to be used in HTML display.

    Puts all of the below functions together into one for convenience.
    """

    df = create_text_dataframe(filename)[::4]
    return text_message_plots(df, resample)


def create_text_dataframe(file_path):
    """
    Process text message csv file into initial DataFrame.
    Input: The csv file path as a string.
    Returns: The resulting DataFrame.

    This function (and, by consequence, this whole program, depends on the format
    of the csv file. The text messages are saved to an xml file with the SMS
    backup app on my phone, and then converted to csv file via the website
    'http://www.convertcsv.com/xml-to-csv.htm'.

    Creates a DataFrame with columns describing the text message body ('Body') and
    the sender of the message ('Sender'), with the text's date and time of sending
    as a DateTime Index as the DataFrame's index.
    """

    df = pd.read_csv(file_path, usecols=['_readable_date', '_body', '_type'],
                     parse_dates=['_readable_date'])
    # rename column, inplace argument modifies existing array
    df.rename(columns={'_body': 'Body', '_type': 'Sender',
                       '_readable_date': 'Date'}, inplace=True)
    # convert Body column to string type to be able to edit
    df['Body'] = df['Body'].astype(str)
    # create new columns; 1 if that text was sent by that person, 0 otherwise
    df[person1] = df['Sender'].apply(
        lambda x: 1 if x == 1 else np.nan)
    df[person2] = df['Sender'].apply(
        lambda x: 1 if x == 2 else np.nan)
    df = df.drop(['Sender'], axis='columns')  # drop the Sender column

    df.set_index('Date', inplace=True)  # set dates to be dataframe index
    return df


def texts_sent_per_time(df, resample):
    """
    Creates a DataFrame of number of texts over resample time interval by each
    sender and in total.
    Input: The DataFrame to be resampled; the resampling time interval as a string.
    Returns: The resampled DataFrame.

    The DataFrame is resampled by taking the sum of each column, thereby giving the
    number of texts sent by each person per time interval. A new column is
    created for the total number of texts sent per day by adding these two
    columns together. The input DataFrame is not altered.
    """

    df = df.drop(['Body'], axis='columns')  # drop the Body column
    # dataframe has not been resampled yet so the total per row is just...
    df['Total'] = 1
    # ... Person 1 + Person 2 = 1
    # resample to get resampled sum of each columns
    texts = df.resample(resample).sum()
    # fill any missing dates with 0 value entries
    texts = texts.reindex(texts.index)

    cum_total = [texts.iloc[0, 2]]

    for i in range(1, texts['Total'].count()):
        cum_total.append(texts.iloc[i, 2] + cum_total[i - 1])
    texts['Cum_Total'] = cum_total

    return texts


def text_pre_processing(df, sender):
    """
    Preprocesses the text messages. This involves: tokenizing,
    converting all words to lowercase, removing English stop words, removing words that
    are not strictly alphabetical.
    Note: This does not take into account typos, abbreviations, slang, etc. in the texts.

    Input: The DataFrame containing the text messages; the sender whose texts are to be
    pre-processed, as a column name (string).

    Returns: List of final words.
    """

    # tokenize words composed only of upper and lowercase letters and apostrophes for...
    # ...the texts sent by the input sender
    pattern = r"\w+\'\w+|\w+"
    words = [regexp_tokenize(text, pattern)
             for text in df[df[sender] == 1]['Body']]
    # convert all words to lowercase
    lower_case = [word.lower() for l in words for word in l]
    # this no_stops line is very slow
    extra_stop_words = ["she's", "how's", "it's",
                        "he's", "that's", "what's", "who's", "there's"]
    no_stops = [word for word in lower_case if word not in stopwords.words('english')
                and word not in extra_stop_words]  # ignores stop words

    return no_stops


def text_length(df1, resample):
    """
    Calculates the average total text length and the average text length for each
    person over the resample time interval. Input DataFrame is altered.
    Input: DataFrame of texts to be resample; the resampling time interval as string.
    Returns: Resampled DataFrame

    Does not count actual words, but tokenizes the text message by whitespace, so the
    actual word count is overestimated.
    """

    df = df1.copy()

    df['Length'] = df['Body'].apply(lambda x: len(
        word_tokenize(x)))  # tokenize text by spaces
    # create new columns of the text length for each person; will be text length if sent...
    # ...by sender, otherwise will be 0
    df[person1 + '_Length'] = df['Length'].multiply(df[person1])
    df[person2 + '_Length'] = df['Length'].multiply(df[person2])
    # resample with mean over input time interval
    df = df.resample(resample).mean()
    # drop the senders columns
    df = df.drop([person1, person2], axis='columns')
    # reindex dataframe to fill missing day records...
    df = df.reindex(df.index)
    # ...with null values NaN

    return df


def text_message_plots(df, resample):
    """
    Plots the line plots of number of texts sent vs. date, average text length per
    date, and number of texts sent pie chart and table of x most common words.

    Input: Original DataFrame from original csv file (ultimately, the DataFrame
    returned by the create_text_dataframe() function above); the resampling time
    interval as a string.

    Returns: Bokeh panel to be used in HTML display.
    """

    temp = texts_sent_per_time(df, 'D')
    avg_texts_per_day = temp['Total'].mean()

    avg_words_person2, avg_words_person1 = 0, 0

    def create_source1(resample):
        data1 = texts_sent_per_time(df, resample)
        return data1

    def create_source2(resample):
        data2 = text_length(df, resample)
        return data2

    texts_per_time = create_source1(resample)
    source1 = ColumnDataSource(data=texts_per_time)

    # series of total texts sent over entire...
    total_texts = texts_per_time.sum()
    # ...elapsed time by each sender and together
    lengths = create_source2(resample)
    avg_words_person2, avg_words_person1 = lengths[person2 + '_Length'].mean(
    ), lengths[person1 + '_Length'].mean()
    source2 = ColumnDataSource(data=lengths)

    resample_strings = resample_interval_string(resample)

    plot_height, plot_width = 350, 900  # set the desired plot dimensions

    # create hover tool for first line plot to show date and texts sent by each person
    # create line plot of number of texts sent vs date
    line1 = figure(x_axis_type='datetime',  # plot datetime objects on x axis
                   y_axis_label='Number of texts sent per ' + \
                   resample_strings[0],
                   tools='pan,box_zoom,reset', plot_width=plot_width, plot_height=plot_height,
                   y_range=(0, texts_per_time[person1].max() + 5))
    texts_per_time_glyph = line1.line(x='Date', y=person1, source=source1,  # plot Person 1 texts in red
                                     color='red', alpha=0.6, line_width=2, legend_label=person1)
    line1.line(x='Date', y=person2, source=source1,  # plot Person 2 texts in blue
               color='green', alpha=0.6, line_width=2, legend_label=person2)
    # line1.y_range = Range1d(start=0, end=600)#end=texts_per_time[person1].max() + 50)
    num_texts_sent_text = (
        'Average of ' + str(round(avg_texts_per_day, 1)) + ' texts sent per day.')
    line1.title.text = 'Average of ' + \
        '{:1.2f}'.format(texts_per_time["Total"].mean()) + \
        ' texts per ' + resample_strings[0]

    line1.extra_y_ranges = {'cumulative': Range1d(
        start=0, end=texts_per_time.iloc[-1, -1] + 2000)}

    cum_axis = LinearAxis(y_range_name='cumulative',
                          axis_label='Cumulative number of texts')
    line1.add_layout(cum_axis, 'right')
    line1.varea(x='Date', y2='Cum_Total', source=texts_per_time,  # area plot of total texts sent
                color='blue', alpha=0.07, legend_label='Cumulative Total', y_range_name='cumulative')
    line1.legend.background_fill_alpha = 0.4

    hover_line1 = HoverTool(tooltips=[('Date', '@Date{%F}'), ('Texts sent by ' + person1, '@' + person1),
                                      ("Texts sent by " + person2, '@' + person2)],
                            formatters={'@Date': 'datetime'}, renderers=[texts_per_time_glyph], mode="vline")
    line1.add_tools(hover_line1)

    # create line plot of average words per text vs. date
    line2 = figure(x_axis_type='datetime', y_axis_label='Average words per text',
                   tools='pan,box_zoom,reset', plot_width=plot_width, plot_height=plot_height)
    words_per_text_glyph = line2.line(x='Date', y=person1 + '_Length', source=source2,  # plot avg Person 1 text length
                                      color='red', alpha=0.6, line_width=2, legend_label=person1)
    line2.line(x='Date', y=(person2 + '_Length'), source=source2,  # plot avg Person 2 text length
               color='green', alpha=0.6, line_width=2, legend_label=person2)
    avg_text_length_text = (person1 + ': ' + str(round(avg_words_person1, 1)) + ' words per text.\t\t\t\t\t\t\t' +
                            person2 + ': ' + str(round(avg_words_person2, 1)) + ' words per text.')
    line2.title.text = avg_text_length_text
    line2.legend.background_fill_alpha = 0.4
    line2.y_range.start = 0

    hover_line2 = HoverTool(
        tooltips=[('Date', '@Date{%F}')], formatters={'@Date': 'datetime'}, renderers=[words_per_text_glyph], mode="vline")
    line2.add_tools(hover_line2)

    # create new dataframe with number of texts sent by each person to plot in pie chart
    total_texts = pd.DataFrame(total_texts)
    total_texts = total_texts.drop(index=['Total', 'Cum_Total'])
    total_texts = total_texts.rename(columns={0: 'count'})
    # add column with angle of pie chart wedge
    total_texts['angle'] = total_texts['count'] / \
        total_texts['count'].sum() * 2 * pi
    total_texts['percent'] = total_texts['count'] / \
        total_texts['count'].sum() * 100
    total_texts['color'] = ['red', 'green']  # colors of wedges

    # hover tool to display total texts sent by each person
    pie_tooltips = """
        <div style="width:200px;">
            <span style="color:#5DAED9">Texts sent: </span><span>@count (@percent%)</span>
        </div>
    """
    hover_pie = HoverTool(
        tooltips=pie_tooltips)
    # plot pie chart by creating a wedge for each sender
    pie = figure(plot_width=int(plot_width/3),
                 plot_height=plot_height, tools=[hover_pie])
    # create a wedge for each row in the total_texts dataframe
    pie.wedge(x=0, y=1, radius=0.8,
              start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
              line_color='white', color='color', alpha=0.6, legend_field='index', source=total_texts)
    pie.axis.visible = False # remove axes
    pie.grid.grid_line_color = None # remove gridlines
    pie.title.text = 'Number of texts sent\n(Total of ' + \
        str(int(total_texts['count'].sum())) + ')'
    pie.legend.background_fill_alpha = 0.4

    line1.x_range = line2.x_range # link x axes of both line plots

    menu = Select(options=['Daily', 'Weekly', 'Monthly'],
                  value='Daily', title='Resampling frequency')

    def callback(attr, old, new):
        if menu.value == 'Daily':
            texts_per_time = create_source1('D')
            lengths = create_source2('D')
            resample_strings = resample_interval_string('D')
        elif menu.value == 'Monthly':
            texts_per_time = create_source1('M')
            lengths = create_source2('M')
            resample_strings = resample_interval_string('M')
        else:
            texts_per_time = create_source1('W')
            lengths = create_source2('W')
            resample_strings = resample_interval_string('W')

        source1.data = texts_per_time
        source2.data = lengths
        # series of total texts sent over entire...
        total_texts = texts_per_time.sum()
        # ...elapsed time by each sender and together
        line1.yaxis.axis_label = 'Number of texts sent per ' + \
            resample_strings[0]
        line1.y_range.end = texts_per_time[person1].max() + 50
        # change back the title of the right cumulative
        cum_axis.axis_label = 'Cumulative number of texts'
        # ... axis, since the above callback changes it back to be the same as the left
        line1.title.text = 'Average of ' + \
            '{:1.2f}'.format(texts_per_time["Total"].mean()) + \
            ' texts per ' + resample_strings[0]

        avg_text_length_text = (person1 + ': ' + str(round(avg_words_person1, 1)) + ' words per text.\t\t\t\t\t\t\t' +
                                person2 + ': ' + str(round(avg_words_person2, 1)) + ' words per text.')
        line2.title.text = avg_text_length_text

    menu.on_change('value', callback)

    '''
    # table of n most common words
    # use second row and last column of subplots
    # NOTE: adding this table greatly slows down the creation of the dashboard
    common_words_person2 = most_common_words(df, person2)
    common_words_person1 = most_common_words(df, person1)
    common_words_df = pd.DataFrame(
        {person1: common_words_person1, person2: common_words_person2})
    common_words_df.index += 1

    common_words_source = ColumnDataSource(common_words_df)

    columns = [TableColumn(field=person1, title=person1),
               TableColumn(field=person2, title=person2), ]
    common_words_table = DataTable(source=common_words_source, columns=columns, width=200, height=280,
                                   index_position=None, align='center')  # hides the index column in the table

    '''

    '''
    # transpose the common words to display as one column per person
    common_words_table = plt.subplot(grid[1, 0])
    cell_text = np.transpose([common_words_person1, common_words_person2])
    table = plt.table(cellText=cell_text, colLabels=['person1', person2], loc='center',
                      cellLoc='center', fontsize=14)
    # need this otherwise font size doesn't change
    table.auto_set_font_size(False)
    # change the colors and transparencies of the header cells
    table[(0,0)].set_facecolor('r')
    table[(0,0)].set_alpha(0.6)
    table[(0,1)].set_facecolor('g')
    table[(0,1)].set_alpha(0.6)

    plt.axis('off') # hide the x and y axes for a plot

    plt.suptitle('Resampling interval: ' + resample_strings[1])
    # add spacing between table and text length plot
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()
    '''

    layout = column(row(line1, pie), row(line2, menu),
                    row())  # create plot layout
    panel = Panel(child=layout, title='Texts')

    return layout


def most_common_words(df, sender):
    """
    Preprocesses the text messages by the input sender and calculates the frequency of
    each word.
    Input: The number of most common words to be extracted as int; the sender as the DataFrame
    column name (string).
    Returns: List of strings, with the word and their frequency.
    """

    count = Counter(text_pre_processing(df, sender))

    words = []
    for word in count.most_common(10):
        words.append(word[0] + ': ' + str(word[1]))

    return words


def resample_interval_string(resample):
    """
    Returns the string equivalents of the resample string for the purposes of figure labels.
    Input: Resample time interval string.
    Returns: The resample interval's corresponding words.
    """

    if resample == 'D':
        return ['day', 'daily']
    elif resample == 'W':
        return ['week', 'weekly']
    elif resample == 'M':
        return ['month', 'monthly']
    else:
        return [resample, resample]
