import pandas as pd
import sys  # to get file name from command line argument, assume only one argument, file name without extension
# function converts seconds to days, hours:minutes:seconds
from datetime import timedelta as delta
import matplotlib.pyplot as plt
import numpy as np  # to replace 0 values with NaN
from os import listdir
import xml.etree.ElementTree as ET
import re
from math import pi

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LabelSet, Title, Label, LinearAxis, Range1d
from bokeh.models.widgets import Tabs, Panel, Select
from bokeh.layouts import column, row, gridplot
from bokeh.transform import cumsum

from texts import resample_interval_string

person1 = "Mary"  # the name of the person who downloaded the call log
person2 = "John"  # the name of your contact as it will appear in the dashboard
person3 = person2  # the name of person2 as listed in the call log


def merge_dataframes():
    """
    Call logs are stored in several xml files. This function combines them all into one DataFrame.
    Input: None.
    Returns: The DataFrame containing the call logs of all XML files.
    """

    path = '/Users/michael/Desktop/Stuffing/Python/Projects/TextsCalls/'
    path = "../data/"

    files = [f for f in listdir(path)]  # get files from folder
    xmls = [f for f in files if re.search('.xml', f)]  # get only xml files

    # create a dataframe from each xml file
    df_list = [create_dataframe(path + f) for f in xmls]
    df_merged = pd.concat(df_list)  # concatenate all the dataframes

    df_merged = df_merged.drop_duplicates()  # drop the duplicate records
    df_merged['Duration (s)'] = df_merged['Duration (s)'].astype(
        'int')  # convert Duration column to int type to be able to add
    df_merged['Date'] = pd.to_datetime(df_merged['Date'])
    # change type to initials of caller for readability
    df_merged['Caller'] = df_merged['Caller'].replace('1', person1)
    df_merged['Caller'] = df_merged['Caller'].replace('2', person2)
    df_merged['Caller'] = df_merged['Caller'].replace('3', '--')  # missed call

    return df_merged


def create_dataframe(file):
    """
    Parse the XML call log file and turn it into a DataFame.
    Input: Path file of XML as a string.
    Returns: Call log DataFrame.
    """

    tree = ET.parse(file)
    root = tree.getroot()

    # initialize the columns we want to keep in the final dataframe
    number = []  # number
    date = []  # readable_date
    duration = []  # duratiion
    caller = []  # type

    # for each child in the xml file extract the relevant attributes
    for child in root:
        if child.get('contact_name') == person3:
            # number.append(child.get('number'))
            date.append(child.get('readable_date'))
            duration.append(child.get('duration'))
            caller.append(child.get('type'))

    labels = ['Date', 'Caller', 'Duration (s)']
    cols = [date, caller, duration]
    data = dict(list(zip(labels, cols)))

    return pd.DataFrame(data)


def plot_calls(df_merged, resample):
    """
    Create display of call history.
    Input: DataFrame of call logs (should be the output DataFrame of the merge_dataframes() function;
    resampling frequency as a string.
    Returns: Bokeh Panel to be used in HTML display.
    """

    resample_strings = resample_interval_string(resample)
    plot_height, plot_width = 350, 900  # set the desired plot dimensions

    df = df_merged.set_index('Date')

    df['Duration (h)'] = df['Duration (s)'] / 3600
    df['Duration (m)'] = df['Duration (s)'] / 60

    average_duration = '{:1.2f}'.format(df['Duration (m)'].mean())

    df['Count'] = 1  # create count column for when we resample
    df[person1] = df['Caller'].apply(
        lambda x: 1 if x == person1 else np.nan)
    df[person2] = df['Caller'].apply(
        lambda x: 1 if x == person2 else np.nan)
    df['Missed'] = df['Caller'].apply(lambda x: 1 if x == '--' else 0)

    # calculate column for cumulative duration
    # cum_total = [df.iloc[0, 2]] # initialize as duration of first call
    # for i in range(1, df['Count'].count()): # add each duration
    # cum_total.append(df.iloc[i, 2] + cum_total[i - 1])
    # df['Cum_Total'] = cum_total

    average_time = 0

    # create functions to create data source of glyphs
    # to be used in the drop down menu callback function
    def create_source1(resample):
        """
        Create DataFrame for cumulative graph.
        """

        data1 = df.resample(resample).sum()
        data1 = data1.reset_index()
        average_time = '{:1.2f}'.format(data1['Count'].mean())

        # calculate column for cumulative phone time in hours
        cum_total = [data1.iloc[0, 2]]  # initialize as duration of first call
        for i in range(1, data1['Duration (h)'].count()):  # add each duration
            cum_total.append(data1.iloc[i, 2] + cum_total[i - 1])
        data1['Cum_Total'] = cum_total
        return data1

    def create_source2(resample):
        """
        Create DataFrame for averaged graph.
        """

        data2 = df.resample(resample).mean()

        return data2

    df_sum = create_source1(resample)
    source1 = ColumnDataSource(data=df_sum)

    df_mean = create_source2(resample)
    source2 = ColumnDataSource(data=df_mean)

    line1 = figure(x_axis_type='datetime',  # plot datetime objects on x axis
                   y_axis_label='Number of calls per ' + resample_strings[0],
                   tools=['pan,box_zoom,reset'], plot_width=plot_width, plot_height=plot_height)
    calls_per_day_glyph = line1.line('Date', person1, source=source1, color='red',
                                     alpha=0.6, line_width=2, legend_label=person1)
    line1.line('Date', person2, source=source1, color='green',
               alpha=0.6, line_width=2, legend_label=person2)
    # line1.y_range = Range1d(start=0, end=df_sum['person1'].max() + 5)
    line1.title.text = 'Average of ' + \
        '{:1.2f}'.format(df_sum['Count'].mean()) + \
        ' calls per ' + resample_strings[0]
    line1.legend.background_fill_alpha = 0.4

    hover_line1 = HoverTool(tooltips=[('Date', '@Date{%F}'), ('Calls made by ' + person1, '@' + person1),
                                      ("Calls made by " + person2, '@' + person2)],
                            formatters={'@Date': 'datetime'}, renderers=[calls_per_day_glyph], mode="vline")
    line1.add_tools(hover_line1)

    line2 = figure(x_axis_type='datetime',  # plot datetime objects on x axis
                   y_axis_label='Average call duration per ' + \
                   resample_strings[0] + ' (minutes)',
                   tools=['pan,box_zoom,reset'], plot_width=plot_width, plot_height=plot_height)
    call_duration_glyph = line2.line('Date', 'Duration (m)', source=source2, color='brown', alpha=0.6, line_width=2,
                                     legend_label='Average call duration (minutes)')
    line2.y_range = Range1d(start=0, end=df_mean['Duration (m)'].max() + 5)
    # add separate y axis to plot cumulative total
    line2.extra_y_ranges = {'cumulative': Range1d(
        start=0, end=df_sum.iloc[-1, -1] + 200)}
    cum_axis = LinearAxis(y_range_name='cumulative',
                          axis_label='Cumulative phone time (h)')
    line2.add_layout(cum_axis, 'right')
    line2.varea(x='Date', y2='Cum_Total', source=df_sum,  # area plot of total texts sent
                color='blue', alpha=0.07, legend_label='Cumulative call time (hours)', y_range_name='cumulative')
    line2.title.text = 'Average of ' + average_duration + ' minutes per call'
    line2.legend.background_fill_alpha = 0.4
    hover_line2 = HoverTool(
        tooltips=[('Date', '@Date{%F}')], formatters={'@Date': 'datetime'}, renderers=[call_duration_glyph], mode="vline")
    line2.add_tools(hover_line2)

    line1.x_range = line2.x_range

    calls_made = pd.DataFrame(df.iloc[:, 5:].sum())
    calls_made = calls_made.reset_index()

    calls_made = calls_made.rename(columns={0: 'count'})
    # add column with angle of pie chart wedge
    calls_made['angle'] = calls_made['count'] / \
        calls_made['count'].sum() * 2 * pi
    calls_made['percent'] = calls_made['count'] / \
        calls_made['count'].sum() * 100
    calls_made['color'] = ['red', 'green', 'blue']  # colors of wedges

    # plot pie chart by creating a wedge for each sender
    pie_tooltips = """
        <div style="width:200px;">
            <span style="color:#5DAED9">Calls made: </span><span>@count (@percent%)</span>
        </div>
    """
    hover_pie = HoverTool(
        tooltips=pie_tooltips)
    pie = figure(plot_width=int(plot_width/3),
                 plot_height=plot_height, tools=[hover_pie])
    # create a wedge for each row in the total_texts dataframe
    pie.wedge(x=0, y=1, radius=0.8,
              start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
              line_color='white', color='color', alpha=0.6, legend_field='index', source=calls_made)
    pie.axis.visible = False  # remove axes
    pie.grid.grid_line_color = None  # remove gridlines
    pie.title.text = 'Number of calls made\n(Total of ' + \
        str(int(calls_made['count'].sum())) + ')'
    pie.legend.background_fill_alpha = 0.4

    menu = Select(options=['Daily', 'Weekly', 'Monthly'],
                  value='Daily', title='Resample frequency')

    def callback(attr, old, new):

        if menu.value == 'Daily':
            df_sum = create_source1('D')
            df_mean = create_source2('D')
            resample_strings = resample_interval_string('D')
        elif menu.value == 'Monthly':
            df_sum = create_source1('M')
            df_mean = create_source2('M')
            resample_strings = resample_interval_string('M')
        else:
            df_sum = create_source1('W')
            df_mean = create_source2('W')
            resample_strings = resample_interval_string('W')

        source1.data = df_sum
        source2.data = df_mean
        line1.title.text = 'Average of ' + \
            '{:1.2f}'.format(df_sum['Count'].mean()) + \
            ' calls per ' + resample_strings[0]
        line1.yaxis.axis_label = 'Number of calls per ' + resample_strings[0]
        line2.yaxis.axis_label = 'Average call duration per ' + \
            resample_strings[0] + ' (minutes)'
        # change back the title of the right cumulative
        cum_axis.axis_label = 'Cumulative phone time (h)'
        # ... axis, since the above callback changes it back to be the same as the left

    menu.on_change('value', callback)

    layout = column(row(line1, pie), row(line2, menu))
    panel = Panel(child=layout, title='Calls')

    # output_file('calls.html')
    # show(layout)

    return layout


def calls(resample):
    df = merge_dataframes()
    panel = plot_calls(df, resample)

    return panel

# calls('D')
