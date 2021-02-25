from texts import text_messages
from calls import calls

import pandas as pd
from bokeh.io import curdoc
from bokeh.models.widgets import Tabs, Panel, Div
from bokeh.layouts import column, row

###########################################################
# WORKING
# To run application, open terminal, navigate to this directoy, and run the following:
# bokeh serve --show Display_app.py
###########################################################

person1 = "John"
person2 = "Mary"


resample = 'D'
# make sure that the desired xml backup file has been converted to xml, then use that csv file here
# see Texts.py for more info
# path from location from wher eyou are running this file
#file = "../../data/texts_demo_data.csv"
file = "/Users/michael/Desktop/Stuffing/Python/Projects/TextsCalls/sms.csv"
file = "data/texts_demo_data.csv"
text_layout = text_messages(file, resample)
call_layout = calls(resample)

tabs = Tabs(tabs=[Panel(child=text_layout, title='Texts'),
                  Panel(child=call_layout, title='Calls')])

curdoc().add_root(tabs)
