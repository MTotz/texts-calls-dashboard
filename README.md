# texts-calls-dashboard

## Create an interactive dashboard of your call and text history

To run, `cd` to inside the `texts-calls-dashboard/src` folder in the terminal and run the following:

```shell
bokeh serve --show dashboard.py
```

As of this writing, this program is very specific in terms of file compatibility. I wrote this to work with the specific format of the files that the backup app on my phone creates. For the texts data file the app creates an XML file, which I then convert to a CSV file using a third-party website; the calls data file I leave as an XML file. I'm not sure how common this format is across different devices and backup apps.

## To-do

1. Create demo gif
2. Add option to switch between only total texts/calls per time and per person
3. Add table of most used words
4. Make the text and call data files more straightforward, in terms of format and location<br>
   (a) Allow for the possibility of multiple text data files, like for calls
