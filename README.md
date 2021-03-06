# texts-calls-dashboard

<img src="sample1.gif" style="width:80%"/>

## Create an interactive dashboard of your call and text history

To run, `cd` to inside the `texts-calls-dashboard/src` folder in the terminal and run the following:

```shell
bokeh serve --show dashboard.py
```

Running the above instructions will create the sample dashboard seen in the GIF above (*to be added*) using the demo data contained in the `data` folder of this repository.

As of this writing, this program is very specific in terms of file compatibility. I wrote this to work with the specific format of the files that the backup app on my phone creates. For the texts data file the app creates an XML file, which I then convert to a CSV file using a third-party website; the calls data file I leave as an XML file. I'm not sure how common this format is across different devices and backup apps.

## To-do

1. Add option to switch between only total texts/calls per time and per person
2. Create config file for variables containing contact names, and resample_interval_string function.
3. Make the text and call data files more straightforward, in terms of format and location
   (a) Allow for the possibility of multiple text data files, like for calls

?. Add some HTML/CSS to spruce it up<br>
?. Look into using Dash to make the dashboard
