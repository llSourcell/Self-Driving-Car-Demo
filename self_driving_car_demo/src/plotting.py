"""
Take the data in the results folder and plot it.
"""
# standard
import csv
import glob
import os
# third-party
import matplotlib.pyplot as pyplot
import numpy

def movingaverage(y, window_size):
    """
    Moving average function from:
    http://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
    """
    window = numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(y, window, 'same')

def plot_file(filename, type='loss'):
    """
    DOCSTRING
    """
    with open(f, 'r') as csvfile:
        reader = csv.reader(csvfile)
        y = []
        for row in reader:
            if type == 'loss':
                y.append(float(row[0]))
            else:
                y.append(float(row[1]))
        if len(y) == 0:
            return
        print(readable_output(f))
        if type == 'loss':
            window = 100
        else:
            window = 10
        y_av = movingaverage(y, window)
        arr = numpy.array(y_av)
        if type == 'loss':
            print("%f\t%f\n" % (arr.min(), arr.mean()))
        else:
            print("%f\t%f\n" % (arr.max(), arr.mean()))
        pyplot.clf()
        pyplot.title(f)
        if type == 'loss':
            pyplot.plot(y_av[:-50])
            pyplot.ylabel('Smoothed Loss')
            pyplot.ylim(0, 5000)
            pyplot.xlim(0, 250000)
        else:
            pyplot.plot(y_av[:-5])
            pyplot.ylabel('Smoothed Distance')
            pyplot.ylim(0, 4000)
        pyplot.savefig(f + '.png', bbox_inches='tight')

def readable_output(filename):
    """
    DOCSTRING
    """
    readable = ''
    f_parts = filename.split('-')
    if f_parts[0] == 'learn_data':
        readable += 'distance: '
    else:
        readable += 'loss: '
    readable += f_parts[1] + ', ' + f_parts[2] + ' | '
    readable += f_parts[3] + ' | '
    readable += f_parts[4].split('.')[0]
    return readable

if __name__ == '__main__':
    os.chdir("results/sonar-frames")
    for f in glob.glob("learn*.csv"):
        plot_file(f, 'learn')
    for f in glob.glob("loss*.csv"):
        plot_file(f, 'loss')
