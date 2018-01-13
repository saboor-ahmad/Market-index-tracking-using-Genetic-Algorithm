import glob
import pandas
import pdb
import numpy

from Optimizer import TrainOptimizer
from Stock import Stock
import matplotlib.pyplot as plt

filenames = glob.glob('Data/*.csv')
indexfile, stockfiles = filenames[0], filenames[1:] # this depends on the filenames in 'Data/' folder
stocks = {}
epics = {}

for index, file in enumerate(stockfiles):
	epic = file.split('/')[1].split('.')[0]
	data = pandas.read_csv(file, names=['Date', 'ClosingPrice'], skiprows=[0])
	stock = Stock(epic, data)
	
	stocks[index] = stock
	epics[index] = epic

indexepic = indexfile.split('/')[1].split('.')[0].split(' ')[0]
data = pandas.read_csv(indexfile, usecols=['Date', 'Adj Close'])
data.columns = ['Date', 'ClosingPrice']
index = Stock(indexepic, data)

opt = TrainOptimizer(index, stocks)
opt.train(train_interval=[0, 10], validate_interval=[10, 20], test_interval=[20, 30])
# print "Best Portfolio:", opt.best_portfolio()




# print "Get Solution:", opt.solution()
