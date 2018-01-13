import numpy
import pdb

class Stock():
	def __init__(self, epic, data):
		self.epic = epic
		self.dates = dict(zip(numpy.arange(data.shape[0]), data['Date'].values))
		data.drop('Date', axis=1, inplace=True)
		self.prices = data.to_dict().values()[0]