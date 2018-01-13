import numpy as np

class Chromosome:
    def __init__(self, no_of_assets):
        '''
        Args:
            no_of_assets(int)
                The number of symbols available. This will determine how many genes on the chromosome are created
        '''                    
        self.no_of_assets = no_of_assets
        self.assets = 0
        self.weights = 0
        self.chromosome = np.random.rand(no_of_assets)
        self.to_replace = False
        self.fitness = -np.inf
        self.validation_fitness = -np.inf
        self.test_fitness = -np.inf
        self.portfolio_prices = 0
        self.validation_portfolio_prices = 0
        self.test_portfolio_prices = 0
        
    def mutate(self, mutrate):
        for i in range(self.no_of_assets):
            if np.random.random() < mutrate:
                self.chromosome[i] = np.random.random()      
        
    def clone(self):
        cln = Chromosome(self.no_of_assets)

        for i in range(self.no_of_assets):
            cln.chromosome[i] = self.chromosome[i]
        
        if len(self.weights) > 0:
            cln.weights = np.zeros(len(self.weights))
            cln.assets = np.zeros(len(self.weights))
    
        for i in range(len(self.weights)):
            cln.weights[i] = self.weights[i]
            cln.assets[i] = self.assets[i]
        cln.to_replace = self.to_replace
        return cln