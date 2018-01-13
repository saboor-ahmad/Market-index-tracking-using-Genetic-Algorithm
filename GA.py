import numpy as np
from Chromosome import Chromosome
from PerformanceEvaluator import IndexTracker

import pdb

class GeneticAlgorithm:
    prob_crossover, replacement_rate, prob_mutation, tournament_size = None, None, None, None

    @classmethod
    def update_params(cls, prob_crossover, replacement_rate, prob_mutation, tournament_size):
        cls.prob_crossover = prob_crossover
        cls.replacement_rate = replacement_rate
        cls.prob_mutation = prob_mutation
        cls.tournament_size = tournament_size

    def __init__(self, ga_cycles, population_size, stocks, index, genetic_params, tracker_params):
        '''
        Args:
           genetic_params (List[int])
                A list of parameters that will be used to construct GA 
           stocks (List[Stock])
                Stocks that are being used for tracking
           index (Stock)
                FTSE-100 index that we are trying to replicate
           tracker_params (List[int])
                List of parameters to define an Index Tracker
        '''
        prob_crossover, replacement_rate, prob_mutation, tournament_size = genetic_params
        self.update_params(prob_crossover, replacement_rate, prob_mutation, tournament_size)

        portfolio_size, min_weight, rebalancing_period, transaction_cost, lmbda = tracker_params

        self.stocks = stocks
        self.ga_cycles = ga_cycles
        self.evaluator = IndexTracker(portfolio_size, min_weight, rebalancing_period, transaction_cost, stocks, index, lmbda)
        self.Genetics = Genetics(population_size, len(stocks), self.__class__)

    def run(self, train_interval, validate_interval, test_interval):
        '''
        Args:
            train_interval (List[int])
                StartIndex and EndIndex of the data on which we want to train the model
            validate_interval (List[int])
                StartIndex and EndIndex of the data on which we want to validate the model
            test_interval (List[int])
                StartIndex and EndIndex of the data on which we want to test the model
        '''
        for i in range(self.ga_cycles):
            if i > 0:
                self.Genetics.get_population()

            for i in range(len(self.Genetics.genes)):
                scores = self.evaluator.evaluate(self.Genetics.genes[i].chromosome, train_interval)
                self.Genetics.genes[i].fitness = scores[0]
                self.Genetics.genes[i].pfreturns = scores[1]
                self.Genetics.genes[i].weights = scores[2]
                self.Genetics.genes[i].assets = scores[3]

        for i in range(len(self.Genetics.genes)):
            scores = self.evaluator.evaluate(self.Genetics.genes[i].chromosome, validate_interval)
            self.Genetics.genes[i].validfitness = scores[0]
            self.Genetics.genes[i].validpfreturns = scores[1]

        for i in range(len(self.Genetics.genes)):
            scores = self.evaluator.evaluate(self.Genetics.genes[i].chromosome, test_interval)
            self.Genetics.genes[i].testfitness = scores[0]
            self.Genetics.genes[i].testpfreturns = scores[1]
            
    def best_portfolio(self, portfoliotype):
        '''
        Args:
            portfoliotype (str)
                The sample where to take the returns from: ['train','valid','test']
        '''
        if portfoliotype is None:
            portfoliotype = 'test'

        best = self.Genetics.best_gene()

        r = best.testpfreturns
        if portfoliotype == 'train':
            r = best.trainpfreturns
        elif portfoliotype == 'valid':
            r = best.validpfreturns
            
        shares = [0 for i in range(len(best.assets))]
        for k in range(len(shares)):
            shares[k] = self.stocks[int(best.assets[k])].epic

        return [shares, best.weights, r]
    
    def solution(self):
        return self.Genetics.best_gene()

class Genetics:
    def __init__(self, population_size, no_of_assets, ga_type):
        '''
        Args:
            population_size (int)
                Number of genes
            no_of_assets (int)
                Number of available stocks
            ga_type (GeneticAlgorithm)
                using this to access the global variables of the class GeneticAlgorithm
        '''
        self.population_size = population_size
        self.no_of_assets = no_of_assets
        self.genes = [Chromosome(no_of_assets) for i in range(population_size)]

        self.fittest_genes = []
        self.unfittest_genes = []
        self.fittest_index = 0 # index of fittest chromosome
        self.ga_type = ga_type
        
    def best_gene(self):
        validation_scores = list(map(lambda i: i.validfitness, self.genes))
        self.fittest_gene = np.argmin(validation_scores)
        return self.genes[self.fittest_gene]

    def get_population(self):
        self.tournament()
        self.crossover()
        self.mutate()
    
    def tournament(self):
        self.fittest_genes = []
        self.unfittest_genes = []
        for i in range(len(self.genes)):
            self.genes[i].replace=True
            self.unfittest_genes.append(self.genes[i])
        
        ftns_thres = np.inf
        for i in range(len(self.genes)):
            if self.genes[i].fitness < ftns_thres:
                ftns_thres=self.genes[i].fitness
                self.fittest_index = i
        self.fittest_gene = self.fittest_index
        
        self.genes[self.fittest_index].replace = False 
        self.fittest_genes.append(self.genes[self.fittest_index])
        self.unfittest_genes.remove(self.genes[self.fittest_index])
        
        for i in range(int((1 - self.ga_type.replacement_rate)*self.population_size)):
            ftns_thres = np.inf
            best = 0
            for j in range(self.ga_type.tournament_size):
                cidx = int(np.random.random()*len(self.unfittest_genes))
                self.unfittest_genes[cidx]
                if self.unfittest_genes[cidx].fitness < ftns_thres:
                    best = cidx
                    ftns_thres = self.unfittest_genes[cidx].fitness
            self.unfittest_genes[best].replace=False
            self.fittest_genes.append(self.unfittest_genes[best])
            self.unfittest_genes.remove(self.unfittest_genes[best])
        
    def mutate(self):
        for i in self.genes:
           if i.replace:
               i = self.fittest_genes[int(np.random.random() * len(self.fittest_genes))].clone()
               i.mutate(self.ga_type.prob_mutation)

    def crossover(self):
        for c1 in self.genes:
            if c1.replace:
                if np.random.random() < self.ga_type.prob_crossover:
                    father = self.fittest_genes[int(np.random.random() * len(self.fittest_genes))]
                    mother = self.fittest_genes[int(np.random.random() * len(self.fittest_genes))]
                    sibling = self.unfittest_genes[int(np.random.random() * len(self.unfittest_genes))]

                    split = np.random.random() * self.no_of_assets

                    for j in range(self.no_of_assets):
                        sibling.chromosome[j] = mother.chromosome[j]
                        sibling.chromosome[j] = father.chromosome[j]
                        if j < split:
                            sibling.chromosome[j] = father.chromosome[j]
                            sibling.chromosome[j] = mother.chromosome[j]
