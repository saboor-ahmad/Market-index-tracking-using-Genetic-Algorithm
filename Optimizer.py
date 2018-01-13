from GA import GeneticAlgorithm

def get_genetics_parameters():
    probCrossover, probReplacement, probMutation, sizeTournament = 0.8, 0.6, 0.075, 4
    return probCrossover, probReplacement, probMutation, sizeTournament

def get_tracker_parameters():
    portfolioSize, minWeight, transactionCost, lmbda, predictionInterval, rebalancingPeriod = 15, 0.01, 1.75, 0.6, 100, 30
    return portfolioSize, minWeight, transactionCost, lmbda, predictionInterval, rebalancingPeriod

class TrainOptimizer:
    def __init__(self, index, stocks):
        '''
        Args:
            index (Stock)
                Index (FTSE-100) represented as a Stock
            stocks (List[Stock])
                List of stocks containing the assets in the portfolio
        '''
        # pdb.set_trace()
        self.index = index
        self.stocks = stocks
    
    def train(self, train_interval, validate_interval, test_interval):
        '''
        Args:
            train_interval (list[int]) # E.G. (10, 100)
                integers - specifying the range (STAR, END) of data we want to use to train the Optimizer 
        '''

        # STEPS:
        # 1. Intialize GA parameters
        # 2. Set Training/Prediction intervals    
        # 3. Initialize Evaluator 
        # 4. Initialize Optimizer
        # 5. Train
        
        # pdb.set_trace()
        portfolioSize, minWeight, transactionCost, lmbda, predictionInterval, rebalancingPeriod = get_tracker_parameters() # we set these paramaters as per the PFGA applciaiton
        probCrossover, probReplacement, probMutation, sizeTournament = get_genetics_parameters()

        self.optimizer = GeneticAlgorithm(
                    50, 100,
                    self.stocks, 
                    self.index,
                    genetic_params=(probCrossover, probReplacement, probMutation, sizeTournament), 
                    tracker_params=(portfolioSize, minWeight, rebalancingPeriod, transactionCost, lmbda)
                )
        self.optimizer.run(train_interval, validate_interval, test_interval)
            
    def best_portfolio(self, datatype='test'):
        return self.optimizer.best_portfolio(datatype)

    def solution(self):
        return self.optimizer.solution()
