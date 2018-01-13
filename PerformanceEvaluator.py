import numpy as np
import pdb

class IndexTracker:
    def __init__(self, portfolio_size, min_weight, rebalancing_period, transaction_cost, stocks, index, lmbda=1):
        ''' 
        Args:
            stocks (Stock)
                Available stocks. This needs to be in the same order as referenced in the solutions/chromo
            portfolio_size (int)
                How many assets are in the portfolio
            min_weight (float)
                Minimum weight an asset can take
            index (Stock)
                Target index to track
            transaction_cost (float)
                Transaction cost as a percentage of the transaction value
            lmbda (float)
                The lambda as defined in Andriosopoulos et al (2013). It defines the weight the tracking 
                error takes in the fitness function. 1-lmbda will be the weight of the risk adjusted excessive prices 
        '''
        # pdb.set_trace()
        self.index = index
        self.stocks = stocks
        self.transaction_cost = transaction_cost
        self.lmbda=lmbda
        self.portfolio_size = portfolio_size
        self.min_weight = min_weight
        self.rebalancing_period = rebalancing_period
      
    def evaluate(self, chromo, evl_range):
        '''
        Args:
            chromo (List[float])
                A list of unscaled weights. This has to be the same length as the number of available stocks. From
                this the actual portfolio weights are calculated
            evl_range (List[int])
                The range to evaluate on. evl_range[0] is the start index, evl_range[1] is the end index
            
        '''
        weights, assets = self.calcweights(chromo)

        indexreturns = self.index.prices

        shares = [0 for i in range(len(assets))]
        currentweights = np.zeros(len(weights))
        for k in range(len(shares)):
            shares[k] = self.stocks[int(assets[k])]
            currentweights[k] = weights[k] 
        
        pfreturns = np.ones(evl_range[1])
        trackingerr = np.zeros(evl_range[1])
        delweights = np.zeros(len(weights))
        
        for j in range(evl_range[0],evl_range[1]):
            # pdb.set_trace()
            cret=0      # current period's return          
            for k in range(len(shares)):
                try:
                    cret += shares[k].prices[j]*currentweights[k]
                except:
                    # happens if the company doesn't exist anymore
                    cret += 0.0*weights[k]
            
            #track current positions
            for k in range(len(shares)):
                try:
                    # current position this share takes in the portfolio
                    currentweights[k] *= (shares[k].prices[j]/cret)
                    delweights[k] = weights[k]-currentweights[k]
                except:
                    delweights[k] = 0
                    
            # rebalance every 3 months
            if np.mod(j-evl_range[0], self.rebalancing_period)==0: 
                # track relative weight change
                cost = sum(abs(delweights))*self.transaction_cost
                cret = cret-cost
                
            pfreturns[j] = cret # portfolio return
            trackingerr[j] = cret - indexreturns[j]
        
        # pdb.set_trace()
        # calculate the tracking error  
        te = np.sqrt(np.sum(np.power(trackingerr[evl_range[0]:evl_range[1]],2))/(evl_range[1]-evl_range[0]))
        # calculate risk adjusted excess prices
        a = np.prod(pfreturns)
        b = np.prod(indexreturns.values()[evl_range[0]:evl_range[1]])
        c = np.subtract(pfreturns[evl_range[0]:evl_range[1]], indexreturns.values()[evl_range[0]:evl_range[1]])
        er = ((a - b) / np.sqrt(np.var(c)))
        fit = self.lmbda*te - (1-self.lmbda)*er

        return [fit, pfreturns, weights, assets]
      
    def calcweights(self, chromo):
        '''
        Args:
            chromo (List[float])
                List of the chromo to derive the portfolio weights from
        '''
        # pdb.set_trace()
        assets = np.zeros(self.portfolio_size)
        weights = np.zeros(self.portfolio_size)        
        sorted_gnome = np.sort(chromo)
        count=0
        for i in range(len(chromo)):
            if count >= self.portfolio_size:
                # This only happens if every baseweight is equal. Should not happen
                break
            if sorted_gnome[-self.portfolio_size] <= chromo[i]:
                assets[count] = i
                # adjust weights so that the zero point is equal to
                # the left-sided boundary baseweight
                weights[count] = max(0.001, chromo[i] - (1 - self.portfolio_size / len(self.stocks)))
                count += 1
        # scale them to add up to 1-numweights*min_weight
        weights = ((1 - self.portfolio_size * self.min_weight) * (weights / np.sum(weights)))
        # add min_weight so all have at least min_weight in pf
        weights += self.min_weight
        
        return weights, assets