import numpy as np
import pandas as pd
import cvxpy as cp

class pyOpt:
    def __init__(self,raw_rets = None):
        self.raw_rets = raw_rets
        self.mu = raw_rets.mean()
        self.Sigma = raw_rets.cov()

    def get_cov_mat(self):
        #Function returns the returns covar matrix input
        return self.Sigma
    
    def get_expected_returns(self):
        return self.mu
    
    def get_constraint_vector(self,w,kappa,mu= None,Sigma=None,min_rets=None,max_vol=None,min_w=None,max_w=None):
        '''
        w: cp variable as input
        mu: array of expected returns
        Sigma: covariance matrix of returns
        kappa: cp variable for risk aversion
        min_rets: minimum returns constraint for the total portfolio
        max_vol: maximum standard deviation constraint for the total portfolio
        min_w: minimum weight constraint for each asset
        max_w: maximum weight constraint for each asset
        '''
        if mu is None:
            mu = self.mu
        if Sigma is None:
            Sigma = self.Sigma
        #Sigma.fillna(0,inplace = True)
        constraints = [w @ mu ==1,
                    cp.sum(w)==kappa,
                    kappa >=0]
        
        for c in [min_rets,max_vol,min_w,max_w]:
            if c is not None:
                if c == min_rets:
                    constraints.append(w.T @ mu >= min_rets)
                elif c == max_vol:
                    constraints.append(cp.quad_form(w, Sigma) <= max_vol**2)
                elif c == min_w:
                    constraints.append(w >= min_w)
                elif c == max_w:
                    constraints.append(w <= max_w * kappa)
        return constraints
    
    def get_max_sharpe_wts(self,mu = None,Sigma = None,total_port_wt = 1,min_rets=None,max_vol=None,min_w=None,max_w=None):
            '''
            Function gets maximum sharpe ratio portfolio weights
            mu: array of expected returns
            Sigma: covariance matrix of returns
            total_port_wt: total portfolio weight (default is 1)
            min_rets: minimum returns constraint for the total portfolio
            max_vol: maximum volatility constraint for the total portfolio
            min_w: minimum weight constraint for each asset
            max_w: maximum weight constraint for each asset

            Returns: Maximum sharpe ratio portfolio weights
            '''
            if mu is None:
                mu = self.mu
            if Sigma is None:
                Sigma = self.Sigma
                
            mu = np.array(mu) # convert to numpy array if not already
            n = len(mu)
            # Sharpe ratio Reformulation
            w = cp.Variable(n)
            portfolio_vol = cp.quad_form(w, Sigma)
            objective = portfolio_vol
            kappa = cp.Variable()

            # Constraints
            constraints = self.get_constraint_vector(w,kappa,mu,Sigma,min_rets,max_vol,min_w,max_w)

            # Formulate problem
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve()
            #print(w.value)
            if total_port_wt ==1:
                w = w.value/kappa.value
            elif total_port_wt ==0:
                #print(kappa.value)
                w = w.value-kappa.value/len(w.value)#/(sum(abs(w.value)))-1/len(w.value) + total_port_wt
                w = w/(sum(abs(w)))
            else:
                return ValueError()
            return w
