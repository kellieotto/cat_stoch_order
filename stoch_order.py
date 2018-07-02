# coding: utf-8

'''
Functions for stochastic ordering permutation tests for categorical data

'''

import numpy as np
import pandas as pd
from npc import t2p, npc, fisher
from scipy.stats import norm

def table_fun(x,y):
    '''
    Compute the crosstable.
    Parameters
    ----------
    x : pandas serie
    y : pandas serie
        

    Returns
    -------
    Pandas data frame
    
    '''
    return pd.crosstab(x,y)
    
    
def dm(x,label,alt):
    
    
    T = np.mean(x[label==2])-np.mean(x[label==1])
    if alt=="greater":
        T = T    
    elif alt=="less":
        T = -T
    elif alt=="two.sided":
        T = np.abs(T)
    return T


def dm_missing(x,label,alt):
    
    S1 = np.sum(x[label==1])
    S2 = np.sum(x[label==2])

    v1 = x[label==1].isnull().sum()
    v2 = x[label==2].isnull().sum()

    T = -S1*np.sqrt(v2/v1)+S2*np.sqrt(v1/v2)
    if alt=="greater":
        T = T    
    elif alt=="less":
        T = -T
    elif alt=="two.sided":
        T = np.abs(T)
    return T


def ad(x,label,alt):
    '''
    Compute the Anderson-Darling test.
    Parameters
    ----------
    x : a pandas serie containing the outcome
    label : a pandas serie containing the treatment levels
        

    Returns
    -------
    A pandas data frame containing the partial test statistic
    
    '''
        
    nt = len(label)
    W = table_fun(x,label)
    k = W.shape[0]
    
    f1 = W.iloc[:,0]
    f2 = W.iloc[:,1]
    f = f1 + f2
    
    F1 = f1.cumsum().iloc[:-1]
    F2 = f2.cumsum().iloc[:-1]
    F = F1 + F2
    
    if alt=="greater":
        T_ad = np.sum(F2/(F*(nt-F))**(1/2))    
    elif alt=="less":
        T_ad = -np.sum(F2/(F*(nt-F))**(1/2))
    elif alt=="two.sided":
        T_ad = (np.sum(F1/(F*(nt-F))**(1/2))**2)
    return T_ad


def permute(x):
    '''
    Permute the treatment levels.
    
    Parameters
    ----------
    x : a pandas serie containing the original treatment levels
        

    Returns
    -------
    A numpy.ndarray containing the permuted treatment levels
    
    '''
    return np.random.choice(x, size=len(x), replace=False)


def stochastic_ordering(data, B, fun="ad", alt="less"):
    '''
    Test the stochastic ordering and return the p-values.
    
    Parameters
    ----------
    data : a pandas data frame with the ids in the first column, the treatment levels
           in the second column and the outcomes in the other ones
    B : number of permutations
    fun: "ad", "dm" or "dm_na"
    alt: "less" or "greater"    

    Returns
    -------
    ppall : partial p-values
    ppvariables: combined p-values by splits (1 x nv)
    ppsplits: combined p-values by variables (1 x (ns-1))
    gpvariables: global p-values combining variables first
    gpsplits: global p-values combining splits first
    
    '''
    
    # Choose the test statistic
    if fun == "ad":
        test_fun = ad
    elif fun == "dm":
        test_fun = dm
    elif fun == "dm_na":
        test_fun = dm_missing
    else:
        raise ValueError("Bad function supplied")
    
    # Calculate data parameters: num variables, num columns, num treatment levels
    nv = len(data.columns) - 2
    nc = nv + 2
    ns = len(np.unique(data.Treatment))

    
    def compute_stats(treatment):
        # Compute the test statistics for each level of treatment
        # Split data into groups: treatment <= i and treatment > i
        tst = np.empty(0)
        for i in range(1, ns):
            data["group"] = 1
            data.group[treatment > i] = 2
        
            # Compute the observed test statistic for each variable
            for j in range(2,nc):
                tst = np.append(tst, test_fun(data.iloc[:,j], data.group, alt))
        return tst
    
    # Initialize an array of permuted statistics
    tst_glob = np.zeros((B+1,nv*(ns-1)))

    # Compute the observed test statistics
    tst_glob[0,:] = compute_stats(data.Treatment)
    
    # Compute permuted statistics
    for i in range(B):
        data["new_treat"] = permute(data.Treatment)
        tst_glob[i+1,:] = compute_stats(data.new_treat)
    
    # Apply t2p to each column of the test statistic array
    # to get partial p-values for each permutation
    # pp_values has the same dimension as tst_glob.
    pp_values = []
    for i in range(0,nv*(ns-1)):
        pp = t2p(tst_glob[:,i],"greater")
        pp_values.append(pp)
    pp_values = np.asarray(pp_values).T
    
    # global p-values: combining the different treatment splits first, then the different variables
    var = []
    for i in range(0,nv):
        temp = pp_values[:,nv*np.arange(ns-1)+i]
        var.append(np.apply_along_axis(fisher, 1, temp))
    var = np.stack(var,axis=1)

    pvar_1 = []
    for i in range(nv):
        pp = t2p(var[:,i],"greater")
        pvar_1.append(pp)
    pvar_1 = np.asarray(pvar_1).T

    distr_1 = np.apply_along_axis(fisher, 1, pvar_1)
    gpvalues_1 = np.sum(distr_1[1:] >= distr_1[0]) / B
    
    # global p-values: combining the different variables first, then the different treatment splits
    var = []
    for i in range(0,ns-1):
        temp = pp_values[:,np.arange(i*nv,(i+1)*nv)]
        var.append(np.apply_along_axis(fisher, 1, temp))
    var = np.stack(var,axis=1)
    
    pvar_2 = []
    for i in range(ns-1):
        pp = t2p(var[:,i],"greater")
        pvar_2.append(pp)
    pvar_2 = np.asarray(pvar_2).T

    distr_2 = np.apply_along_axis(fisher, 1, pvar_2)
    gpvalues_2 = np.sum(distr_2[1:] >= distr_2[0]) / B
    
    return {"ppall": pp_values, 
            "ppvariables": pvar_1[0], 
            "ppsplits": pvar_2[0], 
            "gpvariables": gpvalues_1, 
            "gpsplits": gpvalues_2} 


def corrcheck(marginal, support = None, Spearman = False): 

    # if !all(unlist(lapply(marginal, function(x) (sort(x) == x & min(x) > 0 & max(x) < 1))))) :
    #    stop("Error in assigning marginal distributions!")
    k = len(marginal)
    
    mcmax = np.zeros((k,k))
    np.fill_diagonal(mcmax,1)
    mcmin = np.zeros((k,k))
    np.fill_diagonal(mcmin,1)
    print(mcmax)
    if (support is None) :
        support = {}
        for i in range(k) :
            support[i] = np.arange(1,(len(marginal[i]) + 2))
         
     
    if (Spearman) :
        for i in range(k) :
            
            s1 = np.array(marginal[i]+[1])
            s2 = np.array([0]+marginal[i])
            support[i] = (s1 + s2)/2
         
     
    for i in range(k-1) :
        for j in range(i,k) :
            
            P1 = np.array([0]+marginal[i]+[1])
            P2 = np.array([0]+marginal[j]+[1])
            l1 = len(P1) - 1
            l2 = len(P2) - 1
            print(l1,l2)
            p1 = np.zeros(l1)
            p2 = np.zeros(l2)
            for g in range(l1) :
                p1[g] = P1[g + 1] - P1[g]
             
            for g in range(l2) :
                p2[g] = P2[g + 1] - P2[g]
             
            E1 = sum(p1 * support[i])
            E2 = sum(p2 * support[j])
            V1 = sum(p1 * support[i]**2) - E1**2
            V2 = sum(p2 * support[j]**2) - E2**2
            y1 = 0
            y2 = 0
            lim = 0
            E12 = 0
            PP1 = P1
            PP2 = P2
            PP1 = PP1[1:]
            PP2 = PP2[1:]
            while (len(PP1) > 0) :
                E12 = E12 + support[i][y1] * support[j][y2] * (np.min([PP1[0], PP2[0]]) - lim)
                
                lim = np.min(np.concatenate([PP1, PP2]))
                if (PP1[0] == lim) :
                  PP1 = PP1[1:]
                  y1 = y1 + 1
                 
                if (PP2[0] == lim) :
                  PP2 = PP2[1:]
                  y2 = y2 + 1
                print(lim,PP1,PP2)
                 
             
            c12 = (E12 - E1 * E2)/np.sqrt(V1 * V2)
            y1 = 0
            y2 = l2-1
            lim = 0
            E21 = 0
            PP1 = P1
            PP2 = np.cumsum(np.flip(p2,axis=0))
            PP1 = PP1[1:]
            while (len(PP1) > 0) :
                E21 = E21 + support[i][y1] * support[j][y2] * (np.min([PP1[0], PP2[0]]) - lim)
                lim = np.min(np.concatenate([PP1, PP2]))
                if (PP1[0] == lim) :
                  PP1 = PP1[1:]
                  y1 = y1 + 1
                 
                if (PP2[0] == lim) :
                  PP2 = PP2[1:]
                  y2 = y2 - 1
                 
             
            c21 = (E21 - E1 * E2)/np.sqrt(V1 * V2)
            mcmax[i, j] = c12
            mcmin[i, j] = c21
         
     
    mcmax = np.maximum(mcmax,mcmax.T)
    mcmin = np.maximum(mcmin,mcmin.T)
    return {"mcmin": mcmin, "mcmax": mcmax}
 

