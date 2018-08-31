import pandas as pd
import numpy as np
from toolz.dicttoolz import valmap
import altair as alt
import time
import multiprocessing
from IPython.display import display


def setSearchStepAndArgs(lowerArg, upperArg, decimals):
    #Some hyperparameters only accept ints
    if decimals==0:
        return np.array([(upperArg-lowerArg)/2, 
                         lowerArg, 
                         upperArg],
                        dtype=np.int)
    else:
        return (np.round((upperArg-lowerArg)/2, 
                         decimals),
                lowerArg,
                upperArg)
        

def compareVals(X, y, model, params, var, decimals, newArg, lastArg, lastVal, timesAndScores):
    if np.abs(newArg-lastArg)<=10**(-decimals):
        return pd.DataFrame(timesAndScores)
    
    if lastArg>newArg:
        lowerArg = newArg
        upperArg = lastArg
        
        
        searchStep, lowerArg, upperArg = setSearchStepAndArgs(lowerArg, upperArg, decimals)

        start_time = time.perf_counter()
        lowerVal = model(X, 
                         y,
                         {**params,
                          **{var:lowerArg}})
        end_time = time.perf_counter()
        run_time = end_time - start_time
        timesAndScores = timesAndScores + [{var: lowerArg,
                                           "score": lowerVal,
                                           "time": run_time}]
        
        upperVal = lastVal
        
    if newArg>lastArg:
        
        lowerArg = lastArg
        upperArg = newArg
        
        
        searchStep, lowerArg, upperArg = setSearchStepAndArgs(lowerArg, upperArg, decimals)
        

        
        lowerVal = lastVal
        
        start_time = time.perf_counter()
        upperVal = model(X, 
                         y,
                         {**params, 
                          **{var:upperArg}}, 
                         )
        end_time = time.perf_counter()
        run_time = end_time - start_time
        timesAndScores = timesAndScores + [{var: upperArg,
                                           "score": upperVal,
                                           "time": run_time}]
        
    if lowerVal==upperVal:
        return pd.DataFrame(timesAndScores)
        
    if lowerVal>upperVal:
        return compareVals(X, 
                           y, 
                           model,
                           params,
                           var,
                           decimals,
                           upperArg - searchStep,
                           lowerArg,
                           lowerVal, 
                           timesAndScores)
       
    if upperVal>lowerVal:
        return compareVals(X, 
                           y, 
                           model,
                           params, 
                           var,
                           decimals,
                           lowerArg + searchStep,
                           upperArg,
                           upperVal, 
                           timesAndScores)

    
def compareValsBaseCase(X, y, model, params, var, decimals, lowerArg, upperArg):
    """Run the binary search
    
    Parameters
    ----------
    X : NumPy array
        Training data
    y : 1d NumPy array
        Training data answers
    model : function
        A function that takes in some arguments and returns a metric for an ML algo
    params : dictionary
        Parameters for our ML algo that we want in every run
    var: string
        The parameter we want to optimize
    decimals: int
        How many decimals of difference we want between test values of the parameter
        For instance, some things have to be whole numbers, but others are floats
    lowerArg: int or float
        Lower limit to search
    upperArg: int or float
        Upper limit to search
        
    Returns
    -------
    pandas.DataFrame
        Contains the values that were tested, the performance, and the time it took
        to run
    
    """
    searchStep, lowerArg, upperArg = setSearchStepAndArgs(lowerArg, upperArg, decimals)


    timesAndScores = []
    
    start_time = time.perf_counter()
    lowerVal = model(X, 
                     y,
                     {**params,
                      **{var:lowerArg}},
                     )
    end_time = time.perf_counter()
    run_time = end_time - start_time
    timesAndScores = timesAndScores + [{var: lowerArg,
                                        "score": lowerVal,
                                        "time": run_time}]

    start_time = time.perf_counter()
    upperVal = model(X, 
                     y,
                     {**params, 
                      **{var:upperArg}}, 
                     )
    end_time = time.perf_counter()
    run_time = end_time - start_time
    timesAndScores = timesAndScores + [{var: upperArg,
                                        "score": upperVal,
                                        "time": run_time}]
    
    if lowerVal==upperVal:
        return pd.DataFrame(timesAndScores)
    
    if lowerVal>upperVal:
        return compareVals(X, 
                           y, 
                           model,
                           params, 
                           var,
                           decimals,
                           upperArg - searchStep,
                           lowerArg,
                           lowerVal, 
                           timesAndScores)
    
    if upperVal>lowerVal:
        return compareVals(X, 
                           y,  
                           model,
                           params,  
                           var,
                           decimals,
                           lowerArg + searchStep,
                           lowerArg,
                           lowerVal, 
                           timesAndScores)

def standardizeAddRatioAndMelt(inputDF):
    df = inputDF.copy()
    
    #Feature scaling.  Keeps both values positive, which is better for a ratio
    df[["score", "time"]] = ((df[["score", "time"]]  #Add a little bias term
                              - (df[["score", "time"]].min()) +  0.001 ) /
                             (df[["score", "time"]].max()
                              - df[["score", "time"]].min()))
    
    df["scoreTimeRatio"] = df["score"]/df["time"]
    
    
    
    df["scoreTimeRatio"] = ((df["scoreTimeRatio"]
                              - df["scoreTimeRatio"].min()) /
                             (df["scoreTimeRatio"].max()
                              - df["scoreTimeRatio"].min())) 
    
    display(df)
    
    return df.melt(id_vars=df.columns[0])

def plotTimeAndScore(melted):
    return (alt.Chart(melted).mark_line().encode(
        x=f'{melted.columns[0]}:O',
        y='value:Q',
        color='variable:N')
           ).properties(width=400)

def showTimeScoreChartAndGraph(df):
    melted = standardizeAddRatioAndMelt(df)
    display(df)
    display(plotTimeAndScore(melted))

    
def getTopValsAndScores(df):
    return dict(zip(["val", "score"], 
                    (df
                     .sort_values(by="score", 
                         ascending=False)
                     .iloc[0,:2]
                     .values)))

def getTopVals(df):
    return (df
            .sort_values(by="score", 
                         ascending=False)
            .iloc[0,0])

def binarySearchParams(X, y, model, params, paramRanges):
    valsAndScores = {x[0] : compareValsBaseCase(X, 
                                                y,
                                                model,
                                                params, 
                                                *x)
               for x in paramRanges}
    
    topVals = valmap(getTopVals, valsAndScores)
    
    score = model(X, 
                  y, 
                  {**params, 
                   **topVals},
                  )
    
    return {"values": topVals,
           "score": score,
           "valsAndScores": valsAndScores,
           "n_iterations": sum(x.shape[0] 
                               for x in valsAndScores.values())}


def binarySearchParamsParallel(X, y, model, params, paramRanges):
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        args_generator = ((X, 
                           y,
                           model,
                           params, 
                           *x) for x in paramRanges)
        results = p.starmap(compareValsBaseCase, args_generator)
        name_result_tuples = zip((x[0] for x in paramRanges), 
                                 results)
        valsAndScores = dict(name_result_tuples)
    
    topVals = valmap(getTopVals, valsAndScores)
    
    score = model(X, 
                  y, 
                  {**params, 
                   **topVals},
                  )
    
    return {"values": topVals,
           "score": score,
           "valsAndScores": valsAndScores,
           "n_iterations": sum(x.shape[0] 
                               for x in valsAndScores.values())}