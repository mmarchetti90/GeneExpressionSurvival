#!/usr/bin/env python3

"""
Class for variable correlation to survival with lifelines package
Input should be a pandas data frame with samples/patiens as rows and variables as columns.
First two columns should be time and one-hot encoded events (0 = alive, 1 = dead), in that order.
If gene counts are used as variables, they should be NORMALIZED counts.

The following analyses are run:
1. Univariate Cox regression for normalized counts + proportional_hazard_test
2. Expression splitting and KM + logrank_test
3. Multivariable Cox regression for normalized counts + proportional_hazard_test for variables passing either univariate or KM
"""

### ---------------------------------------- ###

class survival_analysis:
    
    def __init__(self, survival_data, output_prefix='survival_analysis', timepoint_cutoff=5, pvalue_cutoff=0.05):
        
        self.survival_data = survival_data
        self.survival_data.columns = ['years_survival', 'vital_status'] + survival_data.columns.to_list()[2:]
        self.output_prefix = output_prefix
        self.max_time = timepoint_cutoff
        self.pvalue_cutoff = pvalue_cutoff
    
    ### ------------------------------------ ###
    ### SURVIVAL ANALYSIS                    ###
    ### ------------------------------------ ###
    
    def correlate_survival(self):
        
        ### Univariate Cox's regression
        univariate_regression = [self.cox_regression(self.survival_data.loc[:, ['years_survival', 'vital_status', var]]) for var in self.survival_data.columns[2:]]
        univariate_regression = pd.concat(univariate_regression, axis=0)
        univariate_regression.columns = [f'univariate_{col}' for col in univariate_regression.columns]
    
        ### K-M fitting and logrank test
        km_results = [self.km_fitting(self.survival_data.loc[:, ['years_survival', 'vital_status', var]], max_t=self.max_time, plot_prefix=self.output_prefix) for var in self.survival_data.columns[2:]]
        km_results = pd.concat(km_results, axis=0)
        km_results = km_results.assign(logrank_padj = fdrcorrection(km_results.logrank_pval, alpha=0.05, is_sorted=False)[1],
                                       survival_at_endpoint_padj = fdrcorrection(km_results.survival_at_endpoint_pval, alpha=0.05, is_sorted=False)[1])
        
        ### Merge and create filter for pvalues
        results = univariate_regression.merge(km_results, left_index=True, right_index=True)
        variable_filter = [index for index,values in results.iterrows() if values.univariate_p < self.pvalue_cutoff or values.logrank_pval < self.pvalue_cutoff]

        if len(variable_filter):
            
            ### Multivariable Cox's regression for variables passing either test above
            multivariable_regression = self.cox_regression(self.survival_data.loc[:, ['years_survival', 'vital_status'] + variable_filter])
            multivariable_regression.columns = [f'multivariable_{col}' for col in multivariable_regression.columns]
        
            ### Merge
            results = results.merge(multivariable_regression, left_index=True, right_index=True, how='outer')

        ### Save to file
        results.to_csv(f'{self.output_prefix}_survival_analysis.tsv', sep='\t', index=True, header=True)
        
        return results
    
    ### ------------------------------------ ###
    
    @staticmethod
    def cox_regression(survdat):
        
        # Fit Cox’s proportional hazard model
        try:
            
            cph = CoxPHFitter()
            cph.fit(survdat, duration_col='years_survival', event_col='vital_status')
            
        except:
            
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(survdat, duration_col='years_survival', event_col='vital_status')

        # Check proportional hazard model assumptions
        assumptions_check = proportional_hazard_test(cph, survdat, time_transform='rank').summary

        # Extract data
        cph_data = cph.summary
        cph_data.drop(columns=['-log2(p)'])
        cph_data = cph_data.assign(proportional_hazard_test_p = assumptions_check.loc[cph_data.index, 'p'].values)
        
        return cph_data
    
    ### ------------------------------------ ###
    
    @staticmethod
    def km_fitting(survdat, max_t=5, plot_prefix='KM'):
        
        # Extract time, events, and variable
        time = survdat["years_survival"].values
        events = survdat["vital_status"].values
        variable = survdat.iloc[:, 2].values
        variable_name = survdat.columns[-1]
        
        if sum(np.isin(variable, [0, 1])) == len(variable): # Dichotomous, one-hot encoded variable
        
            samples_filter = (variable == 0)
            variable_threshold = np.nan
            logrank_p = logrank_test(time[samples_filter], time[~samples_filter],
                                     events[samples_filter], events[~samples_filter], t=max_t).p_value
        
        else: # Continous variable
            
            q1 = np.median(variable[variable <= np.median(variable)])
            q3 = np.median(variable[variable > np.median(variable)])
            
            pvals, thresholds = [], np.linspace(q1, q3, 100)
            
            for thr in thresholds:
                
                # logrank_test
                pval = logrank_test(time[variable <= thr], time[variable > thr],
                                    events[variable <= thr], events[variable > thr], t=max_t).p_value
                
                pvals.append(pval)
            
            padjs = fdrcorrection(pvals, alpha=0.05, is_sorted=False)[1].tolist()
            
            logrank_p = min(padjs)
            variable_threshold = thresholds[padjs.index(logrank_p)]
        
            samples_filter = (variable <= variable_threshold)
        
        # Fit KM
        timeline = np.linspace(0, max_t, 51)
        kmf_low = KaplanMeierFitter().fit(time[samples_filter], event_observed=events[samples_filter], timeline=timeline, label='Low expression')
        kmf_high = KaplanMeierFitter().fit(time[~samples_filter], event_observed=events[~samples_filter], timeline=timeline, label='High expression')
        
        # 5 year survival
        survival_at_endpoint_pval = survival_difference_at_fixed_point_in_time_test(max_t, kmf_low, kmf_high).p_value
        
        # Plot high/low populations' survival
        if logrank_p < 0.05 or survival_at_endpoint_pval < 0.05:
            
            title = f'{variable_name}\nlogrank p-value = {logrank_p:.3E}\n{max_t}y survival p-value = {survival_at_endpoint_pval:.3E}'
            plt.figure(figsize=(5, 3.5))
            ax = plt.subplot(111)
            ax = kmf_low.plot_survival_function(ax=ax, color='blue')
            ax = kmf_high.plot_survival_function(ax=ax, color='red')
            plt.title(title, loc='center', fontsize = 10)
            plt.xlabel('Survival time (years)', fontsize = 10)
            plt.ylabel('Survival probability', fontsize = 10)
            plt.legend(loc='lower left')
            plt.tight_layout()
            plt.savefig(f'{plot_prefix}_{variable_name}_KM.png', dpi=300)
            plt.close()
        
        km_stats = pd.DataFrame({'km_threshold' : [variable_threshold],
                                 'logrank_pval' : [logrank_p],
                                 'survival_at_endpoint_pval' : [survival_at_endpoint_pval]},
                                index=[variable_name])
        
        return km_stats
    
    ### ------------------------------------ ###

### ------------------MAIN------------------ ###

import numpy as np
import pandas as pd

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test, survival_difference_at_fixed_point_in_time_test
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import fdrcorrection
