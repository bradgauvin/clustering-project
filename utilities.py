#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# LOCAL LIBRARIES
import os
from env import host, user, password
import env

# THIRD PARTY LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pydataset
import scipy.stats as stats

# python data science library's
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model
import sklearn.feature_selection
import sklearn.preprocessing
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE

# default pandas decimal number display format
pd.options.display.float_format = '{:20,.2f}'.format

# visualizations
from pydataset import data
import matplotlib.pyplot as plt
import seaborn as sns

# state properties
np.random.seed(123)
pd.set_option("display.max_columns", None)

# connection server
def get_connection(db, user=user, host=host,password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
def get_db_url(user,host,password,dbname):
    url = f'mysql+pymysql://{user}:{password}@{host}/dbname'
    return url

def plot_histograms(df, columns):
    """ Plots multiple histograms of specified columns argument using data from input df """
    # List of columns
    cols = columns
    plt.figure(figsize=(25, 10))
    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.histplot(data=df[col], bins=10)

        # Hide gridlines.
        plt.grid(False)

    plt.show()

def plot_boxplots(df, columns):
    """ Plots multiple boxplots of specified columns argument using data from input df """
    # List of columns
    cols = columns
    plt.figure(figsize=(16, 6))
    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[col])

        # Hide gridlines.
        plt.grid(False)

    plt.show()

def plot_variable_pairs(df, numerics, categoricals, targets, sample_amt):
    """ Plots pairwise relationships between numeric variables in df along with regression line for each pair. Uses categoricals for hue."""
    # Sampling allows for faster plotting with large datasets at the expense of not seeing all datapoints
    # Checks if a sample amount was inputted
    if sample_amt:
        df = df.sample(sample_amt)
    # Checks if any categorical variables were given to determine how to set the lmplot regression line parameters
    if len(categoricals)==0:
        categoricals = [None]
        # Setting to red makes it easier to see against the default color
        line_kws = {'lw':4, 'color':'red'}
    else:
        line_kws = {'lw':4}
    for cat in categoricals:    
        for col in numerics:
            for y in targets:
                if y == col:
                    continue
                sns.lmplot(data = df, 
                           x=col, 
                           y=y, 
                           hue=cat, 
                           palette='Set1',
                           scatter_kws={"alpha":0.2, 's':10}, 
                           line_kws=line_kws,
                           ci = None)
            

def plot_categorical_and_continuous_vars(df, categorical, continuous, sample_amt):
    """ Accepts dataframe and lists of categorical and continuous variables and outputs plots to visualize the variables"""
    sns.set(font_scale=1.2)
    # Sampling allows for faster plotting with large datasets at the expense of not seeing all datapoints
    if sample_amt:
        df = df.sample(sample_amt)
    # Outputs 3 plots showing high level summary of the inputted data
    for num in continuous:
        for cat in categorical:
            _, ax = plt.subplots(1,3,figsize=(20,8))
            print(f'Generating plots {num} by {cat}')
            # Strip plot
            p = sns.barplot(data = df, x=cat, y=num, ax=ax[0])
            # Mean line
            p.axhline(df[num].mean())
            # Boxplot
            p = sns.boxplot(data = df, x=cat, y = num, ax=ax[1])
            # mean line
            p.axhline(df[num].mean())
            # Violine plot
            p = sns.violinplot(data = df, x=cat, y=num, hue = cat, ax=ax[2])
            #mean line
            p.axhline(df[num].mean())
            plt.suptitle(f'{num} by {cat}', fontsize = 18)
            plt.show()

def plot_pairplot_pairs(df):
    return sns.pairplot(df, corner = True, kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws':{'s': 1, 'alpha': 0.5}})

# for hypothesis test
def stats_result(p,null_h,**kwargs):
    """
    Compares p value to alpha and outputs whether or not the null hypothesis
    is rejected or if it failed to be rejected.
    DOES NOT HANDLE 1-TAILED T TESTS
    
    Required inputs:  p, null_h (str)
    Optional inputs: alpha (default = .05), chi2, r, t, corr
    
    """
    #Get alpha value - Default to .05 if not provided
    alpha=kwargs.get('alpha',.05)
    #get any additional statistical values passed (for printing)
    t=kwargs.get('t',None)
    r=kwargs.get('r',None)
    chi2=kwargs.get('chi2',None)
    corr=kwargs.get('corr',None)
    
    #Print null hypothesis
    print(f'\n\033[1mH\u2080:\033[0m {null_h}')
    #Test p value and print result
    if p < alpha: print(f"\033[1mWe reject the null hypothesis\033[0m, p = {p} | α = {alpha}")
    else: print(f"We failed to reject the null hypothesis, p = {p} | α = {alpha}")
    #Print any additional values for reference
    if 't' in kwargs: print(f'  t: {t}')
    if 'r' in kwargs: print(f'  r: {r}')
    if 'chi2' in kwargs: print(f'  chi2: {chi2}')
    if 'corr' in kwargs: print(f'  corr: {corr}')

    return None

def model_feature_selection(train, validate, test, to_dummy, features_to_scale, columns_to_use):
    """ Performs scaling and feature selection using recursive feature elimination. Performs operations on all three inputed data sets. Requires lists of features to encode (dummy), features to scale, and columns (features) to input to the feature elimination. """
    
    # Gets dummy variables
    X_train_exp = pd.get_dummies(train, columns = to_dummy, drop_first=True)
    
    # Gets dummy variables for validate and test sets as well for later use
    X_train = X_train_exp[columns_to_use]
    X_validate = pd.get_dummies(validate, columns = to_dummy, drop_first=True)[columns_to_use]
    X_test = pd.get_dummies(test, columns = to_dummy, drop_first=True)[columns_to_use]
    
    # Scale train, validate, and test sets using the scale_data function from wrangle.pu
    X_train_scaled, X_validate_scaled, X_test_scaled = wrangle.scale_data2(X_train, X_validate, X_test,features_to_scale, scaler_type)
    
    # Set up the dependent variable in a datafrane
    y_train = train[['tax_value']]
    y_validate = validate[['tax_value']]
    y_test = test[['tax_value']]
    
    # Perform Feature Selection using Recursive Feature Elimination
    # Initialize ML algorithm
    lm = LinearRegression()
    # create RFE object - selects top 3 features only
    rfe = RFE(lm, n_features_to_select=3)
    # fit the data using RFE
    rfe.fit(X_train_scaled, y_train)
    # get mask of columns selected
    feature_mask = rfe.support_
    # get list of column names
    rfe_features = X_train_scaled.iloc[:,feature_mask].columns.tolist()
    # view list of columns and their ranking

    # get the ranks
    var_ranks = rfe.ranking_
    # get the variable names
    var_names = X_train_scaled.columns.tolist()
    # combine ranks and names into a df for clean viewing
    rfe_ranks_df = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})
    # sort the df by rank
    rfe_ranks_df.sort_values('Rank')

    return X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test, rfe_features
    
def model(X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test, rfe_features, show_test = False, print_results = True):
    """ Fits data to different regression algorithms and evaluates on validate (and test if desired for final product). Outputs metrics for each algorithm (r2, rmse) as a Pandas DataFrame. """
    
    ### BASELINE
    
    # 1. Predict tax_value_pred_mean
    tax_value_pred_mean = y_train['tax_value'].mean()
    y_train['tax_value_pred_mean'] = tax_value_pred_mean
    y_validate['tax_value_pred_mean'] = tax_value_pred_mean

    # 2. compute tax_value_pred_median
    tax_value_pred_median = y_train['tax_value'].median()
    y_train['tax_value_pred_median'] = tax_value_pred_median
    y_validate['tax_value_pred_median'] = tax_value_pred_median

    # 3. RMSE of tax_value_pred_mean
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_mean)**(1/2)
    if print_results:
        print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

    # 4. RMSE of tax_value_pred_median
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_median)**(1/2)

    if print_results:

        print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

    ### OLS Linear Regression
    
    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train_scaled, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_lm'] = lm.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm)**(1/2)

    # predict validate
    y_validate['tax_value_pred_lm'] = lm.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lm)**(1/2)

    if print_results:
        print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
    
    # predict test
    if show_test:
        
        y_test['tax_value_pred_lm'] = lm.predict(X_test_scaled)
        rmse_test = mean_squared_error(y_test.tax_value, y_test.tax_value_pred_lm)**(1/2)

    # Lasso-Lars
    
    # create the model object
    lars = LassoLars(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(X_train_scaled, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_lars'] = lars.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lars)**(1/2)

    # predict validate
    y_validate['tax_value_pred_lars'] = lars.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lars)**(1/2)

    if print_results:

        print("RMSE for OLS using LarsLasso\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)

    # predict test
    if show_test:
        
        y_test['tax_value_pred_lars'] = lars.predict(X_test_scaled)
        rmse_test = mean_squared_error(y_test.tax_value, y_test.tax_value_pred_lars)**(1/2)

    # Tweedie
    
    # create the model object
    glm = TweedieRegressor(power=1, alpha=0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train_scaled, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_glm'] = glm.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_glm)**(1/2)

    # predict validate
    y_validate['tax_value_pred_glm'] = glm.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_glm)**(1/2)
    
    if print_results:
        print("RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
        
    # predict test
    if show_test:
        
        y_test['tax_value_pred_glm'] = glm.predict(X_test_scaled)
        rmse_test = mean_squared_error(y_test.tax_value, y_test.tax_value_pred_glm)**(1/2)

    # Polynomial features
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2,interaction_only=True)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train_scaled)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate_scaled)
    X_test_degree2 = pf.transform(X_test_scaled)
    
    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm2)**(1/2)

    # predict validate
    y_validate['tax_value_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lm2)**(1/2)
    
    if print_results:
        print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
        
    # predict test
    if show_test:
        
        y_test['tax_value_pred_lm2'] = lm2.predict(X_test_degree2)
        rmse_test = mean_squared_error(y_test.tax_value, y_test.tax_value_pred_lm2)**(1/2)

    results = pd.concat([
    y_validate.apply(lambda col: r2_score(y_validate.tax_value, col)).rename('r2'),
    y_validate.apply(lambda col: mean_squared_error(y_validate.tax_value, col)).rename('mse'),
    ], axis=1).assign(
        rmse=lambda df: df.mse.apply(lambda x: x**0.5)
    )
    if show_test:
        results = pd.concat([
        y_train.apply(lambda col: r2_score(y_train.tax_value, col)).rename('r2_train'),
        y_train.apply(lambda col: mean_squared_error(y_train.tax_value, col)).rename('mse_train'),
        y_validate.apply(lambda col: r2_score(y_validate.tax_value, col)).rename('r2_validate'),
        y_validate.apply(lambda col: mean_squared_error(y_validate.tax_value, col)).rename('mse_validate'),
        y_test.apply(lambda col: r2_score(y_test.tax_value, col)).rename('r2_test'),
        y_test.apply(lambda col: mean_squared_error(y_test.tax_value, col)).rename('mse_test'),
        ], axis=1).assign(
            rmse_validate=lambda df: df.mse_validate.apply(lambda x: x**0.5)
        )
        
        results = results.assign(rmse_train= lambda results: results.mse_train.apply(lambda x: x**0.5))
        results = results.assign(rmse_test= lambda results: results.mse_test.apply(lambda x: x**0.5))


def select_kbest(X, y, k):
    """ Takes in predictors (X), target (y) , and number of features to select (k) and returns the names 
    of the top k selected features based on the SelectKBest class."""
    # f_regression stats test for top 2
    f_selector = SelectKBest(f_regression, k=k)
    # find the top 2 X's correlated with y
    f_selector.fit(X,y)
    # Boolean mask of whether the column was selected or now
    feature_mask = f_selector.get_support()
    # List of top k features
    return X.iloc[:,feature_mask].columns.tolist()

def rfe(X,y,k):
    """ Takes in predictors (X), target (y) , and number of features to select (k) and returns the names 
    of the top k selected features based on the RFE class."""
    # initialize the ML algorithm
    lm = LinearRegression()

    # create the rfe object, indicating the ML object (lm) and the number of features I want to end up with. 
    rfe = RFE(lm, n_features_to_select=k)

    # fit the data using RFE
    rfe.fit(X,y)  

    # get the mask of the columns selected
    feature_mask = rfe.support_

    # get list of the column names. 
    return X.iloc[:,feature_mask].columns.tolist()

def scale_data2(train, validate, test, features_to_scale, scaler_type):
    """Scales data using MinMax Scaler. 
    Accepts train, validate, and test datasets as inputs as well as a list of the features to scale. 
    Returns dataframe with scaled values added on as columns"""
    
    # Select which scaler to use
    if scaler_type == 'MinMax':
        scaler = sklearn.preprocessing.MinMaxScaler()
    elif scaler_type == 'Standard':
        scaler = sklearn.preprocessing.StandardScaler()
    elif scaler_type == 'Robust':
        scaler = sklearn.preprocessing.RobustScaler()
    else:
        print("Invalid scaler entry, using MinMax")
        scaler = sklearn.preprocessing.MinMaxScaler()
        
    # Fit the scaler to train data only.     
    scaler.fit(train[features_to_scale])
    
    # Generate a list of the new column names with _scaled added on
    scaled_columns = [col+"_scaled" for col in features_to_scale]
    
    # Transform the separate datasets using the scaler learned from train
    scaled_train = scaler.transform(train[features_to_scale])
    scaled_validate = scaler.transform(validate[features_to_scale])
    scaled_test = scaler.transform(test[features_to_scale])
    
    # Concatenate the scaled data to the original unscaled data
    train_scaled = pd.concat([train, pd.DataFrame(scaled_train,index=train.index, columns = scaled_columns)],axis=1).drop(columns = features_to_scale)
    validate_scaled = pd.concat([validate, pd.DataFrame(scaled_validate,index=validate.index, columns = scaled_columns)],axis=1).drop(columns = features_to_scale)
    test_scaled = pd.concat([test, pd.DataFrame(scaled_test,index=test.index, columns = scaled_columns)],axis=1).drop(columns = features_to_scale)
    
    
    return train_scaled, validate_scaled, test_scaled
    

def remove_outliers(df, k, col_list):
    ''' From the class. Removes outliers based on multiple of IQR. Accepts as arguments the dataframe, the k value for number of IQR to use as threshold, and the list of columns. Outputs a dataframe without the outliers.
    '''
    # Create a column that will label our rows as containing an outlier value or not
    num_obs = df.shape[0]
    df['outlier'] = False
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        df['outlier'] = np.where(((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False), True, df.outlier)
    
    df = df[df.outlier == False]
    df.drop(columns=['outlier'], inplace=True)
    # print(f"Number of observations removed: {num_obs - df.shape[0]}")
        
    return df

def remove_outliers_std(df, k, col_list):
    ''' Removes outliers based on multiple of standard devisation from mean. Accepts as arguments the dataframe, the k value for number of standard deviations to use as threshold, and the list of columns. Outputs a dataframe without the outliers.
    '''
    # Create a column that will label our rows as containing an outlier value or not
    num_obs = df.shape[0]
    df['outlier'] = False
    for col in col_list:

        average = df[col].mean()  # get mean
        standard_deviation = df[col].std() # get standard deviation
        
        upper_bound = average + k * standard_deviation   # get upper bound
        lower_bound = average - k * standard_deviation  # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        df['outlier'] = np.where(((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False), True, df.outlier)
    
    df = df[df.outlier == False]
    df.drop(columns=['outlier'], inplace=True)
    # print(f"Number of observations removed: {num_obs - df.shape[0]}")
        
    return df

def outlier_function(df, cols, k):
    #function to detect and handle oulier using IQR rule
    for col in df[cols]:
        q1 = df.annual_income.quantile(0.25)
        q3 = df.annual_income.quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr     
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]
    return df

def min_max_scaler(train, valid, test):
    '''
    Uses the train & test datasets created by the split_my_data function
    Returns 3 items: mm_scaler, train_scaled_mm, test_scaled_mm
    This is a linear transformation. Values will lie between 0 and 1
    '''
    num_vars = list(train.select_dtypes('number').columns)
    scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    train[num_vars] = scaler.fit_transform(train[num_vars])
    valid[num_vars] = scaler.transform(valid[num_vars])
    test[num_vars] = scaler.transform(test[num_vars])
    return scaler, train, valid, test

def wrangle_mall_df():
    # acquire data
    sql = 'select * from customers'
    mall_df = get_mall_customers(sql)
    # handle outliers
    mall_df = outlier_function(mall_df, ['age', 'spending_score', 'annual_income'], 1.5)   
    # encode gender
    dummy_df = pd.get_dummies(mall_df.gender, drop_first=True)
    mall_df = pd.concat([mall_df, dummy_df], axis=1).drop(columns = ['gender'])
    mall_df.rename(columns= {'Male': 'is_male'}, inplace = True)
    # split data
    train, test = train_test_split(mall_df, train_size = 0.8, random_state = 123)
    train, validate = train_test_split(train, train_size = 0.75, random_state = 123)
    return min_max_scaler, train, validate, test

def split_data(df, train_size_vs_train_test = 0.8, train_size_vs_train_val = 0.7, random_state = 123):
    """Splits the inputted dataframe into 3 datasets for train, validate and test (in that order).
    Can specific as arguments the percentage of the train/val set vs test (default 0.8) and the percentage of the train size vs train/val (default 0.7). Default values results in following:
    Train: 0.56
    Validate: 0.24
    Test: 0.2"""
    train_val, test = train_test_split(df, train_size=train_size_vs_train_test, random_state=123)
    train, validate = train_test_split(train_val, train_size=train_size_vs_train_val, random_state=123)
    
    train_size = train_size_vs_train_test*train_size_vs_train_val
    test_size = 1 - train_size_vs_train_test
    validate_size = 1-test_size-train_size
    
    print(f"Data split as follows: Train {train_size:.2%}, Validate {validate_size:.2%}, Test {test_size:.2%}")
    
    return train, validate, test

def scale_data(train, validate, test, features_to_scale):
    """Scales data using MinMax Scaler. 
    Accepts train, validate, and test datasets as inputs as well as a list of the features to scale. 
    Returns dataframe with scaled values added on as columns"""
    
    # Fit the scaler to train data only
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train[features_to_scale])
    
    # Generate a list of the new column names with _scaled added on
    scaled_columns = [col+"_scaled" for col in features_to_scale]
    
    # Transform the separate datasets using the scaler learned from train
    scaled_train = scaler.transform(train[features_to_scale])
    scaled_validate = scaler.transform(validate[features_to_scale])
    scaled_test = scaler.transform(test[features_to_scale])
    
    # Concatenate the scaled data to the original unscaled data
    train_scaled = pd.concat([train, pd.DataFrame(scaled_train,index=train.index, columns = scaled_columns)],axis=1)
    validate_scaled = pd.concat([validate, pd.DataFrame(scaled_validate,index=validate.index, columns = scaled_columns)],axis=1)
    test_scaled = pd.concat([test, pd.DataFrame(scaled_test,index=test.index, columns = scaled_columns)],axis=1)

    return train_scaled, validate_scaled, test_scaled

def remove_outliers(df, k, col_list):
    ''' Removes outliers based on multiple of IQR. Accepts as arguments the dataframe, the k value for number of IQR to use as threshold, and the list of columns. Outputs a dataframe without the outliers.
    '''
    # Create a column that will label our rows as containing an outlier value or not
    num_obs = df.shape[0]
    df['outlier'] = False
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        df['outlier'] = np.where(((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False), True, df.outlier)
    
    df = df[df.outlier == False]
    df.drop(columns=['outlier'], inplace=True)
    print(f"Number of observations removed: {num_obs - df.shape[0]}")
        
    return df

def wrangle_zillow():
    """ Acquires the Zillow housing data from the SQL database or a cached CSV file. Renames columns and outputs data as a Pandas DataFrame"""
    # Acquire data from CSV if exists
    if os.path.exists('zillow_2017.csv'):
        print("Using cached data")
        df = pd.read_csv('zillow_2017.csv')
    # Acquire data from database if CSV does not exist
    else:
        print("Acquiring data from server")
        query = """
                SELECT * FROM properties_2017
                LEFT JOIN (SELECT logerror, transactiondate, parcelid AS parcelid_pred FROM predictions_2017) as preds_2017
                ON preds_2017.parcelid_pred = properties_2017.parcelid
                LEFT JOIN propertylandusetype as prop_type
                USING (propertylandusetypeid)
                LEFT JOIN airconditioningtype as ac_type
                USING (airconditioningtypeid)
                LEFT JOIN architecturalstyletype as arch_type
                USING (architecturalstyletypeid)
                LEFT JOIN buildingclasstype as b_class_type
                USING (buildingclasstypeid)
                LEFT JOIN heatingorsystemtype as heat_type
                USING (heatingorsystemtypeid)
                LEFT JOIN storytype
                USING (storytypeid)
                LEFT JOIN typeconstructiontype
                USING (typeconstructiontypeid)
                INNER JOIN
                (SELECT parcelid AS parcel_id_max_date, MAX(transactiondate) as max_date
                FROM predictions_2017
                GROUP BY parcel_id_max_date) AS latest_transaction
                ON latest_transaction.parcel_id_max_date = preds_2017.parcelid_pred AND latest_transaction.max_date = preds_2017.transactiondate
                WHERE transactiondate IS NOT NULL
                AND longitude IS NOT NULL
                AND latitude IS NOT NULL
                AND transactiondate BETWEEN '2017-01-01' and '2017-12-31';
            """
        df = pd.read_sql(query, get_db_url('zillow'))
        
        # Drop unnecessary foreign keys
        df = df.drop(columns = ['parcel_id_max_date','max_date','parcelid_pred', 'typeconstructiontypeid','storytypeid','heatingorsystemtypeid','buildingclasstypeid','architecturalstyletypeid','architecturalstyletypeid','airconditioningtypeid','propertylandusetypeid'])
        df.to_csv('zillow_2017.csv', index=False)
    
    # Prepare the data for exploration and modeling
    # Rename columns as needed
    df=df.rename(columns = {'bedroomcnt':'bedroom', 
                            'bathroomcnt':'bathroom', 
                            'calculatedfinishedsquarefeet':'square_feet',
                            'taxvaluedollarcnt':'tax_value',
                            'garagecarcnt':'garage',
                           'buildingqualitytypeid':'condition',
                           'regionidzip':'zip',
                           'poolcnt':'pools',
                           'lotsizesquarefeet':'lot_size'})
    
    
    return df

def handle_missing_zillow_values(df):
    """ Specific to Zillow dataset. Filters to single unit properties and deals with null values."""
    
    # Just want single unit properties
    
    # Filter out anything other than unit count = 1 and nans
    df = df[(df.unitcnt==1)|(df.unitcnt.isna())]
    
    # Keep properties that should be single units
    properties_to_keep = ['Single Family Residential','Condominum','Mobile Home','Manufactured, Modular, Prefabricated Homes','Residential General','Townhouse',np.nan]
    df = df[df.apply(lambda row: row.propertylandusedesc in properties_to_keep, axis=1)]
    
    # First pass on removing missing values
    df_nulls_removed = handle_missing_values(df, prop_required_column=0.3, prop_required_row=0.00002)
    
    # Fill na values and drop specific columns based on exploring nans
    df_nulls_removed['garage'] = df_nulls_removed['garage'].fillna(0)
    df_nulls_removed['garagetotalsqft'] = df_nulls_removed['garagetotalsqft'].fillna(0) # 'No garage'
    df_nulls_removed['poolsizesum'] = df_nulls_removed['poolsizesum'].fillna(0)# 'No pool'
    df_nulls_removed['basementsqft'] = df_nulls_removed['basementsqft'].fillna(0) # 'No basement information'
    df_nulls_removed['threequarterbathnbr'] = df_nulls_removed['threequarterbathnbr'].fillna(0)
    df_nulls_removed['taxdelinquencyyear'] = df_nulls_removed['taxdelinquencyyear'].fillna(0) # "Assumed Not Delinquent"
    df_nulls_removed['condition'] = df_nulls_removed['condition'].fillna(-1) # "Not available"
    df_nulls_removed['yardbuildingsqft17'] = df_nulls_removed['yardbuildingsqft17'].fillna(0) # "No Patio Information"
    df_nulls_removed['yardbuildingsqft26'] = df_nulls_removed['yardbuildingsqft26'].fillna(0) # "No Yard Building"
    df_nulls_removed = df_nulls_removed.drop(columns = ['regionidneighborhood','calculatedbathnbr','finishedsquarefeet13','finishedsquarefeet50','finishedsquarefeet6','finishedsquarefeet12','finishedfloor1squarefeet'])
    # Make a column for the county based on FIPS
    df_nulls_removed["county"] = np.select([df_nulls_removed.fips == 6037, df_nulls_removed.fips==6059, df_nulls_removed.fips == 6111],["Los Angeles County", "Orange County", "Ventura County"])
    
    # Fill in binary values with 0s
    for col in df_nulls_removed.columns:
        if df_nulls_removed[col].nunique() == 1:
            df_nulls_removed[col] = df_nulls_removed[col].fillna('None')
    
    # Fill in count, number, and desc values with 0s and not specified
    for col in df_nulls_removed.columns:
        if 'desc' in col:
            df_nulls_removed[col] = df_nulls_removed[col].fillna('Not Specified')
        elif 'cnt' in col:
            df_nulls_removed[col] = df_nulls_removed[col].fillna(0)
        elif 'number' in col:
            df_nulls_removed[col] = df_nulls_removed[col].fillna(0)
            
    # For now, just remove remaining null values
    df_nulls_removed = df_nulls_removed.dropna()
            
    return df_nulls_removed

def handle_missing_values(df, prop_required_column, prop_required_row):
    """ Drops columns and rows from df that have fewer values than required by the 
    arguments prop_required_column and prop_required_row. First drops columns then drops rows. 
    Returns a df without the columns and rows that were dropped. """
    
    # Drop columns with pct of missing rows above threshold
    print("For threshold based dropping: ")
    print(df.shape, " original shape")
    df = df.dropna(thresh = int((prop_required_row)*len(df)), axis=1, inplace=False)
    print(df.shape, " shape after dropping columns with prop required rows below theshold")
    
    # Drop rows with pct of missing columns above threshold
    df = df.dropna(thresh = int(prop_required_column*len(df.columns)), inplace=False)
    print(df.shape, " shape after dropping rows with prop required columns below threshold")
    
    return df


def nulls_by_row(df):
    """ Returns the number of and percent of nulls per row, as well as the number of rows with the given missing num of columns """
    # nulls by row
    info =  pd.concat([
        df.isna().sum(axis=1).rename('num_cols_missing'),
        df.isna().mean(axis=1).rename('pct_cols_missing'),
    ], axis=1)
    
    return pd.DataFrame(info.value_counts(),columns = ['num_rows']).reset_index().sort_values(by='num_rows', ascending=False)

def nulls_by_column(df):
    """ Returns the number of and percent of nulls per column """
    return pd.concat([
        df.isna().sum(axis=0).rename('n_rows_missing'),
        df.isna().mean(axis=0).rename('pct_rows_missing'),
    ], axis=1).sort_values(by='pct_rows_missing', ascending=False)

def get_upper_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.
    
    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))

def get_lower_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the lower outliers for the
    series.
    
    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the lower bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    return s.apply(lambda x: max([x - lower_bound, 0]))

def add_upper_outlier_columns(df, k, describe=False):
    '''
    Add a column with the suffix _upper_outliers for all the numeric columns
    in the given dataframe and the given cutoff k value. Optionally displays a description of the outliers.
    '''
    
    for col in df.select_dtypes('number'):
        df[col + '_upper_outliers'] = get_upper_outliers(df[col], k)
    
    outlier_cols = [col for col in df if col.endswith('_upper_outliers')]

    if describe:
        for col in outlier_cols:
            print('---\n' + col)
            data = df[col][df[col] > 0]
            print(data.describe())

    return df

def add_lower_outlier_columns(df, k, describe = False):
    '''
    Add a column with the suffix _lower_outliers for all the numeric columns
    in the given dataframe and the given cutoff k value. Optionally displays a description of the outliers.
    '''
    
    for col in df.select_dtypes('number'):
        df[col + '_lower_outliers'] = get_lower_outliers(df[col], k)
    
    outlier_cols = [col for col in df if col.endswith('_lower_outliers')]

    if describe:
        for col in outlier_cols:
            print('---\n' + col)
            data = df[col][df[col] > 0]
            print(data.describe())
            
    return df

def summary_info(df): 
    # Summarize data (shape, info, summary stats, dtypes, shape)
    print('--- Shape: {}'.format(df.shape))
    print('--- Descriptions')
    print(df.describe(include='all'))
    print('--- Info')
    return df.info()