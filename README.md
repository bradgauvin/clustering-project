# Clustering - Looking at Zillow 2017 data
*Audience: Target audience for my final report is a data science team*


<hr style="background-color:silver;height:3px;" />

## Project Summary

> - Overall Goal is to document the discoveries you made and work you have done related to uncovering what the drivers of the error in the zestimate is.

<hr style="background-color:silver;height:3px;" />

### Project Deliverables
> - A final report notebook
> - Python modules for automation and to facilitate project reproduction
> - Notebooks that show:
> - This README file
>  - Data acquisition and preparation 
>  - exploratory analysis not included in final report
>  - model creation, refinement and evaluation

### Specific requirements to document
> - Data Acquisition: Data is collected from the codeup cloud database with an appropriate SQL query
> - Data Prep: Column data types are appropriate for the data they contain
> - Data Prep: Missing values are investigated and handled
> - Data Prep: Outliers are investigated and handled
> - Exploration: the interaction between independent variables and the target variable is explored using visualization and statistical testing
> - Exploration: Clustering is used to explore the data. A conclusion, supported by statistical testing and visualization, is drawn on whether or not the clusters are helpful/useful. At least 3 combinations of features for clustering should be tried.
> - Modeling: At least 4 different models are created and their performance is compared. One model is the distinct combination of algorithm, hyperparameters, and features.
> - Best practices on data splitting are followed
> - The final notebook has a good title and the documentation within is sufficiently explanatory and of high quality
> - Decisions and judment calls are made and explained/documented
> - All python code is of high quality


### Key instructions
> - Model on scaled data, explore on unscaled

### Initial questions on the data

>  - Questions
>  - Thoughts
>  - etc

---

<hr style="background-color:silver;height:3px;" />

## Executive Summary
<hr style="background-color:silver;height:3px;" />

**Project Goal:**

**Discoveries and Recommendations**


<hr style="background-color:silver;height:3px;" />

## Data Dictionary
<hr style="background-color:silver;height:3px;" />

|Variable|	Meaning|
|:-------|:----------|
|bedroom|	number of bedrooms in home|
|bathroom|number of bathrooms in home|
|bathroom_bin|	number of bathrooms split into 3 categories|
|bedroom_bin|	number of bedrooms split into 3 categories|
|age|	age of the home|
|square_feet|	total living area of home|
|tax_value|	total tax assessed value of the parcel (target)|
|fips|	Federal Information Processing Standard code (location)|
|bed_bath_ratio|	ratio of bedrooms to bathrooms|
|living_space|	square footage - (bathrooms40 + bedrooms200)|
|longitude|longidudinal coordinates|
|latitude| latitudinal coordinates|
|room_count|	sum of bedrooms and bathrooms|
|pool|	whether the home has a pool|
|has_garage|	whether the home has a garage|
|condition|	assessment of the condition of the home, low values are better|
|heatingorsystemdesc|	heating system|
|fullbathcnt_bin|	binned count of full bathrooms|
|home_size|	size of home binned|
|tax_rate|	tax amount/ tax value|
|structure_dollar_per_sqft|	tax value / square footage|
|land_dollar_per_sqft	land| tax value / square footage|
|abs_logerror|	absolute value of prediction log error|
|tax_value_bin|	binned tax values|
|lot_size_bin|	binned lot sizes|
|structure_value_bin|	structure dollar per sqft binned|
|land_value_bin|	land value binned|
|taxdelinquencyflag|	home is delinquent on taxes|
|delinquent_years|	years delinquent|


<hr style="background-color:silver;height:3px;" />

## Reproducing this project
<hr style="background-color:silver;height:3px;" />

> In order to reproduce this project you will need your own environment file and access to the database. You can reproduce this project with the following steps:
> - Read this README
> - Clone the repository or download all files into your working directory
> - Add your environment file to your working directory:
>  - filename should be env.py
>  - contains variables: username, password, host
> - Run the Final_Report notebook or explore the other notebooks for greater insight into the project.

### Project Plan 

<details>
  <summary><i>Click to expand</i></summary>
  <ul>
   <li><b>Acquire</b> data from XXXX</li>
    <li>Clean and <b>prepare</b>data for the exploration. </li>
    <li>Create wrangle.py to store functions I created to automate the cleaning and preparation process.</li>
    <li>Separate train, validate, test subsets and scaled data.</li>
    <li><b>Explore</b> the data through visualizations; Document findings and takeaways.</li>
    <li>Perform <b>modeling</b>:
    <ul>
        <li>Identify model evaluation criteria</li>
        <li>Create at least three different models.</li>
        <li>Evaluate models on appropriate data subsets.</li>
    </ul>
    </li>
    <li>Create <b>Final Report</b> notebook with a curtailed version of the above steps.</li>
    <li>Create and review README. </li>
    
  </ul>
</details
