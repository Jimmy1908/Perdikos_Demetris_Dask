# demetris_perdikos_ind_python

This is the individual assignment for the Advanced Python Course
Repeating the assignment using Dask instead of Pandas

Most of the work done during the original assignment is included here, with the only ommisions being related to time series plotting and feature engineering.


**Executive summary 
For this project, we performed a study on the bike sharing demand of the company Capital Bike Sharing Washington D.C. between 2011 and 2012. Our aim was to predict the hourly demand between 1st of October 2012 and 31st of December 2012 based on the available hourly data from 1st of January 2011 to 30th of September 2012. In this report, we extensively discuss our approach by sharing our progress plan during three weeks, and giving insights on the approach we followed to come to a prediction of our target metric (R2squared score) of 88% with our preferred ensemble model.

**Data cleaning 
First, we displayed the info, head and null values of all three datasets. hour_data and day_data contain the same variables and will thus be treated similarly. In addition, they do not contain null values. When inspecting the location_data dataset, we notice that the variables cannot be joined with the hour_data. Therefore, we will solely use the location_data for a location density mapping for our client. 
 
hour_data and day_data were both loaded with a parsed dteday variable. The purpose of this is to be able to extract relevant elements from the itself not usable variable. One of our first steps therefore is extracting the day element [between 1 and 31] from the variable, on which we can later plot the average amount of hourly rented bikes. In addition, we included in our data cleaning section the modification of the seasons so that every season only includes one month. This we did according to the season division as found on Wikipedia.


**Plotting the graphs, we soon notice the graphs plotted with the day_data do not add a lot of additional value. 
 
On figures 2 and 3, the upward trend is analysed for both data sets. However, there is more structure captured from the hour_data, e.g. the drop in figure 3 of felt temperature around 0.5 is not shown in the corresponding day graph. Additionally, we have extracted the day variable in hour_data already so we can use this for modelling purposes. 

Also, since our aim is not to predict the daily demand of bikes, we decide to not include the dataset for our models. The only dataset we thus continue with for the modelling phase is the hour_data. The explanation of all the relevant graphs for our EDA can be found in the Summary EDA notebook. Non-relevant graphs for modelling insights, like the ones from day_data and the location_data are only included in the playground notebook. 
 
The visualization we produced with the location_dataset (figure 4) is a density plot, showing the number of bikes there are available per station across the D.C. area. We used the baseline library for this and imported the map of D.C. as an image with the build in function of arcgisimage. With this map, the stakeholder can quickly identify opportunities to install new bike sharing stations


**Data preparation 

After the EDA we start to, based on our insights, prepare the data to be scored in the baseline. The steps we follow can be found in these bullet points. 
 
● We tried out an approach to remodel the hours into five bins.
● We change the categories of the variables according to their actual data type.
● Normalization we approached by transforming the weather variables with the normalize function, included in the preprocessing library. ● We fixed the skewness with a fix skewness function
● One-hot encoded categorical variables and gave them meaningful column names
● Split dataset based on instant 15212, which represents the row with date 01/10/12, the start of the 4th quarter of 2012. We split the dataset also for target variables casual and registered, since we aim to construct models on them to approach the actual target variable count. 

**Baseline construction and pipeline 

● We construct a scoring function with a linear model and score all three target variables, with according training sets on this. 
