from functions import *

"""
From the raw data containing date, worker name, speed, and material worked with, you can derive several statistical insights. Here are some suggestions:

1. **Descriptive Statistics:**
   - Calculate the mean, median, mode, and range of worker speeds.
   - Compute summary statistics such as minimum, maximum, standard deviation, and quartiles of worker speeds.

2. **Performance Trends:**
   - Analyze trends in worker speeds over time (using the date field). This could involve plotting a line chart or time series plot to visualize any patterns or changes in performance over time.
   - Investigate whether there are any seasonal patterns in worker speeds.

3. **Worker Comparison:**
   - Compare the average speeds of different workers. This could involve calculating summary statistics for each worker and comparing them.
   - Use box plots or histograms to visualize the distribution of speeds for each worker.

4. **Material Analysis:**
   - Analyze the speed of work with different materials. You can calculate summary statistics for each material and compare them.
   - Visualize the distribution of worker speeds for each material using histograms or box plots.

5. **Correlation Analysis:**
   - Investigate if there is any correlation between worker speed and other variables such as the type of material being worked with. You can use correlation coefficients like Pearson's correlation or Spearman's rank correlation.
   - Check if there's any correlation between worker speed and time (e.g., if workers tend to work faster or slower at different times of the day or week).

6. **Outlier Detection:**
   - Identify any outliers in the data, such as unusually high or low worker speeds. These outliers could indicate potential issues or exceptional performance that may need further investigation.

7. **Predictive Modeling:**
   - Build predictive models to forecast future worker speeds based on historical data. This could involve techniques such as linear regression or time series forecasting.

8. **Quality Control:**
   - Use statistical process control techniques to monitor and control the quality of work. For example, you could set control limits based on historical data and flag any worker speeds that fall outside these limits for further review.

Remember to choose the appropriate statistical techniques based on the nature of your data and the specific questions you want to answer. Additionally, visualizations can often provide valuable insights, so consider using graphs and charts to complement your statistical analysis.
"""

path = "202403 Raw data.xlsx"
#line_plots(path)

df = excel_to_dateframe(path, "Poslije stimulacija", "Datum")
# workers = unique_values(df,"Ime")
avg_person = averages_per_person(df)
avg_process = averages_per_process(df)
#print(avg_person)
#bar_chart("Snježana Matek", avg_person)
grouped_bar_chart("Antonija Kerhlanko", avg_person, avg_process)
# # for worker in workers:
# #    bar_chart(worker, avg_person)

# print(number_of_workers_by_std_dev(df))