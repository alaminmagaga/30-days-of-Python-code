import datetime

# 1
now = datetime.datetime.now()

day = now.day
month = now.month
year = now.year
hour = now.hour
minute = now.minute
timestamp = datetime.datetime.timestamp(now)

print("Current date and time is: ", now)
print("Day: ", day)
print("Month: ", month)
print("Year: ", year)
print("Hour: ", hour)
print("Minute: ", minute)
print("Timestamp: ", timestamp)

# 2


now = datetime.datetime.now()
formatted_date = now.strftime("%m/%d/%Y, %H:%M:%S")
print("Formatted Date: ", formatted_date)

# 3

time_str = "5 December, 2019"
time_obj = datetime.datetime.strptime(time_str, "%d %B, %Y")
print("Time object: ", time_obj)

# 4

new_year = datetime.datetime(current_year+1, 1, 1)
time_diff = new_year - now
print("Time difference to New Year: ", time_diff)


# 5

from datetime import datetime

start_time = datetime(1970, 1, 1)
current_time = datetime.now()

time_diff = current_time - start_time
print("Time difference between 1 Jan 1970 and now is:", time_diff)

# # 6

# the datetime module can be used for various tasks related to date and time, such as:

# Time series analysis: You can use the datetime module to analyze time series data, such as stock prices, weather data, or website traffic.

# Timestamping: You can use the datetime module to get timestamps of various activities in an application, such as logging events, recording when a user signs up, or tracking user activity.

# Blogging: You can use the datetime module to add timestamps to blog posts, display the time of publication, and sort posts chronologically.

# Scheduling: You can use the datetime module to schedule tasks, such as sending email reminders, running backups, or updating data.

# Data processing: You can use the datetime module to process data that includes date and time information, such as sales data, customer behavior data, or sensor data.