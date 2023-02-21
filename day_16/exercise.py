import datetime

# Get the current date and time
now = datetime.datetime.now()

# Extract the day, month, hour,minute and year
day = now.day
month = now.month
year = now.year
hour = now.hour
minute = now.minute

# Get the timestamp
timestamp = datetime.datetime.timestamp(now)

# Print the results
print("Current date and time is: ", now)
print("Day: ", day)
print("Month: ", month)
print("Year: ", year)
print("Hour: ", hour)
print("Minute: ", minute)
print("Timestamp: ", timestamp)


#2
# Format the current date
current_date = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
print("Current date: ", current_date)



# 3
time_str = "5 December, 2019"
time = datetime.datetime.strptime(time_str, "%d %B, %Y")
print("Time: ", time)

#4
new_year = datetime.datetime(2024, 1, 1)
time_until_new_year = new_year-datetime.datetime.now()
print("Time until New Year: ", time_until_new_year)


#5
epoch = datetime.datetime(1970, 1, 1)
time_since_epoch = datetime.datetime.now() - epoch
print("Time since 1 January 1970: ", time_since_epoch)

#6

#The datetime module in Python is a powerful tool that can be used for a variety of tasks related to date and time. Here are a few examples:

#Time series analysis: The datetime module can be used to manipulate time series data, such as analyzing trends in stock prices or weather patterns over time. The module provides functions to perform arithmetic on dates and times, as well as to format dates and times in various ways.

#Logging and tracking activities in an application: You can use the datetime module to get a timestamp of when an activity occurred in your application. This can be useful for tracking user behavior or debugging issues that occur at specific times. You can use the datetime.now() function to get the current date and time.

#Adding posts on a blog: The datetime module can be used to add timestamps to blog posts, indicating when they were published. This can be useful for organizing and searching through posts by date. You can use the datetime.now() function to get the current date and time, and then format it in a way that is suitable for your blog platform.

#In general, the datetime module is a very useful tool for any task that involves working with dates and times. It provides a wide range of functions for working with time zones, handling daylight saving time, and converting between different date and time formats.