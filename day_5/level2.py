ages = [19, 22, 19, 24, 20, 25, 26, 24, 25, 24]

# Sort the list
ages.sort()

# Find the min and max age
min_age = min(ages)
max_age = max(ages)

# Add the min age and the max age again to the list
ages.append(min_age)
ages.append(max_age)

# Find the median age (one middle item or two middle items divided by two)
length = len(ages)
if length % 2 == 0:
    median = (ages[length//2 - 1] + ages[length//2]) / 2
else:
    median = ages[length//2]

# Find the average age (sum of all items divided by their number)
average = sum(ages) / length

# Find the range of the ages (max minus min)
range_of_ages = max_age - min_age

# Compare the value of (min - average) and (max - average), use abs() method
min_avg_diff = abs(min_age - average)
max_avg_diff = abs(max_age - average)

# Find the middle country(ies) in the countries list
countries = ['China', 'Russia', 'USA', 'Finland', 'Sweden', 'Norway', 'Denmark']
length = len(countries)
if length % 2 == 0:
    middle = (countries[length//2 - 1] + countries[length//2]) / 2
else:
    middle = countries[length//2]

# Divide the countries list into two equal lists if it is even if not one more country for the first half.
half = length // 2
first_half = countries[:half + length % 2]
second_half = countries[half + length % 2:]

# Unpack the first three countries and the rest as scandic countries.
first_three_countries, *scandic_countries = countries
