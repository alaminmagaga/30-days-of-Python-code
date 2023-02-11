# 1
countries = ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
upper_case_countries = list(map(lambda x: x.upper(), countries))
print(upper_case_countries)


# 2
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squared_numbers = list(map(lambda x: x**2, numbers))
print(squared_numbers)


# 3
names = ['Asabeneh', 'Lidiya', 'Ermias', 'Abraham']
upper_case_names = list(map(lambda x: x.upper(), names))
print(upper_case_names)


# 4
countries = ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
filtered_countries = list(filter(lambda x: "land" in x, countries))
print(filtered_countries)

# 5
countries = ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
filtered_countries = list(filter(lambda x: len(x) == 6, countries))
print(filtered_countries)

# 6
countries = ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
filtered_countries = list(filter(lambda x: len(x) >= 6, countries))
print(filtered_countries)


# 7
countries = ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
filtered_countries = list(filter(lambda x: x.startswith("E"), countries))
print(filtered_countries)

# 8
from functools import reduce

def get_string_lists(arr):
  return list(filter(lambda x: isinstance(x, str), arr))

arr = ['Estonia', 1, 'Finland', 2, 'Sweden', 'Denmark', 3, 'Norway', 'Iceland']
string_lists = get_string_lists(arr)
print(string_lists)


# 9
from functools import reduce

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sum_of_numbers = reduce(lambda x, y: x + y, numbers)
print(sum_of_numbers)


# 10
from functools import reduce

countries = ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
sentence = reduce(lambda x, y: x + ', ' + y, countries) + ' are north European countries'
print(sentence)

