# 1
numbers = [-4, -3, -2, -1, 0, 2, 4, 6]
negative_and_zero = [num for num in numbers if num <= 0]
print(negative_and_zero)

# 2
list_of_lists =[[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]]
flattened_list = [item for sublist in list_of_lists for sub_sublist in sublist for item in sub_sublist]
print(flattened_list)

# 3
tuples_list = [(i, 1, i**2, i**3, i**4, i**5, i**6) for i in range(11)]
print(tuples_list)

# 4
countries = [[('Finland', 'Helsinki')], [('Sweden', 'Stockholm')], [('Norway', 'Oslo')]]
flattened_countries = [[country[0][0].upper(), country[0][0][:3].upper(), country[0][1].upper()] for country in countries]
print(flattened_countries)

# 5
def convert_to_dict(countries):
    result = []
    for country in countries:
        country_dict = {}
        country_dict["country"] = country[0][0].upper()
        country_dict["city"] = country[0][1].upper()
        result.append(country_dict)
    return result

countries = [[('Finland', 'Helsinki')], [('Sweden', 'Stockholm')], [('Norway', 'Oslo')]]
print(convert_to_dict(countries))

# 6
calculate = lambda x1, y1, x2, y2, mode: (mode == "slope") and ((y2 - y1) / (x2 - x1)) or ((mode == "y-intercept") and (y1 - ((y2 - y1) / (x2 - x1)) * x1))
print(calculate(0, 0, 1, 1, "slope"))
print(calculate(0, 0, 1, 1, "y-intercept"))
