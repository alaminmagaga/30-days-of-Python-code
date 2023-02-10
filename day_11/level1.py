import math

#1 a function add_two_numbers
def add_two_numbers(num1, num2):
    return num1 + num2

result = add_two_numbers(2, 3)
print(result) 

#2 a function to implement the function area_of_circle
def area_of_circle(radius):
    return math.pi * radius * radius

result = area_of_circle(10)
print(result) 

#3 a function called add_all_nums which takes arbitrary number of arguments and sums all the arguments
def add_all_nums(*args):
    sum = 0
    for arg in args:
        if type(arg) != int and type(arg) != float:
            return "Error: Only numbers are allowed as arguments."
        sum += arg
    return sum


#4 a function to converts °C to °F, convert_celsius_to-fahrenheit.
def convert_celsius_to_fahrenheit(celsius):
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit

celsius = 37
fahrenheit = convert_celsius_to_fahrenheit(celsius)
print(f"{celsius}°C is equal to {fahrenheit}°F")


#5 a function called check-season, it takes a month parameter and returns the season
def check_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn"
    else:
        return "Invalid Month"
result = check_season(2)
print(result)


#6 a function called calculate_slope which return the slope of a linear equation
def calculate_slope(x1, y1, x2, y2):
    # Calculate the rise
    rise = y2 - y1
    # Calculate the run
    run = x2 - x1
    # Calculate the slope
    slope = rise / run
    # Return the slope
    return slope
point1 = (1, 2)
point2 = (3, 4)
slope = calculate_slope(*point1, *point2)
print(slope) 

#7 a function which calculates solution set of a quadratic equation, solve_quadratic_eqn.
def solve_quadratic_eqn(a, b, c):
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return "No real solution exists."
    elif discriminant == 0:
        x = (-b + math.sqrt(discriminant)) / (2 * a)
        return x
    else:
        x1 = (-b + math.sqrt(discriminant)) / (2 * a)
        x2 = (-b - math.sqrt(discriminant)) / (2 * a)
        return (x1, x2)

solve_quadratic_eqn(4, 8, 4)


#8 a function named print_list. It takes a list as a parameter and it prints out each element of the list.
def print_list(list):
  for item in lst:
    print(item)

list = [1, 2, 3, 4, 5]
print_list(list)

#9 a function named reverse_list. It takes an array as a parameter and it returns the reverse of the array (use loops).
def reverse_list(arr):
    result = []
    for i in range(len(arr) - 1, -1, -1):
        result.append(arr[i])
    return result

print(reverse_list([1, 2, 3, 4, 5]))
print(reverse_list(["A", "B", "C"]))


#10 a function named capitalize_list_items. It takes a list as a parameter and it returns a capitalized list of items
def capitalize_list_items(list):
    capitalized_list = []
    for item in list:
        capitalized_list.append(item.capitalize())
    return capitalized_list
fruits = ['apple', 'banana', 'cherry']
print(capitalize_list_items(fruits))

#11 a function named add_item. It takes a list and an item parameters. It returns a list with the item added at the end.
def add_item(list, item):
    list.append(item)
    return list

food_staff = ['Potato', 'Tomato', 'Mango', 'Milk']
print(add_item(food_staff, 'Meat'))

#12 a function named remove_item. It takes a list and an item parameters. It returns a list with the item removed from it.

def remove_item(list, item):
    if item in list:
        list.remove(item)
        return list
    else:
        return "Item not found in the list"

food_staff = ['Potato', 'Tomato', 'Mango', 'Milk'];
print(remove_item(food_staff, 'Mango')) 
numbers = [2, 3, 7, 9]
print(remove_item(numbers, 3))  

#13 a function named sum_of_numbers. It takes a number parameter and it adds all the numbers in that range.
def sum_of_numbers(number):
    return sum(range(1, number + 1))

print(sum_of_numbers(5))  
print(sum_of_numbers(10)) 
print(sum_of_numbers(100)) 

#14  a function named sum_of_odds. It takes a number parameter and it adds all the odd numbers in that range.
def sum_of_odds(num):
    sum = 0
    for i in range(1,num+1):
        if i%2 != 0:
            sum += i
    return sum

print(sum_of_odds(5)) 
print(sum_of_odds(10)) 
print(sum_of_odds(100)) 




#15 a function named sum_of_even. It takes a number parameter and it adds all the even numbers in that - range.
def sum_of_even(number):
    total = 0
    for i in range(number+1):
        if i % 2 == 0:
            total += i
    return total

print(sum_of_even(5)) 
print(sum_of_even(10))
print(sum_of_even(100))

