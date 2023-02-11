# 1 the difference between map, filter, and reduce:
map: is a built-in Python function that takes two or more arguments: a function and one or more iterables, in the form: map(function, iterable, ...). The function is applied to each element of the iterables and returns a map object with the results.

filter: is also a built-in Python function that takes two arguments: a function and an iterable. The function is applied to each element of the iterable and the elements for which the function returns True are returned in a filter object.

reduce: is a function from the functools module in Python and it also takes two arguments: a function and an iterable. The function is applied cumulatively to the elements of the iterable, from left to right, so as to reduce the iterable to a single value.

# 2 the difference between higher order function, closure and decorator:
A higher-order function: is a function that takes a function as an argument or returns a function as output. map, filter, and reduce are examples of higher-order functions.

A closure: is a nested function that has access to variables in the containing function's scope, even after the containing function has finished executing. Closures are often used to preserve state, for example, in Python decorators.

A decorator :is a special type of higher-order function that is used to modify the behavior of another function. A decorator takes a function as input and returns a modified function, usually by wrapping the input function with additional code. Decorators are applied to a function using the @ syntax in Python.

# 3
def square(x):
    return x**2

numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(square, numbers))
print(squared_numbers)

def is_even(x):
    return x % 2 == 0

numbers = [1, 2, 3, 4, 5]
even_numbers = list(filter(is_even, numbers))
print(even_numbers)


# 4 print each country in the countries list
countries = ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for country in countries:
  print(country)

# 5   print each name in the names list
names = ['Asabeneh', 'Lidiya', 'Ermias', 'Abraham']
for name in names:
  print(name)

# 6 print each number in the numbers list.
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for number in numbers:
  print(number)
