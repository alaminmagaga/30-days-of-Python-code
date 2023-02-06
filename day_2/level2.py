#1 check data types of the variables


first_name="Alamin"   # str
last_name="musa"      # str
full_name="Alamin Musa Magaga"    # str
country="Nigeria"
city= 'Helsinki'        # str
age = 250              # int
year=2023              # str
is_married=False       # bool
is_true=False               # bool
is_light_on=True               # bool
school, state, faculty= 'BUK', 'Kano', 'Engineering'



print(type(first_name))   # str
print(type(last_name))    # str
print(type(full_name))     # str
print(type(country))      # str
print(type(city))         # str
print(type(age))          # int
print(type(year))           # int
print(type(is_married))       # bool
print(type(is_true))            # bool
print(type(is_light_on))             # bool


print(type(school))
print(type(state))
print(type(faculty))


#2  length of my first name
print(len(first_name))


#3 Compare the length of my first  and  last name
print("the length of my first name is ",len(first_name)," while the length of my last name is ",len(last_name))

#3i Declare 5 as num_one and 4 as num_two
num_one = 5
num_two = 4

total = num_one + num_two
print("The total is:", total)

#3ii Difference between num_one and num_two
num_one = 5
num_two = 4

diff = num_one - num_two
print("the difference is:", diff)


#3iii the product of num_one and num_two
num_one = 5
num_two = 4

product = num_one * num_two
print("the product is:", product)


#3iv the division of num_one by num_two
num_one = 5
num_two = 4

division = num_one / num_two
print("the division is:", division)

#3v the modulus of num_one and num_two
num_one = 5
num_two = 4

remainder = num_two % num_one
print("the remainder is:", remainder)

#3vi the exponential num_one and num_two
num_one = 5
num_two = 4

exp = num_one ** num_two
print("the Exponentiation is:", exp)

#3vii the floor division between num_one and num_two
num_one = 5
num_two = 4

floor_division = num_one // num_two
print("Floor Division:", floor_division)


#4 RADIUS OF CIRCLE
import math

radius = float(input("Enter the radius of the circle: "))

area_of_circle = math.pi * (radius ** 2)
circum_of_circle = 2 * math.pi * radius

print("The Area of the circle:", area_of_circle, "square meters")
print("The Circumference of the circle:", circum_of_circle, "meters")


#5 built-in input function to get first name, last name, country and age from a user
first_name = input("Enter your first name: ")
last_name = input("Enter your last name: ")
country = input("Enter your country: ")
age = int(input("Enter your age: "))

print("Your first name is", first_name)
print("Your last name is", last_name)
print("Your country is", country)
print("Your age is", age)


#6 Run help('keywords') in Python shell
import keyword
print(keyword.kwlist)



