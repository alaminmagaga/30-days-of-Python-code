# 1
age = 25

# 2
height = 1.75

# 3
complex_number = 2 + 3j


print("Age:", age)
print("Height:", height)
print("Complex number:", complex_number)

# 4
base = float(input("Enter base: "))
height = float(input("Enter height: "))
area = 0.5 * base * height

print("The area of the triangle is", area)


#5 Triangle
print("Triangle Perimeter Calculation")
a = int(input("Enter side a: "))
b = int(input("Enter side b: "))
c = int(input("Enter side c: "))
perimeter = a + b + c
print("The perimeter of the triangle is", perimeter)

#6 Rectangle
print("\nRectangle Area and Perimeter Calculation")
length = int(input("Enter length: "))
width = int(input("Enter width: "))
area = length * width
perimeter = 2 * (length + width)
print("The area of the rectangle is", area)
print("The perimeter of the rectangle is", perimeter)

#7 Circle
print("\nCircle Area and Circumference Calculation")
radius = int(input("Enter radius: "))
pi = 3.14
area = pi * radius * radius
circumference = 2 * pi * radius
print("The area of the circle is", area)
print("The circumference of the circle is", circumference)

#8 Line
print("Line Slope, x-intercept, y-intercept Calculation")
print("y = 2x - 2")
slope = 2
x_intercept = 0
y_intercept = -2
print("Slope:", slope)
print("x-intercept:", x_intercept)
print("y-intercept:", y_intercept)

#9 Euclidean Distance
print("Euclidean Distance Calculation")
x1, y1 = 2, 2
x2, y2 = 6, 10
distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
print("the Euclidean Distance between points (2, 2) and (6, 10) is", distance)


# 10

#11 y = x^2 + 6x + 9
import cmath
print("y = x^2 + 6x + 9")
def calculate_y(x):
  y = x**2 + 6*x + 9
  return y

x = int(input("Enter a value for x: "))
y = calculate_y(x)
print("For x =", x, ", y =", y)

# Finding x such that y = 0


def find_x(a, b, c):
  x1 = (-b + cmath.sqrt(b**2 - 4*a*c)) / (2*a)
  x2 = (-b - cmath.sqrt(b**2 - 4*a*c)) / (2*a)
  return x1, x2

a = 1
b = 6
c = 9
x1, x2 = find_x(a, b, c)
print("The roots of the equation y = 0 are x1 =", x1, "and x2 =", x2)


#12 Length of strings
python = "python"
dragon = "dragon"
print("Length of 'python':", len(python))
print("Length of 'dragon':", len(dragon))

#13 Falsy comparison statement
if len(python) != len(dragon):
  print("Lengths of 'python' and 'dragon' are not equal.")

#14 Checking if 'on' is in both 'python' and 'dragon'
if 'on' in python and 'on' in dragon:
  print("'on' is found in both 'python' and 'dragon'.")
else:
  print("There is no 'on' in both 'python' and 'dragon'.")

#15 Checking if 'jargon' is in the sentence
sentence = "I hope this course is not full of jargon."
if 'jargon' in sentence:
  print("'jargon' is in the sentence.")
else:
  print("'jargon' is not in the sentence.")

#16 Converting the length of 'python' to float and string
length = len(python)
float_length = float(length)
string_length = str(length)
print("Length of 'python' as float:", float_length)
print("Length of 'python' as string:", string_length)

#17 Checking if a number is even
number = int(input("Enter a number: "))
if number % 2 == 0:
  print(number, "is an even number.")
else:
  print(number, "is an odd number.")


#18 Floor division and int conversion comparison
if 7 // 3 == int(2.7):
  print("The floor division of 7 by 3 is equal to the int converted value of 2.7.")
else:
  print("The floor division of 7 by 3 is not equal to the int converted value of 2.7.")

#19 Type comparison
if type('10') == type(10):
  print("Type of '10' is equal to type of 10.")
else:
  print("Type of '10' is not equal to type of 10.")

#20 Int conversion and comparison
if int('9.8') == 10:
  print("int('9.8') is equal to 10.")
else:
  print("int('9.8') is not equal to 10.")

#21 Calculating pay
hours = int(input("Enter hours: "))
rate = int(input("Enter rate per hour: "))
pay = hours * rate
print("Your weekly earning is", pay)

#22 Calculating seconds a person can live
years = int(input("Enter number of years you have lived: "))
seconds = years * 365 * 24 * 60 * 60
print("You have lived for", seconds, "seconds.")

#23 Displaying the table
for i in range(1, 6):
  print(*[j**i for j in range(1, 6)])

