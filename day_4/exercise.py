
#1 Concatenating the string 'Thirty', 'Days', 'Of', 'Python' to a single string, 'Thirty Days Of Python'
print('Thirty' + ' ' + 'Days' + ' ' + 'Of' + ' ' + 'Python') 

#2 Concatenating the string 'Coding', 'For' , 'All' to a single string, 'Coding For All'
print('Coding' + ' ' + 'For' + ' ' + 'All') 

#3 Declaring a variable named company and assign it to an initial value "Coding For All"
company = "Coding For All"

#4 Printing the variable company using print()
print(company)

#5 Printing the length of the company string using len() method and print()
print(len(company)) 

#6 Changing all the characters to uppercase letters using upper() method
print(company.upper())

#7 Changing all the characters to lowercase letters using lower() method
print(company.lower())

#8 Using capitalize(), title(), swapcase() methods to format the value of the string Coding For All
print(company.capitalize())
print(company.title())
print(company.swapcase())

#9 Cutting(slicing) out the first word of Coding For All string
print(company[7:]) 

#10 Checking if Coding For All string contains a word Coding using the method index, find or other methods
print('Coding' in company) # True

#11 Replacing the word coding in the string 'Coding For All' to Python
print(company.replace('Coding', 'Python')) 

#12 Changing Python for Everyone to Python for All using the replace method or other methods
python_for_everyone = "Python for Everyone"
print(python_for_everyone.replace('Everyone', 'All')) 

#13 Splitting the string 'Coding For All' using space as the separator (split())
split_string = company.split()
print(split_string) 

# 14 Splitting the string "Facebook, Google, Microsoft, Apple, IBM, Oracle, Amazon" at the comma
split_string = "Facebook, Google, Microsoft, Apple, IBM, Oracle, Amazon".split(', ')
print(split_string)

#15 Finding the character at index 0 in the string Coding For All
print(company[0]) 

#16 Finding the last index of the string Coding For All
print(len(company) - 1) # 11

#17 Finding the character at index 10 in "Coding For All" string
print(company[10])

#18 Creating an acronym or an abbreviation for the name 'Python For Everyone'
acronym = "".join([word[0] for word in python_for_everyone.split()])
print(acronym) 

#19 Creating an acronym or an abbreviation for the name 'Coding For All'.
name = "Coding For All"

#20 Index of first occurrence of 'C'
print(name.index("C")) 

#21 Index of first occurrence of 'F'
print(name.index("F")) 

#22 Position of last occurrence of 'l'
print(name.rfind("l"))


#23 Position of first occurrence of 'because'
sentence = "You cannot end a sentence with because because because is a conjunction"
print(sentence.find("because")) 

#24 Position of last occurrence of 'because'
print(sentence.rindex("because"))

#25 Slice out the phrase 'because because because'
print(sentence[21:37]) 

#26 Position of first occurrence of 'because'
print(sentence.find("because"))

#27 Slice out the phrase 'because because because'
print(sentence[21:37]) 

#28 Check if 'Coding For All' starts with 'Coding'
print(name.startswith("Coding")) 

#29 Check if 'Coding For All' ends with 'coding'
print(name.endswith("coding"))

#30 Remove left and right trailing spaces
name = "   Coding For All      "
print(name.strip()) 


#31 Check if variables are valid identifier
var1 = '30DaysOfPython'
var2 = 'thirty_days_of_python'
print(var1.isidentifier()) # False
print(var2.isidentifier()) # True

#32 Join list of strings with ' # '
lst = ['Django', 'Flask', 'Bottle', 'Pyramid', 'Falcon']
lst_str = ' # '.join(lst)
print(lst_str) 

#33 New line escape sequence
str3 = "I am enjoying this challenge.\nI just wonder what is next."
print(str3)

#34 Tab escape sequence
str4 = "Name\tAge\tCountry\tCity\nAsabeneh\t250\tFinland\tHelsinki"
print(str4)


#35 String formatting with radius and area variables
radius = 10
area = 3.14 * radius ** 2
result = "The area of a circle with radius {} is {} meters square.".format(radius, area)
print(result)

#36 String formatting for arithmetic operations
add = 8 + 6
sub = 8 - 6
mul = 8 * 6
div = 8 / 6
mod = 8 % 6
floor_div = 8 // 6
exp = 8 ** 6
str5 = "8 + 6 = {}\n8 - 6 = {}\n8 * 6 = {}\n8 / 6 = {:.2f}\n8 % 6 = {}\n8 // 6 = {}\n8 ** 6 = {}".format(add, sub, mul, div, mod, floor_div, exp)
print(str5)


