#1 Declare an empty list
empty_list = []

#2 Declare a list with more than 5 items
colors = ['red', 'blue', 'green', 'yellow', 'purple', 'pink']

#3 Find the length of the list
print(len(colors))

#4 Get the first item, the middle item and the last item of the list
print(colors[0])
print(colors[len(colors)//2])
print(colors[-1])

#5 Declare a list called mixed_data_types, put your(name, age, height, marital status, address)
mixed_data_types = ['Asabeneh', 250, '5.11ft', 'single', 'Helsinki']

#6 Declare a list variable named it_companies and assign initial values Facebook, Google, Microsoft, Apple, IBM, Oracle and Amazon.
it_companies = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']

#7 Print the list using print()
print(it_companies)

#8 Print the number of companies in the list
print(len(it_companies))

#9 Print the first, middle and last company
print(it_companies[0])
print(it_companies[len(it_companies)//2])
print(it_companies[-1])

#10 Print the list after modifying one of the companies
it_companies[2] = 'Tesla'
print(it_companies)

#11 Add an IT company to it_companies
it_companies.append('Alphabet')
print(it_companies)

#12 Insert an IT company in the middle of the companies list
it_companies.insert(len(it_companies)//2, 'Spotify')
print(it_companies)

#13 Change one of the it_companies names to uppercase (IBM excluded!)
it_companies[0] = it_companies[0].upper()
print(it_companies)



#14 Join the it_companies with a string '#;  '
it_companies = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
it_companies_string = "#;  ".join(it_companies)
print(it_companies_string)

#15 Check if a certain company exists in the it_companies list
if 'Google' in it_companies:
    print('Google is in the list')
else:
    print('Google is not in the list')

#16 Sort the list using sort() method
it_companies.sort()
print(it_companies)

#17 Reverse the list in descending order using reverse() method
it_companies.reverse()
print(it_companies)

#18 Slice out the first 3 companies from the list
first_three = it_companies[:3]
print(first_three)

#19 Slice out the last 3 companies from the list
last_three = it_companies[-3:]
print(last_three)

#20 Slice out the middle IT company or companies from the list
middle = it_companies[len(it_companies)//2-1:len(it_companies)//2+2]
print(middle)

#21 Remove the first IT company from the list
it_companies.pop(0)
print(it_companies)

#22 Remove the middle IT company or companies from the list
middle_index = len(it_companies)//2
it_companies.pop(middle_index)
print(it_companies)

#23 Remove the last IT company from the list
it_companies.pop()
print(it_companies)

#24 Remove all IT companies from the list
it_companies.clear()
print(it_companies)

#25 Destroy the IT companies list
del it_companies


#26 Join the following lists:
front_end = ['HTML', 'CSS', 'JS', 'React', 'Redux']
back_end = ['Node','Express', 'MongoDB']
full_stack = front_end + back_end
print(full_stack)

#27 Insert Python and SQL after Redux
full_stack.insert(full_stack.index('Redux') + 1, 'Python')
full_stack.insert(full_stack.index('Python') + 1, 'SQL')
print(full_stack)

