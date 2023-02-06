# Create an empty dictionary called dog
dog = {}

# Add name, color, breed, legs, age to the dog dictionary
dog['name'] = 'Puppy'
dog['color'] = 'Bull Dog'
dog['breed'] = 'Labrador'
dog['legs'] = 4
dog['age'] = 5

# Create a student dictionary
student = {}
student['first_name'] = 'Alamin'
student['last_name'] = 'Musa'
student['gender'] = 'Male'
student['age'] = 25
student['marital_status'] = 'single'
student['skills'] = ['python', 'javascript']
student['country'] = 'Nigeria'
student['city'] = 'kano'
student['address'] = 'Q16 Kontagora Road Kaduna'

# Get the length of the student dictionary
len_student = len(student)

# Get the value of skills and check the data type
skills = student['skills']
print(type(skills))

# Modify the skills values by adding one or two skills
student['skills'].append('c++')
student['skills'].append('ruby')

# Get the dictionary keys as a list
keys = list(student.keys())

# Get the dictionary values as a list
values = list(student.values())

# Change the dictionary to a list of tuples using items() method
student_items = list(student.items())

# Delete one of the items in the dictionary
del student['address']

# Delete one of the dictionaries
del student
student = {'first_name': 'Alamin', 'last_name': 'Musa', 'gender': 'Male', 'age': 25, 'marital_status': 'single', 'skills': ['python', 'data science'], 'country': 'Nigeria', 'city': 'Kaduna', 'address': 'Q16 kontagora Road'}
del student['age']

del student