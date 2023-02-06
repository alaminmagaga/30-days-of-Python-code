
# Create an empty tuple
empty_tuple = ()

# Create a tuple containing names of your sisters and your brothers (imaginary siblings are fine)
sisters = ('maryam', 'hauwa')
brothers = ('musty', 'khalifa')

# Join brothers and sisters tuples and assign it to siblings
siblings = sisters + brothers

# How many siblings do you have?
num_siblings = len(siblings)

# Modify the siblings tuple and add the name of your father and mother and assign it to family_members
father = ('Musa')
mother = ('Bilkisu')
family_members = siblings + father + mother



# LEVEL 2


# Unpacking siblings and parents from family_members
father, mother, *siblings = family_members

# Creating fruits, vegetables, and animal products tuples
fruits = ('apple', 'banana', 'pear')
vegetables = ('carrot', 'lettuce', 'potato')
animal_products = ('milk', 'cheese', 'eggs')

# Joining the three tuples into food_stuff_tp
food_stuff_tp = fruits + vegetables + animal_products

# Converting food_stuff_tp tuple to a food_stuff_lt list
food_stuff_lt = list(food_stuff_tp)

# Slicing out the middle item(s) from food_stuff_tp or food_stuff_lt
middle_index = len(food_stuff_lt) // 2
middle_items = food_stuff_lt[middle_index - 1: middle_index + 1]

# Slicing out the first three items and the last three items from food_stuff_lt
first_three = food_stuff_lt[:3]
last_three = food_stuff_lt[-3:]

# Deleting the food_stuff_tp tuple completely
del food_stuff_tp

nordic_countries = ('Denmark', 'Finland','Iceland', 'Norway', 'Sweden')
if 'Iceland' in nordic_countries:
    print("Iceland is a Nordic country")
else:
    print("Iceland is not a Nordic country")