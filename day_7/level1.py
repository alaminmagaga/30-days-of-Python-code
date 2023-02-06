# sets
it_companies = {'Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon'}
A = {19, 22, 24, 20, 25, 26}
B = {19, 22, 20, 25, 26, 24, 28, 27}

# Find the length of the set it_companies
print(len(it_companies))

# Add 'Twitter' to it_companies
it_companies.add('Twitter')
print(it_companies)

# Insert multiple IT companies at once to the set it_companies
it_companies.update(['Tesla', 'Snap', 'Uber'])
print(it_companies)

# Remove one of the companies from the set it_companies
it_companies.remove('Snap')
print(it_companies)

# What is the difference between remove and discard
# remove method raises an error if the item is not found in the set
it_companies.remove('Netflix') # raises KeyError: 'Netflix'

# while discard method doesn't raise any error if the item is not found in the set
it_companies.discard('Netflix') # doesn't raise any error