# Convert the ages to a set
ages = [19, 22, 19, 24, 20, 25, 26, 24, 25, 24]
ages_set = set(ages)

#1 Compare the length of the list and the set
if len(ages) > len(ages_set):
    print("The list is bigger.")
else:
    print("The set is bigger.")


# Convert the ages list to a set
ages = [19, 22, 19, 24, 20, 25, 26, 24, 25, 24]
unique_ages = set(ages)

# Compare the length of the list and the set
if len(ages) > len(unique_ages):
    print("The list is bigger")
else:
    print("The set is bigger")


#2 Explanation of the difference between the data types
"""
String is An ordered sequence of characters, which can be any text. Strings are immutable, meaning their values cannot be changed after they are created.

List is An ordered collection of items, which can be of any data type including other lists. Lists are mutable, meaning their values can be changed after they are created.

Tupleis An ordered collection of items, which can be of any data type including other tuples. Tuples are immutable, meaning their values cannot be changed after they are created.

Setis An unordered collection of unique items, which can be of any data type but not including other sets. Sets are mutable, meaning their values can be changed after they are created.
"""
# 3
sentence = "I am a teacher and I love to inspire and teach people."
words = sentence.split()
unique_words = set(words)
print("Number of unique words:", len(unique_words))