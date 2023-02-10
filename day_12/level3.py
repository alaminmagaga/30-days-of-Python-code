import random

# 1
def shuffle_list(list_to_shuffle):
    random.shuffle(list_to_shuffle)
    return list_to_shuffle

my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(shuffle_list(my_list))

# 2
import random

def random_unique_numbers():
    numbers = list(range(10))
    random.shuffle(numbers)
    return numbers[:7]

print(random_unique_numbers())

