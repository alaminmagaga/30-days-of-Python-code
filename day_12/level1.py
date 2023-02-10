import random
import string

def random_user_id():
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(6))

def user_id_gen_by_user():
    num_chars = int(input("Enter the number of characters for the user ID: "))
    num_ids = int(input("Enter the number of IDs to generate: "))
    characters = string.ascii_letters + string.digits
    for i in range(num_ids):
        print(''.join(random.choice(characters) for j in range(num_chars)))

def rgb_color_gen():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return "rgb({}, {}, {})".format(r, g, b)

print(random_user_id())
print(user_id_gen_by_user())
print(rgb_color_gen())

