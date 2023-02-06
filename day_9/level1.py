age = int(input("Enter your age: "))
if age >= 18:
    print("You are old enough to drive.")
else:
    years_left = 18 - age
    print(f"You need {years_left} more years to learn to drive.")


my_age = 20
your_age = int(input("Enter your age: "))

if my_age < your_age:
    age_difference = your_age - my_age
    if age_difference == 1:
        print(f"You are 1 year older than me.")
    else:
        print(f"You are {age_difference} years older than me.")
elif my_age > your_age:
    age_difference = my_age - your_age
    if age_difference == 1:
        print(f"I am 1 year older than you.")
    else:
        print(f"I am {age_difference} years older than you.")
else:
    print("We are the same age.")

a = int(input("Enter number one: "))
b = int(input("Enter number two: "))

if a > b:
    print(f"{a} is greater than {b}")
elif a < b:
    print(f"{a} is smaller than {b}")
else:
    print(f"{a} is equal to {b}")