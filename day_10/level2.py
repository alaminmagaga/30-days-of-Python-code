#1 Using for loop to iterate from 0 to 100 and print the sum of all numbers.
sum = 0
for i in range(101):
    sum += i
print("The sum of all numbers is", sum)


#2 Using for loop to iterate from 0 to 100 and print the sum of all evens and the sum of all odds.
even_sum = 0
odd_sum = 0
for i in range(101):
    if i % 2 == 0:
        even_sum += i
    else:
        odd_sum += i
print("The sum of all evens is", even_sum)
print("The sum of all odds is", odd_sum)
