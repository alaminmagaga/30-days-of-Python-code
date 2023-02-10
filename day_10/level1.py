#1 Iterate 0 to 10 using for loop
for i in range(11):
    print(i)

#2 Iterate 10 to 0 using for loop
i = 0
while i < 11:
    print(i)
    i += 1

#3 loop that makes seven calls to print()
for i in range(1, 8):
    print("#" * i)

#4 Use nested loops
for i in range(8):
    for j in range(8):
        print("#", end=" ")
    print()

#5 Print patterns
for i in range(11):
    print(f"{i} x {i} = {i*i}")

#6 using a for loop and print out the items
list = ['Python', 'Numpy','Pandas','Django', 'Flask']
for item in list:
    print(item)

#7 loop to iterate from 0 to 100 and print only even numbers
for i in range(0, 101, 2):
    print(i)

#8 loop to iterate from 0 to 100 and print only odd numbers
for i in range(1, 101, 2):
    print(i)


