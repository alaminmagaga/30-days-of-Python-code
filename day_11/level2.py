# 1 a function named evens_and_odds . It takes a positive integer as parameter and it counts number of evens and odds in the number.
def evens_and_odds(n):
    evens = 0
    odds = 0
    for i in range(1, n + 1):
        if i % 2 == 0:
            evens += 1
        else:
            odds += 1
    print("The number of odds are", odds)
    print("The number of evens are", evens)
    print(evens_and_odds(100))

#2 Call your function factorial, it takes a whole number as a parameter and it return a factorial of the number
def factorial(number):
    if number == 0:
        return 1
    else:
        return number * factorial(number-1)
print(factorial(5))


# 3 Call your function is_empty, it takes a parameter and it checks if it is empty or not
def is_empty(obj):
    if not obj:
        return True
    else:
        return False

print(is_empty("")) 
print(is_empty("Hello"))
print(is_empty([]))
print(is_empty([1, 2, 3]))


# calculate_mean, calculate_median, calculate_mode, calculate_range, calculate_variance, calculate_std (standard deviation).

def calculate_mean(numbers):
    return sum(numbers) / len(numbers)

def calculate_median(numbers):
    numbers.sort()
    mid = len(numbers) // 2
    if len(numbers) % 2 == 0:
        return (numbers[mid - 1] + numbers[mid]) / 2
    else:
        return numbers[mid]

def calculate_mode(numbers):
    from collections import Counter
    c = Counter(numbers)
    mode = [k for k, v in c.items() if v == max(list(c.values()))]
    if len(mode) == len(numbers):
        return None
    else:
        return mode

def calculate_range(numbers):
    return max(numbers) - min(numbers)

def calculate_variance(numbers):
    mean = calculate_mean(numbers)
    return sum([(i - mean) ** 2 for i in numbers]) / len(numbers)

def calculate_std(numbers):
    return calculate_variance(numbers) ** 0.5

numbers = [2, 3, 7, 9, 10, 10]
print("Mean: ", calculate_mean(numbers))
print("Median: ", calculate_median(numbers))
print("Mode: ", calculate_mode(numbers))
print("Range: ", calculate_range(numbers))
print("Variance: ", calculate_variance(numbers))
print("Standard Deviation: ", calculate_std(numbers))
5


