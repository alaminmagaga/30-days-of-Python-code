def get_grade(score):
    if score >= 80 and score <= 100:
        return "A"
    elif score >= 70 and score <= 89:
        return "B"
    elif score >= 60 and score <= 69:
        return "C"
    elif score >= 50 and score <= 59:
        return "D"
    else:
        return "F"

def get_season(month):
    if month in ['September', 'October', 'November']:
        return "Autumn"
    elif month in ['December', 'January', 'February']:
        return "Winter"
    elif month in ['March', 'April', 'May']:
        return "Spring"
    elif month in ['June', 'July', 'August']:
        return "Summer"
    else:
        return "Invalid month"

fruits = ['banana', 'orange', 'mango', 'lemon']

def add_fruit(fruit):
    if fruit in fruits:
        print('That fruit already exists in the list')
    else:
        fruits.append(fruit)
        print('The modified list of fruits:', fruits)
