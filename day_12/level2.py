import random
import string
# 1

def list_of_hexa_colors(num_colors):
    hexa_colors = []
    for i in range(num_colors):
        color = '#'
        for j in range(6):
            color += random.choice(string.hexdigits)
        hexa_colors.append(color)
    return hexa_colors

print(list_of_hexa_colors(10))

# 2
def list_of_rgb_colors(n):
    colors = []
    for i in range(n):
        red = int(input("Enter red value (0-255): "))
        green = int(input("Enter green value (0-255): "))
        blue = int(input("Enter blue value (0-255): "))
        color = (red, green, blue)
        colors.append(color)
    return colors
print(list_of_rgb_colors(10))

# 3 
def generate_colors(color_type, count):
    colors = []
    if color_type == 'hexa':
        for i in range(count):
            hexa_color = '#' + ''.join([random.choice('0123456789abcdef') for j in range(6)])
            colors.append(hexa_color)
    elif color_type == 'rgb':
        for i in range(count):
            rgb_color = 'rgb(' + ','.join([str(random.randint(0,255)) for j in range(3)]) + ')'
            colors.append(rgb_color)
    else:
        print("Invalid color type")
        return
    return colors

print(generate_colors('hexa', 3))
print(generate_colors('hexa', 1))
print(generate_colors('rgb', 3))
print(generate_colors('rgb', 1))

