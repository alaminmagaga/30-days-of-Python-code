# 1
def count_lines_words(filename):
    with open(filename, 'r') as file:
        text = file.read()
        lines = text.count('\n') + 1
        words = len(text.split())
        return lines, words

# Example usage
obama_lines, obama_words = count_lines_words(r'C:\Users\Al Amin\Desktop\30-Days-of-Python-main\data\obama_speech.txt')
print(f"Obama speech: {obama_lines} lines, {obama_words} words")

michelle_lines, michelle_words = count_lines_words(r'C:\Users\Al Amin\Desktop\30-Days-of-Python-main\data\michelle_obama_speech.txt')
print(f"Michelle Obama speech: {michelle_lines} lines, {michelle_words} words")

donald_lines, donald_words = count_lines_words(r'C:\Users\Al Amin\Desktop\30-Days-of-Python-main\data\donald_speech.txt')
print(f"Donald Trump speech: {donald_lines} lines, {donald_words} words")

melania_lines, melania_words = count_lines_words(r'C:\Users\Al Amin\Desktop\30-Days-of-Python-main\data\melina_trump_speech.txt')
print(f"Melania Trump speech: {melania_lines} lines, {melania_words} words")

# 2
import json

def most_spoken_languages(filename, n):
    with open(filename) as f:
        data = json.load(f)
    
    languages = {}
    for country in data:
        for lang in country['languages']:
            if lang['name'] in languages:
                languages[lang['name']] += 1
            else:
                languages[lang['name']] = 1
    
    sorted_languages = sorted(languages.items(), key=lambda x: x[1], reverse=True)
    return sorted_languages[:n]

print(most_spoken_languages(filename='C:/Users/Al Amin/Desktop/30-Days-of-Python-main/data/countries_data.json', n=10))
print(most_spoken_languages(filename='C:/Users/Al Amin/Desktop/30-Days-of-Python-main/data/countries_data.json', n=3))


# 3
import json

def most_populated_countries(filename, n):
    with open(filename) as f:
        data = json.load(f)

    countries = []
    for country in data:
        country_data = {
            'country': country['name'],
            'population': country['population']
        }
        countries.append(country_data)

    countries.sort(key=lambda x: x['population'], reverse=True)
    return countries[:n]
    
print(most_spoken_languages(filename='C:/Users/Al Amin/Desktop/30-Days-of-Python-main/data/countries_data.json', n=10))
print(most_spoken_languages(filename='C:/Users/Al Amin/Desktop/30-Days-of-Python-main/data/countries_data.json', n=3))

# 4
import re

emails = []

with open(r'C:\Users\Al Amin\Desktop\30-Days-of-Python-main\data\email_exchanges_big.txt', 'r') as f:
    for line in f:
        matches = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', line)
        if matches:
            emails.extend(matches)

print(emails)

# 5
def find_most_common_words(file_name, n):
    word_counts = {}
    with open(file_name, 'r') as f:
        for line in f:
            words = line.split()
            for word in words:
                word = word.lower()
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_word_counts[:n]
print(find_most_common_words('sample.txt', 10))

# 6

def find_most_common_words(file_name, n):
    word_counts = {}
    with open(file_name, 'r') as f:
        for line in f:
            words = line.split()
            for word in words:
                word = word.lower()
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_word_counts[:n]

def read_text_file(file_name):
    with open(file_name, 'r') as f:
        text_data = f.read()
    return text_data

obama_file = r'C:\Users\Al Amin\Desktop\30-Days-of-Python-main\data\obama_speech.txt'
michelle_file = r'C:\Users\Al Amin\Desktop\30-Days-of-Python-main\data\michelle_obama_speech.txt'
trump_file = r'C:\Users\Al Amin\Desktop\30-Days-of-Python-main\data\donald_speech.txt'
melania_file = r'C:\Users\Al Amin\Desktop\30-Days-of-Python-main\data\melina_trump_speech.txt'


obama_text = read_text_file(obama_file)
michelle_text = read_text_file(michelle_file)
trump_text = read_text_file(trump_file)
melania_text = read_text_file(melania_file)


print('Obama:', find_most_frequent_words(obama_text, 10))
print('Michelle:', find_most_frequent_words(michelle_text, 10))
print('Trump:', find_most_frequent_words(trump_text, 10))
print('Melania:', find_most_frequent_words(melania_text, 10))


# 7


# 8
def find_most_repeated_words(file_name, n):
    word_counts = {}
    with open(file_name, 'r') as f:
        for line in f:
            words = line.split()
            for word in words:
                word = word.lower()
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_word_counts[:n]

print(find_most_repeated_words(r'C:\Users\Al Amin\Desktop\30-Days-of-Python-main\data\romeo_and_juliet.txt', 10))

# 9
import csv

# Define functions to count lines containing specific words
def count_python_lines(file_name):
    count = 0
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if 'python' in row[1].lower():
                count += 1
    return count

def count_javascript_lines(file_name):
    count = 0
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if 'javascript' in row[1].lower():
                count += 1
            elif 'javascript' in row[2].lower():
                count += 1
    return count

def count_java_not_javascript_lines(file_name):
    count = 0
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if 'java' in row[1].lower() and 'javascript' not in row[1].lower():
                count += 1
            elif 'java' in row[2].lower() and 'javascript' not in row[2].lower():
                count += 1
    return count

# Test the functions
print(count_python_lines(r'C:\Users\Al Amin\Desktop\30-Days-of-Python-main\data\hacker_news.csv'))
print(count_javascript_lines(r'C:\Users\Al Amin\Desktop\30-Days-of-Python-main\data\hacker_news.csv'))
print(count_java_not_javascript_lines(r'C:\Users\Al Amin\Desktop\30-Days-of-Python-main\data\hacker_news.csv'))
