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

