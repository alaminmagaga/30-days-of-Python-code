# 1
import requests
import string
from collections import Counter

url = 'http://www.gutenberg.org/files/1112/1112.txt'
response = requests.get(url)
data = response.text

# Convert the text to lowercase
data = data.lower()

# Remove all the punctuation marks
data = data.translate(str.maketrans('', '', string.punctuation))
words = data.split()

#  Filter out the stop words
stop_words = ['the', 'and', 'to', 'of', 'that', 'is', 'in', 'a', 'not', 'you']
filtered_words = [word for word in words if word not in stop_words]

word_counts = Counter(filtered_words)

sorted_word_counts = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True))
print(list(sorted_word_counts.items())[:10])

# 2
import requests
import statistics

cats_api = 'https://api.thecatapi.com/v1/breeds'

# Read the data from the API
response = requests.get(cats_api)
data = response.json()

weights = []
lifespans = []
country_breed = {}

for cat in data:
    if cat.get('weight') and cat.get('weight').get('metric'):
        weight = cat['weight']['metric'].split()[0]
        weights.append(float(weight))
    
    if cat.get('life_span'):
        lifespan = cat['life_span'].split()[0]
        if lifespan.isdigit():
            lifespans.append(float(lifespan))
    
    if cat.get('origin') and cat.get('name'):
        origin = cat['origin'].strip().lower()
        breed = cat['name'].strip().lower()
        if origin and breed:
            country_breed.setdefault(origin, []).append(breed)

#  Calculate the statistics for weight and lifespan
weight_min = min(weights)
weight_max = max(weights)
weight_mean = sum(weights) / len(weights)
weight_median = statistics.median(weights)
weight_stdev = statistics.stdev(weights)

lifespan_min = min(lifespans)
lifespan_max = max(lifespans)
lifespan_mean = sum(lifespans) / len(lifespans)
lifespan_median = statistics.median(lifespans)
lifespan_stdev = statistics.stdev(lifespans)

# frequency table of country and breed of cats
country_breed_freq = {}
for country, breeds in country_breed.items():
    breed_freq = Counter(breeds)
    country_breed_freq[country] = breed_freq


print("Weight statistics:")
print("Min: ", weight_min, "kg")
print("Max: ", weight_max, "kg")
print("Mean: ", weight_mean, "kg")
print("Median: ", weight_median, "kg")
print("Standard deviation: ", weight_stdev, "kg")
print()
print("Lifespan statistics:")
print("Min: ", lifespan_min, "years")
print("Max: ", lifespan_max, "years")
print("Mean: ", lifespan_mean, "years")
print("Median: ", lifespan_median, "years")
print("Standard deviation: ", lifespan_stdev, "years")
print()
print("Country and breed frequency table:")
for country, breeds in country_breed_freq.items():
    print(country.capitalize(), ":")
    for breed, freq in breeds.items():
        print("- ", breed.capitalize(), ": ", freq)

# 3
import requests

countries_api = 'https://restcountries.com/v3.1/all'

Read the data from the API
response = requests.get(countries_api)
data = response.json()


country_sizes = {}
language_counts = {}

for country in data:
    name = country.get('name').get('common')
    area = country.get('area')
    languages = country.get('languages')
    if name and area:
        country_sizes[name] = area
    
    if languages:
        for language in languages.keys():
            if language in language_counts:
                language_counts[language] += 1
            else:
                language_counts[language] = 1

# Calculate the statistics
largest_countries = sorted(country_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
most_spoken_languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:10]
total_languages = len(language_counts)


print("The 10 largest countries:")
for country, size in largest_countries:
    print("-", country, ":", size, "sq km")
print()

print("The 10 most spoken languages:")
for language, count in most_spoken_languages:
    print("-", language.capitalize(), ":", count, "countries")
print()

print("The total number of languages in the countries API: ", total_languages)

# 4
import requests
from bs4 import BeautifulSoup

uci_url = 'https://archive.ics.uci.edu/ml/datasets.php'

#  Fetch the HTML content of the UCI website
response = requests.get(uci_url)
html_content = response.content

#  Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# SFind all the tables that contain the datasets
tables = soup.find_all('table', {'cellspacing': '1', 'cellpadding': '5'})

# Extract the dataset information from each table
for table in tables:
    rows = table.find_all('tr')
    for row in rows[1:]:
        columns = row.find_all('td')
        dataset_name = columns[0].get_text().strip()
        dataset_url = columns[0].find('a')['href']
        dataset_task = columns[1].get_text().strip()
        dataset_attributes = columns[2].get_text().strip()
        dataset_instances = columns[3].get_text().strip()
        dataset_features = columns[4].get_text().strip()
        dataset_year = columns[5].get_text().strip()
        print(f"{dataset_name} ({dataset_url}) - {dataset_task} - Attributes: {dataset_attributes}, Instances: {dataset_instances}, Features: {dataset_features}, Year: {dataset_year}")

