# 1
countries = [
    {
        "name": "Afghanistan",
        "capital": "Kabul",
        "languages": [
            "Pashto",
            "Uzbek",
            "Turkmen"
        ],
        "population": 27657145,
        "flag": "https://restcountries.eu/data/afg.svg",
        "currency": "Afghan afghani"
    },
    {
        "name": "Åland Islands",
        "capital": "Mariehamn",
        "languages": [
            "Swedish"
        ],
        "population": 28875,
        "flag": "https://restcountries.eu/data/ala.svg",
        "currency": "Euro"
    }
]

# Sort countries by name
sorted_countries_by_name = sorted(countries, key=lambda x: x['name'])

# Sort countries by capital
sorted_countries_by_capital = sorted(countries, key=lambda x: x['capital'])

# Sort countries by population
sorted_countries_by_population = sorted(countries, key=lambda x: x['population'], reverse=True)

# Sort out the ten most spoken languages by location
languages = []
for country in countries:
    for language in country['languages']:
        languages.append(language)

from collections import Counter
sorted_languages = dict(Counter(languages))
sorted_languages = sorted(sorted_languages.items(), key=lambda x: x[1], reverse=True)

top_ten_languages = sorted_languages[:10]




# Sort out the ten most populated countries
sorted_countries_by_population = sorted(countries, key=lambda x: x['population'], reverse=True)

top_ten_populated_countries = sorted_countries_by_population[:10]




