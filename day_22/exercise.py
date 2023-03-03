# 1
import requests
from bs4 import BeautifulSoup
import json

bu_url = 'http://www.bu.edu/president/boston-university-facts-stats/'
response = requests.get(bu_url)
soup = BeautifulSoup(response.content, 'html.parser')
table = soup.find('table')
if table is None:
    print('Table not found in HTML content:')
    print(response.content)
else:
    table_rows = table.find_all('tr')
    data = []
    for tr in table_rows:
        td = tr.find_all('td')
        row = [i.text.strip() for i in td]
        if row:
            data.append(row)
    with open('bu_data.json', 'w') as f:
        json.dump(data, f)

# 2
import requests
from bs4 import BeautifulSoup
import json

uci_url = 'https://archive.ics.uci.edu/ml/datasets.php'
response = requests.get(uci_url)
soup = BeautifulSoup(response.content, 'html.parser')

table = soup.find('table', {'cellspacing': '1'})

data = []
if table:
    rows = table.find_all('tr')
    headers = [header.text.strip() for header in rows[0].find_all('th')]

    for row in rows[1:]:
        row_data = {}
        for i, cell in enumerate(row.find_all('td')):
            row_data[headers[i]] = cell.text.strip()
        data.append(row_data)

with open('uci_data.json', 'w') as f:
    json.dump(data, f)

# 3

import requests
from bs4 import BeautifulSoup
import json

wiki_url = 'https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States'
response = requests.get(wiki_url)
soup = BeautifulSoup(response.content, 'html.parser')

table = soup.find('table', {'class': 'wikitable'})

data = []
if table:
    rows = table.find_all('tr')
    headers = [header.text.strip() for header in rows[0].find_all('th')]

    for row in rows[1:]:
        row_data = {}
        cells = row.find_all('td')
        if len(cells) == len(headers):
            for i, cell in enumerate(cells):
                row_data[headers[i]] = cell.text.strip()
            data.append(row_data)

with open('presidents_data.json', 'w') as f:
    json.dump(data, f)

