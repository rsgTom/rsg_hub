from bs4 import BeautifulSoup
import requests
import lxml

url = requests.get('https://connect.biorxiv.org/relate/feed/181')
brief = requests.get('https://resolutestrategy.com/the-brief?format=rss')

soup_a = BeautifulSoup(url.content, 'xml')
soup_b = BeautifulSoup(brief.content,'xml')
soup = soup_a + soup_b

items = soup.find_all('item')


for item in items:
    title = item.title.text
    summary = item.description.text
    category = item.category.text
    link = item['rdf:about']
    print(f"Title: {title}\n\nCategory {category}\n\nSummary: {summary}\n\nLink: {link}\n\n------------------------")