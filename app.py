from flask import Flask, render_template
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)

@app.route('/')
def index():
    url = requests.get('https://connect.biorxiv.org/relate/feed/181')
    soup = BeautifulSoup(url.content, 'xml')
    items = soup.find_all('item')

    feed_items = []
    for item in items:
        title = item.title.text if item.title else 'No title'
        date = item.find('dc:date').text if item.find('dc:date') else 'No date'
        category = item.category.text if item.category else 'Uncategorized'
        summary = item.description.text if item.description else 'No summary'
        link = item.link.text.strip() if item.link else '#'
        feed_items.append({'title': title, 'date': date, 'category': category, 'summary': summary, 'link': link})

    return render_template('index.html', feed_items=feed_items)

if __name__ == '__main__':
    app.run(debug=True)
