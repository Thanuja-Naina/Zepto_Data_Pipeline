import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"

GBP_TO_INR = 105.50

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

books = []

###########################################################
# SCRAPE FIRST 5 PAGES (100 BOOKS)
###########################################################

for page in range(1, 6):

    url = BASE_URL.format(page)

    response = requests.get(url)

    soup = BeautifulSoup(response.text, "lxml")

    articles = soup.find_all("article", class_="product_pod")

    for article in articles:

        title = article.h3.a["title"]

        price = article.find("p", class_="price_color").text.strip()

        rating = article.find("p")["class"][1]

        availability = article.find(
            "p",
            class_="instock availability"
        ).text.strip()

        ###################################################
        # Get Book Details Page
        ###################################################

        link = article.h3.a["href"]

        link = link.replace("../../../", "")

        detail_url = "https://books.toscrape.com/catalogue/" + link

        detail = requests.get(detail_url)

        detail_soup = BeautifulSoup(detail.text, "lxml")

        breadcrumb = detail_soup.find_all("ul", class_="breadcrumb")[0]

        category = breadcrumb.find_all("li")[2].text.strip()

        books.append({
            "title": title,
            "price": price,
            "rating_text": rating,
            "availability": availability,
            "category": category
        })

###########################################################
# CREATE DATAFRAME
###########################################################

df = pd.DataFrame(books)

print(df.head())

###########################################################
# CLEAN PRICE
###########################################################

df["price_gbp"] = (
    df["price"]
    .str.replace("£", "", regex=False)
)

df["price_gbp"] = pd.to_numeric(
    df["price_gbp"],
    errors="coerce"
)

###########################################################
# CLEAN RATING
###########################################################

df["rating"] = df["rating_text"].map(rating_map)

###########################################################
# CLEAN STOCK
###########################################################

df["in_stock"] = (
    df["availability"]
    .str.contains("In stock")
)

###########################################################
# HANDLE MISSING VALUES
###########################################################

median_price = df["price_gbp"].median()

df["price_gbp"] = df["price_gbp"].fillna(median_price)

median_rating = df["rating"].median()

df["rating"] = df["rating"].fillna(median_rating)

###########################################################
# INR CONVERSION
###########################################################

df["price_inr"] = df["price_gbp"] * GBP_TO_INR

###########################################################
# DROP UNUSED COLUMNS
###########################################################

df = df.drop(
    columns=[
        "price",
        "rating_text",
        "availability"
    ]
)

###########################################################
# CREATE DATABASE
###########################################################

conn = sqlite3.connect("books.db")

cursor = conn.cursor()

###########################################################
# CATEGORY TABLE
###########################################################

cursor.execute("""
CREATE TABLE IF NOT EXISTS categories(

category_id INTEGER PRIMARY KEY AUTOINCREMENT,

category_name TEXT UNIQUE

)
""")

###########################################################
# BOOK TABLE
###########################################################

cursor.execute("""

CREATE TABLE IF NOT EXISTS books(

book_id INTEGER PRIMARY KEY AUTOINCREMENT,

title TEXT,

price_gbp REAL,

price_inr REAL,

rating INTEGER,

in_stock INTEGER,

category_id INTEGER,

FOREIGN KEY(category_id)

REFERENCES categories(category_id)

)

""")

###########################################################
# INSERT CATEGORIES
###########################################################

categories = df["category"].unique()

for category in categories:

    cursor.execute("""

    INSERT OR IGNORE INTO categories(category_name)

    VALUES(?)

    """, (category,))

###########################################################
# CATEGORY LOOKUP
###########################################################

category_lookup = {}

cursor.execute("""

SELECT *

FROM categories

""")

for row in cursor.fetchall():

    category_lookup[row[1]] = row[0]

###########################################################
# INSERT BOOKS
###########################################################

for _, row in df.iterrows():

    cursor.execute("""

    INSERT INTO books(

    title,

    price_gbp,

    price_inr,

    rating,

    in_stock,

    category_id

    )

    VALUES(?,?,?,?,?,?)

    """, (

        row["title"],

        row["price_gbp"],

        row["price_inr"],

        int(row["rating"]),

        int(row["in_stock"]),

        category_lookup[row["category"]]

    ))

conn.commit()

conn.close()

print("Database Created Successfully!")