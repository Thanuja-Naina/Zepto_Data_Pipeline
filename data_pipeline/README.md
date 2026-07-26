# Data Pipeline Module – Book Catalog Scraping, Cleaning, and SQLite Database

## Overview

This project demonstrates a complete data engineering pipeline using the public practice website **https://books.toscrape.com**.

The pipeline performs the following tasks:

- Scrapes book information from the website.
- Cleans and transforms the scraped data.
- Converts book prices from GBP to INR using a fixed conversion rate.
- Stores the cleaned data in a normalized SQLite database.
- Executes SQL queries for analysis.
- Reads SQL query results into Pandas DataFrames.
- Reproduces the SQL JOIN using Pandas `merge()`.

---

# Technologies Used

- Python 3.x
- Requests
- BeautifulSoup4
- Pandas
- SQLite3

---

# Project Structure

```
data_pipeline/
│
├── scrape_books.py
├── queries.py
├── books.db
├── requirements.txt
├── README.md
└── outputs/
    ├── query1.csv
    ├── query2.csv
    ├── query3.csv
    ├── query4.csv
    ├── query5.csv
    └── join_query.csv
```

---

# Installation

Clone the repository.

```bash
git clone <repository-url>
```

Navigate into the project.

```bash
cd Zepto_Data_Pipeline/data_pipeline
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate the virtual environment.

Windows PowerShell

```powershell
.\.venv\Scripts\Activate
```

Install the required libraries.

```bash
pip install -r requirements.txt
```

---

# How to Run

Run the scraping and database creation script.

```bash
python scrape_books.py
```

This script will

- Scrape books from the first five catalogue pages.
- Clean and transform the data.
- Create the SQLite database.
- Insert all records into the database.

Next, execute the SQL queries.

```bash
python queries.py
```

This script will

- Execute all required SQL queries.
- Display the query results.
- Save the outputs as CSV files inside the `outputs` folder.
- Demonstrate `pd.read_sql()`.
- Compare SQL JOIN and Pandas `merge()` results.

---

# Data Cleaning Decisions

The following cleaning steps were applied:

### Price

- Removed the `£` currency symbol.
- Converted prices to floating-point numbers.
- Stored the cleaned values in the `price_gbp` column.

Example

```
£51.77

↓

51.77
```

---

### Rating

The website stores ratings as text.

```
One
Two
Three
Four
Five
```

These were converted into integers.

```
One   → 1
Two   → 2
Three → 3
Four  → 4
Five  → 5
```

---

### Availability

Availability text was converted into Boolean values.

```
In stock

↓

True
```

```
Out of stock

↓

False
```

The database stores these values as integers.

```
True  → 1
False → 0
```

---

### Missing Values

Numeric fields were converted using

```python
pd.to_numeric(errors="coerce")
```

If any numeric value failed to parse, the missing value was replaced using **median imputation**.

Essential text fields such as the book title or category were considered mandatory. Rows missing these values would be dropped to preserve data quality.

---

# Currency Conversion

This project uses the required fixed baseline conversion rate specified in the assignment.

```
1 GBP = 105.50 INR
```

This is a fixed project-defined constant.

No external currency-conversion API was used.

The INR price was calculated as

```
price_inr = price_gbp × 105.50
```

---

# Database Design

The project uses a normalized SQLite database with two related tables.

## Categories Table

| Column | Type |
|---------|------|
| category_id | INTEGER PRIMARY KEY |
| category_name | TEXT UNIQUE |

---

## Books Table

| Column | Type |
|---------|------|
| book_id | INTEGER PRIMARY KEY |
| title | TEXT |
| price_gbp | REAL |
| price_inr | REAL |
| rating | INTEGER |
| in_stock | INTEGER |
| category_id | INTEGER (Foreign Key) |

Relationship

```
categories.category_id
        │
        │
        ▼
books.category_id
```

---

# SQL Queries Implemented

The following SQL features were demonstrated.

- SELECT
- WHERE
- ORDER BY
- LIMIT
- DISTINCT
- BETWEEN
- IN
- INNER JOIN

---

# Pandas Operations

The project demonstrates

- `pd.read_sql()`
- `pd.merge()`

The JOIN performed using SQL was reproduced using

```python
pd.merge()
```

Both methods produce equivalent results.

---

# Output Files

All SQL query results are saved inside the `outputs` folder.

```
outputs/

query1.csv
query2.csv
query3.csv
query4.csv
query5.csv
join_query.csv
```

---

# Expected Output

Running

```bash
python scrape_books.py
```

creates

```
books.db
```

Running

```bash
python queries.py
```

prints all SQL query results and saves them as CSV files.

---

# Conclusion

This project demonstrates an end-to-end data engineering workflow including:

- Web scraping
- Data cleaning
- Data transformation
- Currency conversion
- Relational database design
- SQL querying
- Pandas integration
- SQL JOIN vs Pandas merge comparison

The implementation satisfies all the requirements specified in the assignment.