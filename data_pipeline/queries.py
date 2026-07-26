import sqlite3
import pandas as pd
import os

os.makedirs("outputs", exist_ok=True)

conn = sqlite3.connect("books.db")

queries = {
    "query1": """
        SELECT title, price_gbp, price_inr, rating
        FROM books
        WHERE rating = 5;
    """,

    "query2": """
        SELECT title, price_gbp, price_inr
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,

    "query3": """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating;
    """,

    "query4": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 30
        ORDER BY price_gbp;
    """,

    "query5": """
        SELECT title, rating, price_gbp
        FROM books
        WHERE rating IN (4,5)
        ORDER BY rating DESC, price_gbp DESC;
    """,

    "join_query": """
        SELECT
            b.title,
            c.category_name,
            b.rating,
            b.price_gbp,
            b.price_inr,
            b.in_stock
        FROM books b
        INNER JOIN categories c
        ON b.category_id = c.category_id
        ORDER BY c.category_name, b.rating DESC;
    """
}

for name, query in queries.items():
    print(f"\n{'='*60}")
    print(name.upper())
    print('='*60)

    df = pd.read_sql(query, conn)

    print(df)

    df.to_csv(f"outputs/{name}.csv", index=False)

conn.close()

print("\nAll query outputs saved successfully!")



conn = sqlite3.connect("books.db")

# SQL JOIN
join_query = """
SELECT
    b.title,
    c.category_name,
    b.rating,
    b.price_gbp,
    b.price_inr,
    b.in_stock
FROM books b
JOIN categories c
ON b.category_id = c.category_id
ORDER BY c.category_name;
"""

sql_join_df = pd.read_sql(join_query, conn)

# Load tables separately
books_df = pd.read_sql("SELECT * FROM books", conn)
categories_df = pd.read_sql("SELECT * FROM categories", conn)

# Pandas merge
merged_df = books_df.merge(
    categories_df,
    on="category_id",
    how="inner"
)

merged_df = merged_df[
    [
        "title",
        "category_name",
        "rating",
        "price_gbp",
        "price_inr",
        "in_stock"
    ]
].sort_values(by="category_name").reset_index(drop=True)

sql_join_df = sql_join_df.reset_index(drop=True)

print("SQL JOIN")
print(sql_join_df)

print("\nPandas Merge")
print(merged_df)



conn.close()