import psycopg

conn = psycopg.connect(
    host="localhost",
    port=5432,
    dbname="dengue_db",
    user="dengue",
    password="dengue",
)
print("Conectou!")
conn.close()