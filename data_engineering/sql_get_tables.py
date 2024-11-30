import sqlite3

# Path to the databases
state_db_path = "/Users/jonathanmiller/Documents/GitHub/erdosEMS/data/Destinations.db/destination_State_d_2022.db"
county_db_path = "/Users/jonathanmiller/Documents/GitHub/erdosEMS/data/Destinations.db/destination_County_d_2022.db"

# Function to list tables in a database
def list_tables(db_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables in {db_path}: {tables}")

# List tables in each database
list_tables(state_db_path)
list_tables(county_db_path)
