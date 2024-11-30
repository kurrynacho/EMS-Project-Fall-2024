import sqlite3

# Path to your SQLite database file
db_path = "Destinations.db/destination_County_d_2022.db"
table_name = "destination_county"  # Replace with the actual table name

#db_path = "SAS/SAS.db/SAS 2020.db/pub_pcrevents.db"
#table_name = "pub_pcrevents"

# Connect to the database
with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()

    # Use PRAGMA to get column information
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()

    # Extract and print column names
    column_names = [column[1] for column in columns]
    print("Column names:", column_names)
