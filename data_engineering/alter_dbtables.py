from pathlib import Path
import os
import sqlite3

#for year in (2018, 2019):
for year in [2018]:
    dir = "/Users/jonathanmiller/Documents/GitHub/erdosEMS/data/SAS/"
    dir_path = dir + "SAS.db/SAS " + str(year) + ".db"

    print(dir_path)
    directory_path = Path(dir_path)

    # Loop through each item in the directory
    for file_path in directory_path.iterdir():
        # Check if it's a file
        if file_path.is_file():
            #print("Processing file:", file_path)
            filnam= os.path.basename(file_path)
            table = filnam.strip(".db");

            # Define the database paths
            db_file = dir + "SAS.db/SAS " + str(year) + ".db/" + table + ".db"
            db_path = dir + "SAS " + str(year) + ".db/" + table + ".db"
#            db_path = dir + "SAS.db/SAS " + str(year) + ".db/" + table + ".db"
            print(table)
            print(db_path)
            # Rename table
            with sqlite3.connect(db_file) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(f'ALTER TABLE "{db_path}" RENAME TO "{table}";')
                except:
                    print("error:", table)

