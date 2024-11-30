# sas_to_sqlite.py: python wrapper for sqlite3 conversion from sas7bdat to sql .db format.
# ===========================================================================================
# We find python is necessary to import sas7bdat format!
# ===========================================================================================
# sas7bdat files provided directly by nemsis on request solely to Jessica Yuan Liu for this project.
# These files are stored on Ms. Liu's local directory, and available via dropbox.
# JM will figure out longer-term storage past Dec 2024.
# ===========================================================================================

import pyreadstat
import pandas as pd
import sqlite3
import os
import shutil
import sys


# courtesy of chatGPT
def create_or_clean_directory(path):
    """Creates a directory if it doesn't exist, otherwise cleans it up."""

    if not os.path.exists(path):
        os.makedirs(path)
    else:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

# Example usage
directory_path = 'my_directory'
create_or_clean_directory(directory_path)

# Directory structure
# $project_dir/data
#  - SAS
#    - "SAS ${year}"/SASfileName/.sas7bdat
#      -  SAS.db/
#        - "SAS ${year}.db"
#          - SASfileName.db
#   

# Set your project directory
project_dir = "/Users/jonathanmiller/Documents/GitHub/erdosEMS/"
data_dir = project_dir + "data/"
SAS_dir = data_dir + "SAS/"
SASdb_dir = SAS_dir + "SAS.db/"

year="2018"
in_directory = SAS_dir + "SAS " + year + "/"
out_directory = SASdb_dir + "SAS " + year + ".db/"
create_or_clean_directory(out_directory)
print(in_directory, out_directory)

# computedelements.sas7bdat

# List all files in the directory
file_names = os.listdir(in_directory)
sas7dat_files = [f for f in os.listdir(in_directory) if f.endswith('.sas7bdat')]

#print(*sas7dat_files,"\n")
for sas_file in sas7dat_files:
    file_path = os.path.join(in_directory, sas_file)
    print(file_path)
    with open(file_path, 'r') as file:
        file_prefix = sas_file.replace('.sas7bdat', ' ')
        #print(sas_file)
        print(file_prefix)

#        keys = ["PcrKey","eTimes_03"]
#        keys = None

        file_prefix_stripped = file_prefix.replace(" ","")
        dbfile = out_directory + file_prefix_stripped + ".db"
        print(dbfile)

        conn = sqlite3.connect(dbfile)
        cursor = conn.cursor()

        # Define the chunk size
        chunk_size = 100000  # You can adjust this depending on memory and performance needs
        #Read large SAS files in chunks
        # Initialize chunk iterator
        chunk_iter = pd.read_sas(file_path, chunksize=chunk_size, format='sas7bdat')

        # Loop over each chunk
        for i, chunk in enumerate(chunk_iter):
            # Write each chunk to the SQLite database
            chunk.to_sql(dbfile, conn, if_exists='append', index=False)
            #print(f"Processed chunk {i+1}")

        # Commit changes and close connection
        conn.commit()
        conn.close()
