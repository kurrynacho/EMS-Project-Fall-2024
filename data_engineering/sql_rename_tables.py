import sqlite3

#for year in (2023, 2022, 2021, 2020, 2019, 2018):
for year in (2018):
# Define the database paths
    state_db_path = "/Users/jonathanmiller/Documents/GitHub/erdosEMS/data/Destinations.db/destination_State_d_" + str(year) + ".db"
    county_db_path = "/Users/jonathanmiller/Documents/GitHub/erdosEMS/data/Destinations.db/destination_County_d_" + str(year) + ".db"

# Rename table in state_db
#   with sqlite3.connect(state_db_path) as conn:
#       try:
#           cursor = conn.cursor()
#           cursor.execute(f'ALTER TABLE "{state_db_path}" RENAME TO destination_state;')
#       except:
#           print(state_db_path);

# Rename table in county_db
    with sqlite3.connect(county_db_path) as conn:
        try:
            cursor = conn.cursor()
            cursor.execute(f'ALTER TABLE "{county_db_path}" RENAME TO destination_county;')
        except:
            print(county_db_path);
