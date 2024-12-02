

import pandas as pd
import dask.dataframe as dd
import os

#reads a large SAS file from read_path 
#creates a folder at folder_path
#converts the SAS file to parquet files and saves the parquet files to folder_path

def check_folder_path(folder_path):
    if not os.path.exists(folder_path):
        # Create the folder if it doesn't exist
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def sas_to_parquet(read_path, folder_path, chunksize=100000):
    check_folder_path(folder_path)
    df = pd.read_sas(read_path, chunksize=chunksize, encoding='utf-8', iterator=True)
    i=0
    for chunk in df:
        chunk.to_parquet(folder_path + "/" +str(i)+".parquet")
        i=i+1

def parquet_to_df(folder_path):
    dask_df = dd.read_parquet(folder_path + "/*.parquet")
    return dask_df

#return dataframe with just the columns you care about
# column_names is a list of strings
def filtered_df(df, column_names):
    filtered_df = df[column_names]
    return filtered_df


# Adds masked state identifiers to dask dataframe and save to folder of parquet files
# returns the combined data frame
def add_state_id(dask_df, state_id_path, folder_path):
    check_folder_path(folder_path)
    key_to_state = pd.read_csv(state_id_path)

    #make sure that the PcrKey values in both dataframes have the same type
    dask_df['PcrKey'] = dask_df['PcrKey'].astype('int')
    key_to_state['PcrKey'] = key_to_state['PcrKey'].astype('int')
    combined_df = dask_df.merge(key_to_state, on='PcrKey', how='inner')
    combined_df.to_parquet(folder_path, write_index=False)
    return combined_df

# separates the dataframe into a separate file for each state
# Save the file for each state to folder_path 
def separate_to_states(dask_df, folder_path):
    check_folder_path(folder_path)
    if 'Masked_DestinationState' in dask_df.columns:

        unique_states = dask_df['Masked_DestinationState'].drop_duplicates().compute()
    elif 'masked_DestinationStateID' in dask_df.columns:
        unique_states = dask_df['masked_DestinationStateID'].drop_duplicates().compute()

    for state in unique_states:
        # Filter rows for the current state
        if 'Masked_DestinationState' in dask_df.columns:
            state_df = dask_df[dask_df['Masked_DestinationState'] == state]
        elif 'masked_DestinationStateID' in dask_df.columns:
            state_df = dask_df[dask_df['masked_DestinationStateID'] == state]
        # Compute the filtered state DataFrame
        state_df = state_df.compute()
        
        # Save to a Parquet file
        state_df.to_parquet(f'{folder_path}/state_{state}.parquet', index=False)

def unique_states(file_path):
    df = pd.read_csv(file_path)
    statelist = df['Masked_DestinationState'].drop_duplicates()
    return statelist


def load_states(folder_path, statelist):
    statedict = {}
    for state in statelist:
        try:
            if not os.path.exists('D:/Graduate Center Dropbox/Yuan Liu/Data Science Project/parquetfiles/'+str(i)+'states/state_'+state+'.parquet'):
                raise FileNotFoundError(f"{state} not found")
            df = pd.read_parquet(folderpath + '/state_'+state+'.parquet')
            statedict[state]=df
        except FileNotFoundError as e:
            print(e)
            continue
    return statedict
