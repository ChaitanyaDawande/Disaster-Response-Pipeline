import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """
    Load data from a database
    Input: messages_filepath - String
            The path to the messagages data csv
           categories_filepath - String
            The path to the categories data csv
    Output: df = DataFrame
            Merged DataFrame of message and categories data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge the two datasets together
    df = pd.merge(messages, categories)
    
    return df


def clean_data(df):
    """
    Extracts categories and flags from categories data, remove duplicates
    Input: df - DataFrame
            Dataframe output from load_data function
    Output: df - DataFrame
            Cleansed dataframe of the input data
            
    """        
    # Split the categories
    categories = df['categories'].str.split(pat=';',expand=True)
    
    #Fix the categories columns name
    row = categories.iloc[0]
    category_colnames=row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()
    
    return df
    
    


def save_data(df, database_filename, table_name="disaster_response"):
    
    
    """
    Save data to database
    Input: df - DataFrame
            DataFrame from clean_data dataframe
           database_filename - String
           Database file location of where data is to be stored
           table_name - String (default is disaster_response)
           The name of the table that the data should be saved to 
    """    
    
    engine = create_engine('sqlite:///{}'.format(database_filename)) 
    df.to_sql(table_name, engine, index=False, if_exists='replace')
  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()