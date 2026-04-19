import sqlite3
import pandas as pd

def extract_top_pysr_models(db_path, output_csv):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    
    # Query to select the top 10 models sorted by highest accuracy
    # We are pulling the trial ID, accuracy, complexity, loss, and the LaTeX equation
    query = """
    SELECT 
        trial_id, 
        accuracy, 
        variables_json,
        complexity, 
        loss, 
        latex_format 
    FROM 
        results 
    ORDER BY 
        accuracy DESC 
    LIMIT 10;
    """
    
    try:
        # Read the SQL query directly into a pandas DataFrame
        df = pd.read_sql_query(query, conn)
        
        # Export the DataFrame to a CSV file
        df.to_csv(output_csv, index=False)
        print(f"Success! Top 10 models successfully exported to {output_csv}")
        
        # Display the top 3 in the console for a quick preview
        print("\nTop 3 Models Preview:")
        print(df.head(3).to_string())
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        # Always close the database connection
        conn.close()

# Define your file paths here
database_file = 'Results.db'
output_file = 'Top_10_PySR_Models.csv'

if __name__ == "__main__":
    extract_top_pysr_models(database_file, output_file)