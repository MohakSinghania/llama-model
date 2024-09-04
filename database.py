
import constants
from dependency import *

class PDFDataDatabase:
    def __init__(self):
        self.dbname = constants.DBNAME
        self.user = constants.DBUSER
        self.password = constants.DBPW
        self.host = constants.DBHOST
        self.port = constants.DBPORT
        self.connection = None

    def connect(self):
        try:
            self.connection = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            print("Connected to the database successfully")
        except Exception as e:
            print(f"Error connecting to the database: {e}")

    def create_table(self):
        if self.connection is None:
            print("No database connection. Call connect() first.")
            return

        create_table_query = '''
            CREATE TABLE IF NOT EXISTS pdf_data (
                upload_by INTEGER NOT NULL,
                pdf_file_name TEXT NOT NULL UNIQUE,
                pdf_path TEXT NOT NULL,
                upload_date_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                class TEXT NOT NULL,
                PRIMARY KEY (pdf_file_name)
            );
        '''

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(create_table_query)
                self.connection.commit()
                print("Table 'pdf_data' created successfully (or already exists).")
        except Exception as e:
            print(f"Error creating table: {e}")
            self.connection.rollback()

    def insert_or_update_data(self, upload_by, pdf_file_name, pdf_path, class_name, upload_date_time=None):
        if self.connection is None:
            print("No database connection. Call connect() first.")
            return
        
        if upload_date_time is None:
            upload_date_time = datetime.now()

        upsert_query = '''
            INSERT INTO pdf_data (upload_by, pdf_file_name, pdf_path, upload_date_time, class)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (pdf_file_name)
            DO UPDATE SET
                upload_by = EXCLUDED.upload_by,
                upload_date_time = EXCLUDED.upload_date_time;
        '''

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(upsert_query, (upload_by, pdf_file_name, pdf_path, upload_date_time, class_name))
                self.connection.commit()
                print("Data inserted or updated successfully.")
        except Exception as e:
            print(f"Error inserting or updating data: {e}")
            self.connection.rollback()


    def get_files_by_class(self, class_name):
        if self.connection is None:
            print("No database connection. Call connect() first.")
            return []

        try:
            with self.connection.cursor() as cursor:
                # Retrieve file names, upload_by, and upload_date_time for the specified class
                cursor.execute('''
                    SELECT pdf_file_name, pdf_path, upload_by, upload_date_time
                    FROM pdf_data
                    WHERE class = %s;
                ''', (class_name,))
                
                results = cursor.fetchall()
                # Convert the result to a list of dictionaries
                files = [
                    {
                        "pdf_file_name": row[0],
                        "pdf_path": row[1],
                        "upload_by": row[2],
                        "upload_date_time": row[3]
                    }
                    for row in results
                ]
                return files

        except Exception as e:
            print(f"Error retrieving data: {e}")
            return []

    def get_pdf_path(self, pdf_file_name, upload_by, class_name):
        if self.connection is None:
            print("No database connection. Call connect() first.")
            return None

        try:
            with self.connection.cursor() as cursor:
                # Retrieve the pdf_path for the given pdf_file_name, upload_by, and class
                cursor.execute('''
                    SELECT pdf_path
                    FROM pdf_data
                    WHERE pdf_file_name = %s AND upload_by = %s AND class = %s;
                ''', (pdf_file_name, upload_by, class_name))
                
                result = cursor.fetchone()
                
                if result:
                    return result[0]  # Return the pdf_path
                else:
                    print("No record found.")
                    return None

        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None

    def delete_record(self, pdf_file_name, pdf_file_path, class_name):
        if self.connection is None:
            print("No database connection. Call connect() first.")
            return

        try:
            with self.connection.cursor() as cursor:
                # Delete the record matching the given upload_by, pdf_file_name, and class
                cursor.execute('''
                    DELETE FROM pdf_data
                    WHERE pdf_file_name = %s AND pdf_path = %s AND class = %s;
                ''', (pdf_file_name, pdf_file_path, class_name))
                
                self.connection.commit()
                print(f"Record with pdf_file_name = '{pdf_file_name}', and class = '{class_name}' deleted successfully.")
                
        except Exception as e:
            print(f"Error deleting record: {e}")
            self.connection.rollback()

    def close_connection(self):
        if self.connection:
            self.connection.close()
            print("Database connection closed.")

# # Example usage
# if __name__ == "__main__":   
#     db = PDFDataDatabase()
#     db.connect()
#     db.create_table()
#     db.close_connection()
