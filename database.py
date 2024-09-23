import pytz
import uuid
import psycopg2
import constants
from datetime import datetime
from langchain_community.vectorstores import PGVector
from psycopg2.extensions import register_adapter, AsIs


# Register adapter for UUID
def adapt_uuid(uuid_val):
    return AsIs(f"'{str(uuid_val)}'")


register_adapter(uuid.UUID, adapt_uuid)


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
                pdf_id UUID NOT NULL PRIMARY KEY,
                upload_by INTEGER NOT NULL,
                pdf_file_name TEXT NOT NULL UNIQUE,
                pdf_path TEXT NOT NULL,
                upload_date_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                class TEXT
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

    def create_table_selection(self):
        if self.connection is None:
            print("No database connection. Call connect() first.")
            return

        create_table_query = '''
                CREATE TABLE IF NOT EXISTS pdf_data_selection_type (
                    pdf_id UUID NOT NULL PRIMARY KEY,
                    upload_by INTEGER NOT NULL,
                    pdf_file_name TEXT NOT NULL UNIQUE,
                    pdf_path TEXT NOT NULL,
                    upload_date_time TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Kolkata'),
                    school_college_ce TEXT NOT NULL,
                    board_type TEXT,
                    state_board TEXT,
                    class_name TEXT,
                    stream_name TEXT,
                    college_name TEXT,
                    course_name TEXT,
                    subject_name TEXT,
                    competitve_exam_name TEXT
                );
        '''

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(create_table_query)
                self.connection.commit()
                print("Table 'pdf_data_selection_type' created successfully (or already exists).")
        except Exception as e:
            print(f"Error creating table: {e}")
            self.connection.rollback()

    def insert_or_update_data_selection(self, data, upload_date_time=None):
        if self.connection is None:
            print("No database connection. Call connect() first.")
            return

        if upload_date_time is None:
            upload_date_time = datetime.now()

        # Convert upload_date_time to IST
        ist_timezone = pytz.timezone('Asia/Kolkata')
        upload_date_time = upload_date_time.astimezone(ist_timezone)

        upsert_query = '''
            INSERT INTO pdf_data_selection_type (pdf_id, upload_by, pdf_file_name, pdf_path, upload_date_time, school_college_ce, board_type,
                                                state_board, class_name, stream_name, college_name, course_name, subject_name, competitve_exam_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (pdf_file_name)
            DO UPDATE SET
                pdf_id = EXCLUDED.pdf_id,
                upload_by = EXCLUDED.upload_by,
                pdf_path = EXCLUDED.pdf_path,
                upload_date_time = EXCLUDED.upload_date_time,
                school_college_ce = EXCLUDED.school_college_ce,
                board_type = EXCLUDED.board_type,
                state_board = EXCLUDED.state_board,
                class_name = EXCLUDED.class_name,
                stream_name = EXCLUDED.stream_name,
                college_name = EXCLUDED.college_name,
                course_name = EXCLUDED.stream_name,
                subject_name = EXCLUDED.subject_name,
                competitve_exam_name = EXCLUDED.competitve_exam_name;
        '''

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(upsert_query, (
                    data['pdf_id'], data['upload_by'], data['pdf_file_name'], data['pdf_path'], upload_date_time,
                    data['school_college_ce'], data['board_type'], data['state_board'], data['class_name'],
                    data['stream_name'], data['college_name'], data['course_name'], data['subject_name'], data['competitve_exam_name']
                ))
                self.connection.commit()
                print("Data inserted or updated successfully.")
        except Exception as e:
            print(f"Error inserting or updating data: {e}")
            self.connection.rollback()

    def insert_or_update_data_class(self, pdf_id, upload_by, pdf_file_name, pdf_path, class_name=None, upload_date_time=None):
        if self.connection is None:
            print("No database connection. Call connect() first.")
            return

        if upload_date_time is None:
            upload_date_time = datetime.now()

        upsert_query = '''
            INSERT INTO pdf_data (pdf_id, upload_by, pdf_file_name, pdf_path, upload_date_time, class)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (pdf_file_name)
            DO UPDATE SET
                pdf_id = EXCLUDED.pdf_id,
                upload_by = EXCLUDED.upload_by,
                pdf_path = EXCLUDED.pdf_path,
                upload_date_time = EXCLUDED.upload_date_time,
                class = EXCLUDED.class;
        '''

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(upsert_query, (pdf_id, upload_by, pdf_file_name, pdf_path, upload_date_time, class_name))
                self.connection.commit()
                print("Data inserted or updated successfully.")
        except Exception as e:
            print(f"Error inserting or updating data: {e}")
            self.connection.rollback()

    def get_files_by_class(self, class_name=None):
        if self.connection is None:
            print("No database connection. Call connect() first.")
            return []

        try:
            if class_name is not None:
                with self.connection.cursor() as cursor:
                    # Retrieve file names, upload_by, and upload_date_time for the specified class
                    cursor.execute('''
                        SELECT pdf_id, pdf_file_name, pdf_path, upload_by, upload_date_time
                        FROM pdf_data
                        WHERE class = %s;
                    ''', (class_name,))

                    results = cursor.fetchall()
                    # Convert the result to a list of dictionaries
                    files = [
                        {
                            "pdf_id": row[0],
                            "pdf_file_name": row[1],
                            "pdf_path": row[2],
                            "upload_by": row[3],
                            "upload_date_time": row[4]
                        }
                        for row in results
                    ]
                    return files
            else:
                with self.connection.cursor() as cursor:
                    # Retrieve file names, upload_by, and upload_date_time for the specified class
                    cursor.execute('''
                        SELECT pdf_id, pdf_file_name, pdf_path, upload_by, upload_date_time
                        FROM pdf_data
                        WHERE class IS NULL;
                    ''')

                    results = cursor.fetchall()
                    # Convert the result to a list of dictionaries
                    files = [
                        {
                            "pdf_id": row[0],
                            "pdf_file_name": row[1],
                            "pdf_path": row[2],
                            "upload_by": row[3],
                            "upload_date_time": row[4]
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

    def delete_record(self, pdf_id, pdf_file_name, class_name=None):
        if self.connection is None:
            print("No database connection. Call connect() first.")
            return {"pdf_path": None}
        if class_name is not None:
            try:
                with self.connection.cursor() as cursor:
                    # Delete the record matching the given upload_by, pdf_file_name, and class
                    cursor.execute('''
                        SELECT pdf_path FROM pdf_data
                        WHERE pdf_id = %s
                    ''', (pdf_id,))
                    record = cursor.fetchone()
                    pdf_path = record[0]
                    cursor.execute('''
                        DELETE FROM pdf_data
                        WHERE pdf_id = %s AND pdf_file_name = %s AND class = %s;
                    ''', (pdf_id, pdf_file_name, class_name))

                    self.connection.commit()
                    return {"pdf_path": pdf_path}

            except Exception as e:
                print(f"Error deleting record: {e}")
                self.connection.rollback()
                return {"pdf_path": None}
        else:
            try:
                with self.connection.cursor() as cursor:
                    # Delete the record matching the given upload_by, pdf_file_name, and class
                    cursor.execute('''
                        SELECT pdf_path FROM pdf_data
                        WHERE pdf_id = %s
                    ''', (pdf_id,))
                    record = cursor.fetchone()
                    pdf_path = record[0]
                    cursor.execute('''
                        DELETE FROM pdf_data
                        WHERE pdf_id = %s AND pdf_file_name = %s AND class IS NULL;
                    ''', (pdf_id, pdf_file_name))

                    self.connection.commit()
                    return {"pdf_path": pdf_path}

            except Exception as e:
                print(f"Error deleting record: {e}")
                self.connection.rollback()
                return {"pdf_path": None}

    def close_connection(self):
        if self.connection:
            self.connection.close()
            print("Database connection closed.")

    def delete_all(self):
        is_rec_exist = True
        try:
            with psycopg2.connect(host=constants.DBHOST, database=constants.DBNAME, user=constants.DBUSER, password=constants.DBPW) as db:
                cursor = db.cursor()
                cursor.execute("DELETE FROM pdf_data")
                cursor.execute("DELETE FROM langchain_pg_collection")
                is_rec_exist = False
        except Exception:
            pass
        return {'is_rec_exist': is_rec_exist}


class VectorStorePostgresVector:
    def __init__(self, collection_name, embeddings):
        self.collection_name = collection_name
        self.connection = constants.CONNECTION_SETTINGS
        self.embeddings = embeddings

    def get_or_create_collection(self):
        import pdb;pdb.set_trace()
        return PGVector(
            collection_name=self.collection_name,
            connection_string=self.connection,
            embedding_function=self.embeddings
        )

    def store_docs_to_collection(self, document_id, docs, document_path):
        for doc in docs:
            inner_metadata = {
                'id': document_id, 'source': document_path
            }
            for k, v in doc.metadata.items():
                if k in ['id', 'page', 'source', 'page_number', 'row']:
                    if v is None:
                        inner_metadata[k] = ""
                    else:
                        if k == "page_number":
                            k = 'page'
                            v = int(v) - 1
                        if k == "source":
                            if v != document_path:
                                v = document_path
                        inner_metadata[k] = str(v)
            doc.metadata = inner_metadata
        vector_db = self.get_or_create_collection()

        texts, metadatas, ids = [], [], []
        for doc in docs:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
            ids.append(f"{document_id}")
        vector_db.add_texts(texts, metadatas, ids=ids)
        return True

    def delete_documents_from_collection(self, document_id):
        vector_db = self.get_or_create_collection()
        vector_db.delete([f"{document_id}"])

    def check_if_record_exist(self, document_id):
        is_rec_exist = False
        try:
            with psycopg2.connect(host=constants.DBHOST, database=constants.DBNAME, user=constants.DBUSER, password=constants.DBPW) as db:
                cursor = db.cursor()
                cursor.execute("SELECT to_regclass('langchain_pg_embedding');")
                record = cursor.fetchone()
                if record:
                    record = record[0]
                    if record == 'langchain_pg_embedding':
                        cursor.execute("SELECT EXISTS (SELECT 1 FROM langchain_pg_embedding WHERE custom_id = %s LIMIT 1)", (document_id,))
                        record = cursor.fetchone()
                        record = record[0]
                        is_rec_exist = record
        except Exception:
            pass
        return {'is_rec_exist': is_rec_exist}

    def delete_file_embeddings_from_collection(self, pdf_id):
        is_rec_exist = True
        try:
            with psycopg2.connect(host=constants.DBHOST, database=constants.DBNAME, user=constants.DBUSER, password=constants.DBPW) as db:
                cursor = db.cursor()
                cursor.execute("DELETE FROM langchain_pg_embedding WHERE custom_id = %s", (pdf_id,))
                is_rec_exist = False
        except Exception:
            pass
        return {'is_rec_exist': is_rec_exist}


# Example usage
if __name__ == "__main__":
    db = PDFDataDatabase()
    db.connect()
    db.create_table()
    db.create_table_selection()
    db.close_connection()
