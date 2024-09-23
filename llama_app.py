import boto3
import torch
import ollama
import secrets
import constants
from flask_cors import CORS
from database import PDFDataDatabase
from llama_rag_model import llama_model
from flask import Flask, request, jsonify, session, Response
from botocore.exceptions import NoCredentialsError, ClientError

app = Flask(__name__)
CORS(app)
app.secret_key = secrets.token_hex(16)
in_memory_cache = {}
ALLOWED_BASE_DIR = '/home/ubuntu/llama-model/pdf_files'

rag_function = llama_model()


@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello There!</h1>"


@app.route('/ollama-chat', methods=['GET', 'POST'])
def ollama_chat():
    user_query = request.args.get("user_query")
    if not user_query:
        return jsonify({'message': 'Missing query or student ID', 'status': 400})
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "user",
                "content": user_query,
            },
        ],
    )
    answer = response["message"]["content"]
    torch.cuda.empty_cache()
    response = jsonify({'message': 'Query processed successfully', 'status': 200, 'question': user_query, 'answer': answer})
    return response


@app.route('/upload-pdf-class', methods=['GET', 'POST'])
def upload_pdf_class():
    pdf_file = request.files.getlist('pdf_file')
    teacher_id = request.form['teacher_id']
    class_name = request.form['class_name']
    invalid_files = []

    if pdf_file == []:
        return jsonify({'message': 'Please Provide PDF', 'status': 400})
    for file in pdf_file:
        if file.filename.endswith(".pdf"):
            pdf_details = rag_function._pdf_file_save_class(teacher_id, file, class_name)
            message = rag_function._create_embedding_all(pdf_details['pdf_id'], pdf_details['pdf_path'], class_name)
        else:
            invalid_files.append(file.filename)

    if invalid_files != []:
        invalid_message = f'{invalid_files} files are invalid'
        return jsonify({'message': f'PDF Uploaded Successfully and {invalid_message}', 'status': 201})
    else:
        return jsonify(message)


@app.route('/rag-model-class', methods=['GET', 'POST'])
def rag_model_class():
    user_query = request.args.get("user_query")
    student_id = request.args.get('student_id')
    class_name = request.args.get('class_name', "")
    if not user_query or not student_id:
        return jsonify({'message': 'Missing query or student ID', 'status': 400})

    if student_id not in session:
        data = {"class_name": class_name}
        session[student_id] = data

    elif class_name != "":
        data = session[student_id]
        data.update({"class_name": class_name})

    else:
        data = session[student_id]
        class_name = data["class_name"]
    try:
        answer = rag_function._get_answer_to_query(user_query, class_name)
        return jsonify(answer)

    except Exception as e:
        return jsonify({'message': f'Error generating answer: {str(e)}', 'status': 404})


@app.route('/upload-pdf', methods=['GET', 'POST'])
def upload_pdf():
    pdf_file = request.files.getlist('pdf_file')
    teacher_id = request.form['teacher_id']
    invalid_files = []

    if pdf_file == []:
        return jsonify({'message': 'Please Provide PDF', 'status': 400})
    for file in pdf_file:
        if file.filename.endswith(".pdf"):
            pdf_details = rag_function._pdf_file_save(teacher_id, file)
            message = rag_function._create_embedding_all(pdf_details['pdf_id'], pdf_details['pdf_path'])
        else:
            invalid_files.append(file.filename)

    if invalid_files != []:
        invalid_message = f'{invalid_files} files are invalid'
        return jsonify({'message': f'PDF Uploaded Successfully and {invalid_message}', 'status': 201})
    else:
        return jsonify(message)


@app.route('/rag-model', methods=['GET', 'POST'])
def rag_model():
    user_query = request.args.get("user_query")
    student_id = request.args.get('student_id')
    if not user_query or not student_id:
        return jsonify({'message': 'Missing query or student ID', 'status': 400})
    try:
        answer = rag_function._get_answer_to_query(user_query)
        return jsonify(answer)

    except Exception as e:
        return jsonify({'message': f'Error generating answer: {str(e)}', 'status': 404})


@app.route('/upload-pdf-hierarchy', methods=['GET', 'POST'])
def upload_pdf_hierarchy():
    pdf_file = request.files.getlist('pdf_file')
    teacher_id = request.form['teacher_id']
    school_college_competitiveexam = request.form['s_c_ce_type']
    board_type = request.form.get('board_type', None)
    state_board = request.form.get('state_board_type', None)
    stream_name = request.form.get('stream_name', None)
    class_name = request.form.get('class_name', None)
    college_name_type = request.form.get('college_name_type', None)
    course_type = request.form.get('course_type', None)
    subject_type = request.form.get('subject_type', None)

    invalid_files = []

    if pdf_file == []:
        return jsonify({'message': 'Please Provide PDF', 'status': 400})
    for file in pdf_file:
        if file.filename.endswith(".pdf"):
            if school_college_competitiveexam == "school":
                pdf_details = rag_function._pdf_file_save_selection(
                                                                    teacher_id, file, school_college_competitiveexam, board_type=board_type,
                                                                    state_board=state_board, class_name=class_name, stream_type=stream_name
                                                                )
                message = rag_function._create_embedding_selection(
                                                                    pdf_details['pdf_id'], pdf_details['pdf_path'], school_college_competitiveexam,
                                                                    board_type=board_type, state_board=state_board, class_name=class_name,
                                                                    stream_type=stream_name
                                                                )
            elif school_college_competitiveexam == "college":
                pdf_details = rag_function._pdf_file_save_selection(
                                                                    teacher_id, file, school_college_competitiveexam, college_name=college_name_type,
                                                                    course_name=course_type, subject_name=subject_type, stream_type=stream_name
                                                                )
                message = rag_function._create_embedding_selection(
                                                                    pdf_details['pdf_id'], pdf_details['pdf_path'], school_college_competitiveexam,
                                                                    college_name=college_name_type, course_name=course_type,
                                                                    subject_name=subject_type, stream_type=stream_name
                                                                )
            else:
                pass
        else:
            invalid_files.append(file.filename)

    if invalid_files != []:
        invalid_message = f'{invalid_files} files are invalid'
        return jsonify({'message': f'PDF Uploaded Successfully and {invalid_message}', 'status': 201})
    else:
        return jsonify(message)


@app.route('/rag-model-hierarchy', methods=['GET', 'POST'])
def rag_model_hierarchy():
    user_query = request.args.get("user_query")
    student_id = request.args.get('student_id')
    school_college_competitiveexam = request.args.get('s_c_ce_type')
    board_type = request.args.get('board_type') or None
    state_board = request.args.get('state_board_type') or None
    class_name = request.args.get('class_name') or None
    stream_name = request.args.get('stream_name') or None
    college_name_type = request.args.get('college_name_type') or None
    course_type = request.args.get('course_type') or None
    subject_type = request.args.get('subject_type') or None

    if not user_query or not student_id:
        return jsonify({'message': 'Missing query or student ID', 'status': 400})

    data = {
        "school_college_ce": school_college_competitiveexam,
        "board": board_type,
        "state_board": state_board,
        "class_name": class_name,
        "stream": stream_name,
        "college_name": college_name_type,
        "stream_name": course_type,
        "subject": subject_type
    }

    try:
        answer = rag_function._get_answer_to_query_selection(user_query, data)
        return jsonify(answer)

    except Exception as e:
        return jsonify({'message': f'Error generating answer: {str(e)}', 'status': 404})


@app.route('/display-files', methods=['GET', 'POST'])
def display_files():
    class_name = request.args.get('class_name', "None")
    admin_id = request.args.get('admin_id', "")
    try:
        file_names = rag_function._display_files(admin_id, class_name)
        return jsonify(file_names)
    except Exception as e:
        return jsonify({'message': f'Error fetching files {str(e)}', 'status': 404})


@app.route('/delete-files', methods=['DELETE'])
def delete_files():
    # Get JSON payload from the request
    data = request.json
    file_id = data.get('file_id')
    file_name = data.get('file_name')
    file_path = data.get('file_path')
    class_name = data.get('class_name', "None")

    # Check if file_path is provided
    if not file_path and not file_id:
        return jsonify({'message': 'Please provide a file to delete', 'status': 400})

    # Call the function to delete files
    delete_message = rag_function._delete_files(file_id, file_name, class_name)
    try:
        if delete_message['status'] == 200:
            return jsonify({'message': 'PDF deleted successfully', 'status': 201})
        else:
            return jsonify({'message': 'There is no such PDF Files to delete', 'status': 401})
    except Exception:
        return jsonify({'message': 'PDF not deleted successfully',
                        'status': 403})


@app.route('/files/<path:filename>')
def serve_file(filename):
    bucket_name = constants.BUCKETNAME
    s3_client = boto3.client(
                's3',
                aws_access_key_id=constants.ACCESS_KEY,
                aws_secret_access_key=constants.SECRET_KEY,
                region_name=constants.REGION
            )
    s3_key = filename
    try:
        # Fetch file from S3
        file_obj = s3_client.get_object(Bucket=bucket_name, Key=s3_key)

        # Stream the PDF file directly in the browser
        return Response(
            file_obj['Body'].read(),
            content_type='application/pdf',  # Ensure correct MIME type for PDF
            headers={
                "Content-Disposition": f"inline; filename={filename.split('/')[-1]}"  # Display PDF inline
            }
        )
    except NoCredentialsError:
        return "Invalid S3 credentials", 403
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return "File not found in S3", 404
        else:
            return f"Error fetching file from S3: {e}", 500


@app.route('/update-changes', methods=['GET', 'POST'])
def update_changes():
    teacher_id = request.args.get('teacher_id', "None")
    bucket_name = constants.BUCKETNAME
    s3_client = boto3.client(
                's3',
                aws_access_key_id=constants.ACCESS_KEY,
                aws_secret_access_key=constants.SECRET_KEY,
                region_name=constants.REGION
            )
    # Initialize the paginator to handle large result sets
    # Initialize the paginator to handle large result sets
    prefix = constants.PDF_DIRECTORY_CLASS
    paginator = s3_client.get_paginator('list_objects_v2')

    # Use the paginator to iterate over all objects in the bucket with the given prefix (empty for root)
    response_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    file_dict = {}

    for page in response_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                file_key = obj['Key']

                # Extract the file name from the file key
                file_name = file_key.split('/')[-1] if '/' in file_key else file_key

                # Ignore if file_name is an empty string
                if not file_name:
                    continue

                # Extract the collection name (subdirectories between prefix and file_name)
                collection_parts = file_key[len(prefix):].split('/')[:-1]  # Remove the file name
                collection_name = "_".join(part for part in collection_parts if part)  # Join by underscore

                # Store the file name as key and file path and collection name as sub-keys
                file_dict[file_name] = {
                    'file_path': file_key,  # Full path of the file
                    'collection_name': collection_name  # Concatenated subdirectory names
                }

    db = PDFDataDatabase()
    db.delete_all()
    for file, value in file_dict.items():
        if file.endswith(".pdf"):
            pdf_details = rag_function._pdf_file_process(file, value["file_path"], teacher_id)
            message = rag_function._create_embedding_all(pdf_details['pdf_id'], pdf_details['pdf_path'], value["collection_name"])

    return jsonify(message)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
