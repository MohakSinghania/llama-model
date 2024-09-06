import torch
import ollama
import secrets
from flask_cors import CORS
from llama_rag_model import llama_model
from flask import Flask, request, jsonify, session, send_file

app = Flask(__name__)
CORS(app)
app.secret_key = secrets.token_hex(16)
in_memory_cache = {}
ALLOWED_BASE_DIR = '/home/ubuntu/llama-model/pdf_files'

rag_function = llama_model()


@app.route('/ollama-chat', methods=['GET', 'POST'])
def ollama_chat():
    user_query = request.args.get("user_query")
    if not user_query:
        return jsonify({'message': 'Missing query or student ID', 'status': 400})
    response = ollama.chat(
        model="llama3",
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

    # pdf_file_path = os.path.join(constants.PDF_DIRECTORY, class_name)
    # if not os.path.exists(pdf_file_path):
    #     return jsonify({'message': f'There is no PDF for {class_name}', 'status': 404})

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


@app.route('/display-files', methods=['GET', 'POST'])
def display_files():
    class_name = request.args.get('class_name', "None")
    admin_id = request.args.get('admin_id', "")
    # if not class_name:
    #     return jsonify({'message': 'Please Provide Class Name', 'status': 400})
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
    delete_message = rag_function._delete_files(file_id, file_name, file_path, class_name)
    try:
        if delete_message['status'] == 201:
            return jsonify({'message': 'PDF deleted successfully', 'status': 201})
        else:
            return jsonify({'message': 'There is no such PDF Files to delete', 'status': 401})
    except Exception:
        return jsonify({'message': 'PDF not deleted successfully',
                        'status': 403})


@app.route('/files/<path:filename>')
def serve_file(filename):
    prefix = "/home/ubuntu/llama-model/pdf_files/"

    # Ensure filename has a leading slash
    if not filename.startswith('/'):
        filename = '/' + filename

    # Perform the replacement
    new_filename = filename.replace(prefix, "")
    try:
        # Ensure filename is the full path
        filepath = f"/home/ubuntu/llama-model/pdf_files/{new_filename}"
        return send_file(filepath)
    except FileNotFoundError:
        return "File not found", 404


if __name__ == "__main__":
    app.run(host='0.0.0.0')
