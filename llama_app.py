from dependency import *
from llama_rag_model import llama_model

app = Flask(__name__)
CORS(app)
app.secret_key = secrets.token_hex(16)
in_memory_cache = {}

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
    response = jsonify({'message': 'Query processed successfully', 'status': 200, 'question':user_query, 'answer': answer})
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
            rag_function._pdf_file_save_class(file , class_name)
        else:
            invalid_files.append(file.filename)
            
    message = rag_function._create_embedding(class_name)

    if invalid_files != []:
        invalid_message = f'{invalid_files} files are invalid'
        return jsonify({'message': f'PDF Uploaded Successfully and {invalid_message}', 'status': 201})
    
    else:
        return jsonify(message)
    
@app.route('/rag-model-class', methods=['GET', 'POST'])
def rag_model_class():
    user_query = request.args.get("user_query")
    student_id = request.args.get('student_id')
    class_name = request.args.get('class_name',"")
    
    if not user_query or not student_id:
        return jsonify({'message': 'Missing query or student ID', 'status': 400})
 
    if student_id not in session:
        data = {"class_name":class_name}
        session[student_id] = data

    elif class_name != "":
        data = session[student_id]
        data.update({"class_name":class_name})

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
            rag_function._pdf_file_save(file)
        else:
            invalid_files.append(file.filename)
            
    message = rag_function._create_embedding_all()

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

if __name__ == "__main__":
    app.run(host='0.0.0.0')