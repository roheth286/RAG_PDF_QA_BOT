"""
Gemini PDF Q&A Bot - Flask Web Server

This module runs the Flask application serving the web interface. It handles uploading
PDF documents, initializing/resetting the RAG pipeline, and routing user chat messages
to the QA bot.
"""

import os
from flask import Flask, request, jsonify, render_template
import qabot

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        try:
            # Initialize RAG pipeline with the uploaded PDF
            qabot.initialize_rag(file_path)
            # Reset conversation history for the new document
            qabot.history = []
            return jsonify({
                'message': 'PDF uploaded and analyzed successfully!',
                'filename': file.filename
            })
        except Exception as e:
            return jsonify({'error': f'Failed to process PDF: {str(e)}'}), 500
            
    return jsonify({'error': 'Only PDF files are supported'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    if qabot.qa_chain is None:
        return jsonify({'error': 'No document has been analyzed yet. Please upload a PDF first.'}), 400
        
    data = request.get_json() or {}
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'error': 'Message cannot be empty'}), 400
        
    try:
        answer = qabot.ask_question(message)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': f'Error generating response: {str(e)}'}), 500

if __name__ == '__main__':
    # Running locally
    app.run(host='127.0.0.1', port=5000, debug=True)
