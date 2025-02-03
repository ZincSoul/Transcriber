import os
from flask import Flask, request, jsonify, send_from_directory
from pydub import AudioSegment
import speech_recognition as sr
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(file_path)
    
    temp_file = "temp.wav"
    audio.export(temp_file, format="wav")
    
    with sr.AudioFile(temp_file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    
    return text

def summarize_text(text):
    summarizer = pipeline("summarization", model="t5-base", framework="tf")
    summary = summarizer(text, max_length=150, min_length=25, do_sample=False)
    return summary[0]['summary_text']

@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"
    if file:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        text = transcribe_audio(file_path)
        summary = summarize_text(text)
        return jsonify({"transcription": text, "summary": summary})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
