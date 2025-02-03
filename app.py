import os
from flask import Flask, request, jsonify, render_template
from pydub import AudioSegment
import speech_recognition as sr
from transformers import pipeline

app = Flask(__name__)

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
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    transcript = transcribe_audio(file_path)
    summary = summarize_text(transcript)

    return jsonify({"transcript": transcript, "summary": summary})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
