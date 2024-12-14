# transcribe.py
import whisper
import sys
import json

def transcribe_audio(audio_path):
    try:

        # Load the model
        model = whisper.load_model("base")


        
        # Perform transcription
        result = model.transcribe(audio_path)
        
        # Format the response
        response = {
            "text": result["text"],
            "language": result["language"],
            "segments": result["segments"],
            "duration": result.get("duration", 0)
        }
        
        # Print as JSON for Node.js to parse
        print(json.dumps(response))
        
    except Exception as e:
        error_response = {
            "error": str(e)
        }
        print(json.dumps(error_response))
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Please provide an audio file path"}))
        sys.exit(1)
        
    audio_path = sys.argv[1]
    transcribe_audio(audio_path)