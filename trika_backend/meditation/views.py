from django.shortcuts import render
import json
from django.http import FileResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from pathlib import Path
from pathlib import Path
from django.conf import settings
import shutil

def index(request):
    from django.http import JsonResponse
    return JsonResponse({"message": "meditation is working"})

@csrf_exempt
def generate_meditation_session(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        # Parse incoming JSON
        data = json.loads(request.body.decode("utf-8"))

        # Extract values (example)
        session_name = data.get("sessionName", "session")
        binaural_frequency = data.get("binauralFrequency", "theta")
        duration = data.get("duration", 10)
        ambient_sounds = data.get("ambientSounds", [])
        
        # --- Here you'd generate the audio file dynamically ---
        # For now, we'll just return an existing example audio file.
        # Example: static/sample_audio.mp3
        audio_path = Path("static/meditationSessionAudio/sample_audio.mp3")
        
        if not audio_path.exists():
            return JsonResponse({"error": "Audio file not found"}, status=404)
        
        # Return file as downloadable response  
        return FileResponse(open(audio_path, "rb"), as_attachment=True, filename=f"{session_name}.mp3")

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON format"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
        session_name = data.get("sessionName", "session")

        # Example: copy a sample audio to media folder
        source_audio = Path("static/meditationSessionAudio/sample_audio.mp3")
        target_dir = Path(settings.MEDIA_ROOT) / "sessions"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_audio = target_dir / f"{session_name}.mp3"
        shutil.copy(source_audio, target_audio)

        # Return the public URL to the audio file
        audio_url = f"{settings.MEDIA_URL}sessions/{session_name}.mp3"
        return JsonResponse({"audio_url": audio_url})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)