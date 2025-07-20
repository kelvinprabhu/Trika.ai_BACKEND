from channels.generic.websocket import AsyncWebsocketConsumer
import json
from trikavision.models.posture_predictor import predict_from_landmark_frame

class PostureConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def receive(self, text_data):
        data = json.loads(text_data)
        landmarks = data.get("landmarks", [])

        if not landmarks:
            await self.send(text_data=json.dumps({"error": "No landmarks received"}))
            return

        workout, confidence = predict_from_landmark_frame(landmarks)

        await self.send(text_data=json.dumps({
            "workout": workout,
            "confidence": f"{confidence:.2f}",
            "feedback": "Keep your core engaged.",
            "landmarks": landmarks
        }))

    async def disconnect(self, close_code):
        print("WebSocket Disconnected:", close_code)
