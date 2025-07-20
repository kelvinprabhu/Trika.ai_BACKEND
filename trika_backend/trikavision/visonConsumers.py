from channels.generic.websocket import AsyncWebsocketConsumer
import json

class PostureConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def receive(self, text_data):
        data = json.loads(text_data)
        landmarks = data.get("landmarks", [])

        print("✅ Received landmarks from frontend:")
        for idx, lm in enumerate(landmarks[:5]):
            print(f"  [{idx}] x: {lm['x']:.3f}, y: {lm['y']:.3f}, z: {lm['z']:.3f}")

        response = {
            "workout": "Push-Up",
            "feedback": "Keep your back straight.",
            "landmarks": landmarks  # Echo full landmarks back
        }

        await self.send(text_data=json.dumps(response))

    async def disconnect(self, close_code):
        print("❌ WebSocket Disconnected:", close_code)
