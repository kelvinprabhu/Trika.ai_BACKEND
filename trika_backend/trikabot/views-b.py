# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
from .llm_client2 import Llama3AgentWithSemanticMemory

# Create one agent instance with semantic memory
agent = Llama3AgentWithSemanticMemory()

@csrf_exempt
@require_POST 
def Chat(request):
    """Enhanced chat endpoint with semantic memory and current conversation awareness."""
    if request.method == 'POST':
        try:
            # Parse request body as JSON
            data = json.loads(request.body)

            user_email = data.get('userEmail')
            query = data.get('Query')
            session_id = data.get('sessionId')

            # Validate input
            if not user_email or not query or not session_id:
                return JsonResponse({
                    "error": "Missing required fields",
                    "required": ["userEmail", "Query", "sessionId"]
                }, status=400)

            # Get response from AI agent with semantic memory
            ai_response = agent.ask(query, session_id=session_id, user_email=user_email)

            return JsonResponse({
                # "message": f"Response for {user_email}:",
                "message": ai_response,
                # "session_id": session_id
            })

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
        except Exception as e:
            return JsonResponse({"error": f"Server error: {str(e)}"}, status=500)

    else:
        return JsonResponse({"error": "Only POST requests are allowed"}, status=405)

@csrf_exempt
@require_POST 
def refresh_conversation_data(request):
    """Refresh the knowledge base with latest conversation and user data."""
    try:
        data = json.loads(request.body)
        user_email = data.get('userEmail')
        session_id = data.get('sessionId')
        
        if not user_email or not session_id:
            return JsonResponse({
                "error": "Missing required fields",
                "required": ["userEmail", "sessionId"]
            }, status=400)
        
        success = agent.refresh_session_data(session_id, user_email)
        
        if success:
            return JsonResponse({
                "message": "Conversation data refreshed successfully",
                "session_id": session_id
            })
        else:
            return JsonResponse({"error": "Failed to refresh conversation data"}, status=500)
            
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON format"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Server error: {str(e)}"}, status=500)

@csrf_exempt
@require_POST 
def get_conversation_summary(request):
    """Get summary of the current conversation from memory."""
    try:
        data = json.loads(request.body)
        session_id = data.get('sessionId')
        
        if not session_id:
            return JsonResponse({
                "error": "Missing sessionId"
            }, status=400)
        
        summary = agent.get_conversation_summary(session_id)
        return JsonResponse({
            "summary": summary,
            "session_id": session_id
        })
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON format"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Server error: {str(e)}"}, status=500)

@csrf_exempt
@require_POST 
def cleanup_session(request):
    """Cleanup resources for a specific session."""
    try:
        data = json.loads(request.body)
        session_id = data.get('sessionId')
        
        if not session_id:
            return JsonResponse({
                "error": "Missing sessionId"
            }, status=400)
        
        agent.cleanup_session(session_id)
        return JsonResponse({
            "message": "Session cleaned up successfully",
            "session_id": session_id
        })
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON format"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Server error: {str(e)}"}, status=500)

# Optional: Health check endpoint
@csrf_exempt
def health_check(request):
    """Simple health check for the chatbot service."""
    return JsonResponse({
        "status": "healthy",
        "service": "semantic_chatbot",
        "active_sessions": len(agent.current_session_data)
    })