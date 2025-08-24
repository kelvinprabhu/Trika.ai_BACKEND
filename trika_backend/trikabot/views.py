from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from trikabot.utils import (
    fetch_user_data,
    preprocess_and_chunk_data,
    create_retriever,
    create_retrieval_qa_chain,
)
import json
# Create your views here.
from .llm_client import Llama3AgentWithMemory

# Create one agent instance for the entire server to use
agent = Llama3AgentWithMemory()

def index(request):
    from django.http import JsonResponse
    return JsonResponse({"message": "trikabot is working"})

@csrf_exempt
@require_POST 
def Chat(request):
    if request.method == 'POST':
        try:
            # Parse request body as JSON
            data = json.loads(request.body)

            user_email = data.get('userEmail')
            query = data.get('Query')
            session_id = data.get('sessionId')

            # Validate input
            if not user_email or not query or not session_id:
                return JsonResponse({"error": "Missing userEmail, Query, or sessionId"}, status=400)
            

            # Get response from AI agent (now includes user_email parameter)
            ai_response = agent.ask(query, session_id=session_id, user_email=user_email)
            agent.refresh_session_data(session_id, user_email)

            return JsonResponse({
                # "message": f"Hey {user_email}, here is the bot response:{session_id}",
                "message": ai_response
            })

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    else:
        return JsonResponse({"error": "Only POST requests are allowed"}, status=405)

# Optional: Add an endpoint to manually refresh session data
@csrf_exempt
@require_POST 
def refresh_session(request):
    """Manually refresh the knowledge base for a specific session."""
    try:
        data = json.loads(request.body)
        user_email = data.get('userEmail')
        session_id = data.get('sessionId')
        
        if not user_email or not session_id:
            return JsonResponse({"error": "Missing userEmail or sessionId"}, status=400)
        
        success = agent.refresh_session_data(session_id, user_email)
        
        if success:
            return JsonResponse({"message": "Session data refreshed successfully"})
        else:
            return JsonResponse({"error": "Failed to refresh session data"}, status=500)
            
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# Optional: Add an endpoint to cleanup session resources
@csrf_exempt
@require_POST 
def cleanup_session(request):
    """Cleanup resources for a specific session."""
    try:
        data = json.loads(request.body)
        session_id = data.get('sessionId')
        
        if not session_id:
            return JsonResponse({"error": "Missing sessionId"}, status=400)
        
        agent.cleanup_session(session_id)
        return JsonResponse({"message": "Session cleaned up successfully"})
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
# @csrf_exempt
# @require_POST 
# def Chat(request):
#     if request.method == 'POST':
#         try:
#             # Parse request body as JSON
#             data = json.loads(request.body)

#             user_email = data.get('userEmail')
#             query = data.get('Query')
#             session_id = data.get('sessionId')

#             # Validate input
#             if not user_email or not query or not session_id:
#                 return JsonResponse({"error": "Missing userEmail, Query, or sessionId"}, status=400)

#             # Get response from AI agent
#             ai_response = agent.ask(query, session_id=session_id)

#             return JsonResponse({
#                 "message": f"Hey {user_email}, here is the bot response:",
#                 "response": ai_response
#             })

#         except json.JSONDecodeError:
#             return JsonResponse({"error": "Invalid JSON"}, status=400)
#         except Exception as e:
#             return JsonResponse({"error": str(e)}, status=500)

#     else:
#         return JsonResponse({"error": "Only POST requests are allowed"}, status=405)



@csrf_exempt
@require_POST
def generate_challenge(request):
    """
    Django endpoint to generate a fitness challenge for a user.

    Expects a POST request with a JSON body containing the user's email.
    """
    try:
        data = json.loads(request.body)
        email = data.get("userEmail")

        if not email:
            return JsonResponse({"error": "Email is required"}, status=400)

        # Step 1: Fetch and preprocess user data
        user_data = fetch_user_data(email)
        if not user_data:
            return JsonResponse({"error": "Failed to fetch user data"}, status=500)

        documents = preprocess_and_chunk_data(user_data)

        # Step 2: Create retriever from documents
        retriever = create_retriever(documents)

        # Step 3: Create the Retrieval QA chain
        qa_chain = create_retrieval_qa_chain(retriever)

        # Step 4: Invoke the chain with a specific query
        query = "generate 1 challenge for my profile"
        response = qa_chain.invoke({"query": query})

        # The response is a JSON string, so we parse it
        try:
            challenge_json = json.loads(response['result'])
            return JsonResponse(challenge_json, safe=False, status=200)
        except json.JSONDecodeError:
            return JsonResponse(
                {"error": "Failed to parse challenge JSON from AI response"},
                status=500
            )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON format"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)