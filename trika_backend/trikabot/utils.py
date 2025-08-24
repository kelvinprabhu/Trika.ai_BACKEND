import json
import requests
import os
from typing import Dict, List
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from django.conf import settings
# --- Step 1: Data Fetching and Preprocessing (Same as before) ---
def fetch_user_data(email: str) -> Dict:
    """Fetches user data from your local API."""
    url = f"http://172.24.112.1:3000/api/user/complete-info?email={email}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()["data"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None

def preprocess_and_chunk_data(data: Dict) -> List[Document]:
    """
    Transforms the raw JSON data into a list of LangChain Document objects.
    Each Document represents a text chunk with associated metadata.
    """
    documents = []
    user_info = data["user"]
    
    user_profile_text = (
        f"User: {user_info['firstName']} {user_info['lastName']}, Email: {user_info['email']}. "
        f"Age: {user_info['personalInfo']['age']}, Gender: {user_info['personalInfo']['gender']}, "
        f"Height: {user_info['personalInfo']['height']}cm, Weight: {user_info['personalInfo']['weight']}kg. "
        f"Activity Level: {user_info['personalInfo']['activityLevel']}. "
        f"Primary Goal: {user_info['fitnessGoals']['primaryGoal']}, "
        f"Fitness Level: {user_info['fitnessGoals']['fitnessLevel']}, "
        f"Workout Frequency: {user_info['fitnessGoals']['workoutFrequency']} times a week, "
        f"Preferred Workouts: {', '.join(user_info['fitnessGoals']['preferredWorkouts'])}."
    )
    documents.append(Document(
        page_content=user_profile_text,
        metadata={"source": "user_profile", "user_email": user_info['email'], "id": user_info['id']}
    ))

    for habit in data["habits"]:
        habit_text = (
            f"User {user_info['email']} has a habit named '{habit['name']}' in the category '{habit['category']}'. "
            f"Description: '{habit.get('description', 'N/A')}'. "
            f"Current streak: {habit['streak']}, best streak: {habit['bestStreak']}. "
            f"Completed {habit['completedThisWeek']} times this week."
        )
        documents.append(Document(
            page_content=habit_text,
            metadata={"source": "habit", "user_email": user_info['email'], "id": habit['_id'], "name": habit['name']}
        ))

    for schedule in data["schedules"]:
        schedule_text = (
            f"User {user_info['email']} has a schedule titled '{schedule['title']}'. "
            f"It's a {schedule['type']} event, status: {schedule['status']}, "
            f"Date: {schedule['date'].split('T')[0]}, Time: {schedule['time']}. "
            f"Priority: {schedule['priority']}. "
            f"It repeats: {schedule['repeat']}. Description: '{schedule.get('description', 'N/A')}'."
        )
        documents.append(Document(
            page_content=schedule_text,
            metadata={"source": "schedule", "user_email": user_info['email'], "id": schedule['_id'], "title": schedule['title']}
        ))

    for conversation in data["conversations"]:
        messages = [
            f"{msg['role']}: {msg['content']}" for msg in conversation['messages']
        ]
        conversation_text = (
            f"Conversation for user {user_info['email']}. "
            f"Session ID: {conversation['sessionId']}. "
            f"Messages: {' | '.join(messages)}"
        )
        documents.append(Document(
            page_content=conversation_text,
            metadata={"source": "conversation", "user_email": user_info['email'], "id": conversation['_id']}
        ))

    return documents

# --- Step 2 & 3: Embedding, Storage, and Retriever Creation (Same as before) ---
def create_retriever(documents: List[Document]):
    """
    Generates embeddings for the documents using a Hugging Face model,
    stores them in a Chroma vector store, and returns a retriever object.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db_hf"
    )
    
    retriever = vectorstore.as_retriever()
    print("Vector store and retriever created successfully using a Hugging Face model.")
    return retriever

# --- Step 4: Building the Retrieval QA Chain with a Custom Prompt ---
def create_retrieval_qa_chain(retriever):
    """
    Initializes the Groq LLM and creates a RetrievalQA chain with a custom prompt template.
    """
    GROQ_API_KEY = settings.GROQ_API_KEY
    # Initialize the Groq LLM
    # You can choose a different model like 'mixtral-8x7b-32768' if you prefer
    llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=GROQ_API_KEY)

    # --------------------------------------------------------------------------------
    # Define the custom prompt template
    # --------------------------------------------------------------------------------
    template = """
    You are the AI challenge generator for the **Trika AI Fitness Platform**.

    **About TrikaVision:** TrikaVision is an AI-powered workout experience that guides users through interactive, follow-along fitness challenges using computer vision and real-time form tracking. It adapts exercises based on the user’s fitness level, goals, and available equipment, ensuring sessions are engaging, safe, and achievable.

    Use the provided retrieved user knowledge (including habits, schedule, fitness information, goals, preferred workout type, and available equipment) to create up to **2** TrikaVision challenges.

    **Rules:**

    1. Use only the retrieved user data — no assumptions beyond that.
    2. Total duration of each challenge must be **30 minutes or less**.
    3. Maximum **2 challenges** per request.
    4. Output must be **strictly valid JSON** — no extra text, no explanation, no markdown.
    5. Each challenge must include:
       * "userEmail" (string)
       * "title" (string)
       * "description" (string)
       * "category": "TrikaVision"
       * "type" (string) — should match preferred workout type from user info
       * "fitnessLevel" (string)
       * "goalTags" (array of strings)
       * "workout" (string)
       * "exercises" (array) — each with:
         * "exercise" (string)
         * "durationMinutes" (integer)
         * "reason" (string)
       * "aiRecommendationReason" (string)
       * "status": "active"

    **Final output format example:**
    ```json
    [
    {{
    "userEmail": "kelvinprabhu2071@gmail.com",
    "title": "Warm-up and Cardio Blast",
    "description": "A quick, high-energy cardio session to boost endurance.",
    "category": "TrikaVision",
    "type": "Home Workout",
    "fitnessLevel": "Beginner",
    "goalTags": ["cardio", "endurance"],
    "workout": "Warm-up and Cardio Blast",
    "exercises": [
    {{
    "exercise": "Jumping Jacks",
    "durationMinutes": 3,
    "reason": "Get those endorphins pumping and warm up your muscles."
    }},
    {{
    "exercise": "Burpees",
    "durationMinutes": 2,
    "reason": "Increase your heart rate and improve cardio endurance."
    }}
    ],

    "aiRecommendationReason": "Selected based on your beginner fitness level and cardio goal.",
   
    "status": "active"
    }}
    ]
    ```

    Answer the following question based on the provided context. Make sure your final answer is a JSON array that strictly follows the format example above.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    YOUR RESPONSE IN JSON:
    """

    # Create the prompt template
    # The ChatPromptTemplate uses a list of message templates
    custom_rag_prompt = ChatPromptTemplate.from_template(template)

    # --------------------------------------------------------------------------------

    # Create the RetrievalQA chain, passing the custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        # Pass the custom prompt here via chain_type_kwargs
        chain_type_kwargs={"prompt": custom_rag_prompt}
    )

    print("RetrievalQA chain created successfully with a custom prompt.")
    return qa_chain

