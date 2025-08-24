import json
import requests
import os
from typing import Dict, List, Optional
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from django.conf import settings

class Llama3AgentWithMemory:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name="llama3-70b-8192"
        )
        self.vector_stores = {}  # Store vector databases per session
        self.retrievers = {}     # Store retrievers per session
        self.qa_chains = {}      # Store QA chains per session
        self.current_session_data = {}  # Cache current session data
        
    def fetch_user_data(self, email: str) -> Optional[Dict]:
        """Fetches user data from your local API."""
        url = f"http://172.24.112.1:3000/api/user/complete-info?email={email}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()["data"]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching user data from API: {e}")
            return None
    
    def fetch_conversation_history(self, session_id: str, user_email: str) -> Optional[Dict]:
        """Fetches conversation history for a specific session."""
        url = f"http://172.24.112.1:3000/api/chatbot/{session_id}?userEmail={user_email}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching conversation history: {e}")
            return None
    
    def preprocess_and_chunk_data(self, data: Dict, conversation_data: Optional[Dict] = None) -> List[Document]:
        """
        Transforms the raw JSON data into a list of LangChain Document objects.
        Includes current conversation history if provided.
        """
        documents = []
        user_info = data["user"]
        
        # User profile information
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

        # Habits information
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

        # Schedules information
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

        # Add current conversation history if provided
        if conversation_data and "conversation" in conversation_data:
            conv = conversation_data["conversation"]
            messages_text = []
            for msg in conv["messages"]:
                messages_text.append(f"{msg['type']}: {msg['content']} (at {msg['timestamp']})")
            
            conversation_text = (
                f"Current conversation for user {user_info['email']}. "
                f"Session ID: {conv['sessionId']}. Title: {conv['title']}. "
                f"Total messages: {conv['totalMessages']}. "
                f"Conversation history: {' | '.join(messages_text)}"
            )
            documents.append(Document(
                page_content=conversation_text,
                metadata={"source": "current_conversation", "user_email": user_info['email'], "session_id": conv['sessionId']}
            ))

        # Add historical conversations (excluding current one if it exists)
        for conversation in data["conversations"]:
            if conversation_data and conversation.get("sessionId") == conversation_data["conversation"]["sessionId"]:
                continue  # Skip current conversation as we already added it with full messages
                
            messages = [
                f"{msg['role']}: {msg['content']}" for msg in conversation['messages']
            ]
            conversation_text = (
                f"Historical conversation for user {user_info['email']}. "
                f"Session ID: {conversation['sessionId']}. "
                f"Messages: {' | '.join(messages)}"
            )
            documents.append(Document(
                page_content=conversation_text,
                metadata={"source": "historical_conversation", "user_email": user_info['email'], "id": conversation['_id']}
            ))

        return documents
    
    def setup_retriever_for_session(self, session_id: str, user_email: str) -> bool:
        """Sets up or updates the retriever for a specific session."""
        try:
            # Check if we need to update (new session or no existing data)
            if (session_id not in self.current_session_data or 
                self.current_session_data[session_id].get("user_email") != user_email):
                
                print(f"Setting up retriever for new session: {session_id}")
                
                # Fetch user data
                user_data = self.fetch_user_data(user_email)
                if not user_data:
                    print(f"Failed to fetch user data for {user_email}")
                    return False
                
                # Fetch current conversation history
                conversation_data = self.fetch_conversation_history(session_id, user_email)
                
                # Process and chunk the data
                documents = self.preprocess_and_chunk_data(user_data, conversation_data)
                
                # Create or update vector store for this session
                vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=f"session_{session_id}".replace("-", "_")  # Clean collection name
                )
                
                # Store the vector store and create retriever
                self.vector_stores[session_id] = vector_store
                self.retrievers[session_id] = vector_store.as_retriever(search_kwargs={"k": 5})
                
                # Create QA chain for this session
                system_prompt = """You are a fitness and wellness assistant. Answer questions using ONLY the provided context data.

RESPONSE RULES:
1. Keep answers SHORT and PRECISE - 1-6 sentences maximum
2. Answer ONLY what is directly asked - no extra information
3. Use EXACT data from context (numbers, dates, names)
4. If context doesn't contain the answer, say "I don't have that information in your current data"
5. No generic advice - only personalized responses based on user's specific data

Context includes: user profile, habits with streaks/progress, schedules/events, conversation history.

Format: Direct answer → One actionable suggestion (if relevant)"""
                prompt_template = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(system_prompt),
                    HumanMessagePromptTemplate.from_template("Context: {context}\n\nQuestion: {question}")
                ])
                
                self.qa_chains[session_id] = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.retrievers[session_id],
                    chain_type_kwargs={"prompt": prompt_template},
                    return_source_documents=True
                )
                
                # Cache session data
                self.current_session_data[session_id] = {
                    "user_email": user_email,
                    "last_updated": "now"  # You might want to use actual timestamps
                }
                
                print(f"Successfully set up retriever for session {session_id}")
                return True
            else:
                print(f"Using existing retriever for session {session_id}")
                return True
                
        except Exception as e:
            print(f"Error setting up retriever for session {session_id}: {e}")
            return False
    
    def ask(self, query: str, session_id: str, user_email: str = None) -> str:
        """
        Main method to ask questions to the AI agent.
        Automatically updates retriever if needed for the session.
        """
        try:
            # Extract user email from existing session data if not provided
            if not user_email and session_id in self.current_session_data:
                user_email = self.current_session_data[session_id].get("user_email")
            
            if not user_email:
                return "Error: User email is required for new sessions."
            
            # Setup or update retriever for this session
            if not self.setup_retriever_for_session(session_id, user_email):
                return "Error: Failed to set up knowledge retriever for this session."
            
            # Get QA chain for this session
            qa_chain = self.qa_chains.get(session_id)
            if not qa_chain:
                return "Error: QA chain not found for this session."
            
            # Get response from the AI
            result = qa_chain({"query": query})
            
            return result["result"]
            
        except Exception as e:
            print(f"Error in ask method: {e}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"
    
    def refresh_session_data(self, session_id: str, user_email: str) -> bool:
        """Manually refresh the data for a specific session."""
        # Remove cached data to force refresh
        if session_id in self.current_session_data:
            del self.current_session_data[session_id]
        if session_id in self.vector_stores:
            del self.vector_stores[session_id]
        if session_id in self.retrievers:
            del self.retrievers[session_id]
        if session_id in self.qa_chains:
            del self.qa_chains[session_id]
            
        return self.setup_retriever_for_session(session_id, user_email)
    
    def cleanup_session(self, session_id: str):
        """Clean up resources for a specific session."""
        if session_id in self.current_session_data:
            del self.current_session_data[session_id]
        if session_id in self.vector_stores:
            del self.vector_stores[session_id]
        if session_id in self.retrievers:
            del self.retrievers[session_id]
        if session_id in self.qa_chains:
            del self.qa_chains[session_id]
        print(f"Cleaned up resources for session {session_id}")