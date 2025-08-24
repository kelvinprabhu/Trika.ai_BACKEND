import json
import requests
import os
from typing import Dict, List, Optional
from datetime import datetime
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from django.conf import settings

class Llama3AgentWithSemanticMemory:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name="llama3-70b-8192"
        )
        self.vector_stores = {}  # Store vector databases per session
        self.retrievers = {}     # Store retrievers per session
        self.qa_chains = {}      # Store QA chains per session
        self.memories = {}       # Store conversation memories per session
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
    
    def fetch_all_conversations(self, user_email: str) -> Optional[Dict]:
        """Fetches all conversations for a user."""
        url = f"http://172.24.112.1:3000/api/chatbot?userEmail={user_email}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching all conversations: {e}")
            return None
    
    def fetch_current_conversation(self, session_id: str, user_email: str) -> Optional[Dict]:
        """Fetches current conversation history for a specific session."""
        url = f"http://172.24.112.1:3000/api/chatbot/{session_id}?userEmail={user_email}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching current conversation: {e}")
            return None
    
    def create_semantic_documents(self, data: Dict, all_conversations: Optional[Dict] = None, current_conversation: Optional[Dict] = None) -> List[Document]:
        """
        Creates semantic documents with better structure for retrieval.
        Separates current conversation from historical data for better context.
        """
        documents = []
        user_info = data["user"]
        
        # === USER PROFILE - Comprehensive Profile ===
        user_profile_text = (
            f"USER PROFILE: {user_info['firstName']} {user_info['lastName']} ({user_info['email']})\n"
            f"Physical Stats: {user_info['personalInfo']['age']} years old, {user_info['personalInfo']['gender']}, "
            f"{user_info['personalInfo']['height']}cm tall, {user_info['personalInfo']['weight']}kg\n"
            f"Activity Level: {user_info['personalInfo']['activityLevel']}\n"
            f"FITNESS GOALS: Primary goal is {user_info['fitnessGoals']['primaryGoal']}\n"
            f"Current fitness level: {user_info['fitnessGoals']['fitnessLevel']}\n"
            f"Workout frequency: {user_info['fitnessGoals']['workoutFrequency']} times per week\n"
            f"Preferred workout types: {', '.join(user_info['fitnessGoals']['preferredWorkouts'])}"
        )
        documents.append(Document(
            page_content=user_profile_text,
            metadata={"source": "user_profile", "type": "profile", "user_email": user_info['email']}
        ))

        # === CURRENT HABITS STATUS ===
        if data.get("habits"):
            habits_summary = f"CURRENT HABITS STATUS for {user_info['email']}:\n"
            for habit in data["habits"]:
                habits_summary += (
                    f"• {habit['name']} ({habit['category']}): {habit['streak']} day streak, "
                    f"best streak {habit['bestStreak']} days, completed {habit['completedThisWeek']} times this week\n"
                )
            
            documents.append(Document(
                page_content=habits_summary,
                metadata={"source": "habits", "type": "current_status", "user_email": user_info['email']}
            ))

        # === UPCOMING SCHEDULES ===
        if data.get("schedules"):
            schedules_text = f"UPCOMING SCHEDULES for {user_info['email']}:\n"
            for schedule in data["schedules"]:
                schedules_text += (
                    f"• {schedule['title']} ({schedule['type']}): {schedule['date'].split('T')[0]} at {schedule['time']}\n"
                    f"  Status: {schedule['status']}, Priority: {schedule['priority']}, Repeats: {schedule['repeat']}\n"
                )
            
            documents.append(Document(
                page_content=schedules_text,
                metadata={"source": "schedules", "type": "upcoming", "user_email": user_info['email']}
            ))

        # === CURRENT CONVERSATION CONTEXT ===
        if current_conversation and "conversation" in current_conversation:
            conv = current_conversation["conversation"]
            
            # Create a focused current conversation document
            current_messages = []
            for msg in conv["messages"]:
                timestamp = msg['timestamp'].split('T')[0] if 'T' in msg['timestamp'] else msg['timestamp']
                current_messages.append(f"{msg['type']}: {msg['content']} ({timestamp})")
            
            current_conv_text = (
                f"CURRENT CONVERSATION SESSION: {conv['sessionId']}\n"
                f"Title: {conv['title']}\n"
                f"Total messages: {conv['totalMessages']}\n"
                f"Recent conversation:\n" + "\n".join(current_messages)
            )
            
            documents.append(Document(
                page_content=current_conv_text,
                metadata={
                    "source": "current_conversation", 
                    "type": "active_session", 
                    "user_email": user_info['email'],
                    "session_id": conv['sessionId']
                }
            ))

        # === HISTORICAL CONVERSATIONS SUMMARY ===
        if all_conversations and "conversations" in all_conversations:
            historical_convs = []
            current_session = current_conversation["conversation"]["sessionId"] if current_conversation else None
            
            for conv in all_conversations["conversations"]:
                # Skip current session to avoid duplication
                if current_session and conv["sessionId"] == current_session:
                    continue
                    
                conv_summary = (
                    f"Session {conv['sessionId']}: {conv['title']} "
                    f"({conv['totalMessages']} messages, last active: {conv['lastActivity'].split('T')[0]})"
                )
                historical_convs.append(conv_summary)
            
            if historical_convs:
                historical_text = f"PREVIOUS CONVERSATIONS for {user_info['email']}:\n" + "\n".join(historical_convs)
                documents.append(Document(
                    page_content=historical_text,
                    metadata={"source": "conversation_history", "type": "historical", "user_email": user_info['email']}
                ))

        return documents
    
    def setup_semantic_memory(self, session_id: str) -> ConversationSummaryBufferMemory:
        """Sets up semantic memory for conversation context."""
        if session_id not in self.memories:
            self.memories[session_id] = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=1000,
                return_messages=True,
                memory_key="chat_history"
            )
        return self.memories[session_id]
    
    def setup_retriever_for_session(self, session_id: str, user_email: str, force_refresh: bool = False) -> bool:
        """Sets up or updates the retriever for a specific session with semantic memory."""
        try:
            # Check if we need to update
            if (force_refresh or 
                session_id not in self.current_session_data or 
                self.current_session_data[session_id].get("user_email") != user_email):
                
                print(f"Setting up semantic retriever for session: {session_id}")
                
                # Fetch user data
                user_data = self.fetch_user_data(user_email)
                if not user_data:
                    print(f"Failed to fetch user data for {user_email}")
                    return False
                
                # Fetch current conversation and all conversations
                current_conversation = self.fetch_current_conversation(session_id, user_email)
                all_conversations = self.fetch_all_conversations(user_email)
                
                # Create semantic documents
                documents = self.create_semantic_documents(user_data, all_conversations, current_conversation)
                
                # Create vector store with better collection naming
                collection_name = f"user_{user_email.replace('@', '_').replace('.', '_')}_session_{session_id}".replace("-", "_")
                
                vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=collection_name
                )
                
                # Store vector store and create retriever with better search parameters
                self.vector_stores[session_id] = vector_store
                self.retrievers[session_id] = vector_store.as_retriever(
                    search_type="mmr",  # Maximum Marginal Relevance for diversity
                    search_kwargs={"k": 6, "lambda_mult": 0.7}
                )
                
                # Setup semantic memory
                memory = self.setup_semantic_memory(session_id)
                
                # Load current conversation into memory if it exists
                if current_conversation and "conversation" in current_conversation:
                    conv = current_conversation["conversation"]
                    for msg in conv["messages"]:
                        if msg["type"] == "user":
                            memory.chat_memory.add_user_message(msg["content"])
                        elif msg["type"] == "bot":
                            memory.chat_memory.add_ai_message(msg["content"])
                
                # Create conversational retrieval chain with memory
                system_prompt = """You are a fitness and wellness assistant with access to the user's complete profile and conversation history.

RESPONSE GUIDELINES:
1. Keep answers CONCISE (1-4 sentences) and DIRECTLY relevant
2. Use SPECIFIC data from the user's profile, habits, schedules, and conversation history
3. Reference exact numbers, dates, streaks, and progress when available
4. If asked about progress, compare current vs previous data
5. For scheduling questions, check actual calendar events
6. If information is missing, say "I don't have that information" 

CONTEXT PRIORITY:
- Current conversation > Recent habits/schedules > User profile > Historical conversations
- Always prioritize the most recent and relevant information

Provide personalized, actionable responses based on the user's specific data."""

                self.qa_chains[session_id] = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.retrievers[session_id],
                    memory=memory,
                    return_source_documents=True,
                    verbose=True,
                    condense_question_prompt=ChatPromptTemplate.from_template(
                        "Given the following conversation and a follow up question, "
                        "rephrase the follow up question to be a standalone question.\n"
                        "Chat History: {chat_history}\n"
                        "Follow Up Input: {question}\n"
                        "Standalone question:"
                    ),
                    combine_docs_chain_kwargs={
                        "prompt": ChatPromptTemplate.from_messages([
                            SystemMessagePromptTemplate.from_template(system_prompt),
                            HumanMessagePromptTemplate.from_template(
                                "Context: {context}\n"
                                "Chat History: {chat_history}\n"
                                "Human: {question}\n"
                                "Assistant:"
                            )
                        ])
                    }
                )
                
                # Cache session data
                self.current_session_data[session_id] = {
                    "user_email": user_email,
                    "last_updated": datetime.now().isoformat()
                }
                
                print(f"Successfully set up semantic retriever for session {session_id}")
                return True
            else:
                print(f"Using existing retriever for session {session_id}")
                return True
                
        except Exception as e:
            print(f"Error setting up retriever for session {session_id}: {e}")
            return False
    
    def ask(self, query: str, session_id: str, user_email: str = None) -> str:
        """
        Main method with semantic memory and current conversation awareness.
        """
        try:
            # Extract user email from existing session data if not provided
            if not user_email and session_id in self.current_session_data:
                user_email = self.current_session_data[session_id].get("user_email")
            
            if not user_email:
                return "Error: User email is required for new sessions."
            
            # Setup or update retriever for this session
            if not self.setup_retriever_for_session(session_id, user_email):
                return "Error: Failed to set up semantic knowledge retriever."
            
            # Get conversational chain for this session
            qa_chain = self.qa_chains.get(session_id)
            if not qa_chain:
                return "Error: Conversational chain not found for this session."
            
            # Get response from the AI with memory
            result = qa_chain({"question": query})
            
            return result["answer"]
            
        except Exception as e:
            print(f"Error in ask method: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def refresh_session_data(self, session_id: str, user_email: str) -> bool:
        """Manually refresh all data for a session including current conversation."""
        try:
            # Clear existing data
            self.cleanup_session(session_id)
            
            # Force setup with refresh
            return self.setup_retriever_for_session(session_id, user_email, force_refresh=True)
            
        except Exception as e:
            print(f"Error refreshing session {session_id}: {e}")
            return False
    
    def get_conversation_summary(self, session_id: str) -> str:
        """Get summary of current conversation from memory."""
        if session_id in self.memories:
            return self.memories[session_id].predict_new_summary(
                self.memories[session_id].chat_memory.messages,
                ""
            )
        return "No conversation history available."
    
    def cleanup_session(self, session_id: str):
        """Clean up resources for a specific session."""
        resources = [
            (self.current_session_data, "session data"),
            (self.vector_stores, "vector store"),
            (self.retrievers, "retriever"),
            (self.qa_chains, "QA chain"),
            (self.memories, "memory")
        ]
        
        for resource_dict, resource_name in resources:
            if session_id in resource_dict:
                del resource_dict[session_id]
                
        print(f"Cleaned up all resources for session {session_id}")