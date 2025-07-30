import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Configuration management for the travel assistant."""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "mock-key-for-demo")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4.1")
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1000"))
        self.data_dir = Path("data")
        self.flights_file = self.data_dir / "flights.json"
        self.knowledge_file = self.data_dir / "visa_rules.md"
        self.embedding_model = "text-embedding-ada-002"
        self.chunk_size = 1000
        self.chunk_overlap = 200

class FlightQuery(BaseModel):
    origin: str = Field(description="Departure city")
    destination: str = Field(description="Destination city")
    departure_date: str = Field(description="Departure date (YYYY-MM-DD)")
    return_date: Optional[str] = Field(default=None, description="Return date (YYYY-MM-DD)")
    preferred_airlines: List[str] = Field(default_factory=list, description="Preferred airlines")
    preferred_alliances: List[str] = Field(default_factory=list, description="Preferred alliances")
    max_price: Optional[float] = Field(default=None, description="Maximum price in USD")
    avoid_overnight_layovers: bool = Field(default=False, description="Avoid overnight layovers")
    refundable_only: bool = Field(default=False, description="Refundable tickets only")

class FlightResult(BaseModel):
    airline: str
    alliance: str
    from_city: str
    to_city: str
    departure_date: str
    return_date: str
    layovers: List[str]
    price_usd: float
    refundable: bool
    total_duration: str
    score: float

class ConversationState(BaseModel):
    """Conversation state management."""
    messages: List[Dict] = Field(default_factory=list)
    current_query: Optional[FlightQuery] = None
    flight_results: List[FlightResult] = Field(default_factory=list)
    knowledge_results: List[Dict] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)

class FlightDataManager:
    """Manages flight data operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.flights = self._load_flights()
    
    def _load_flights(self) -> List[Dict]:
        try:
            with open(self.config.flights_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Flights file not found")
            return
    
    
    def search_flights(self, query: FlightQuery) -> List[FlightResult]:
        results = []
        for flight in self.flights:
            if (flight["from"].lower() != query.origin.lower() or 
                flight["to"].lower() != query.destination.lower()):
                continue
            if flight["departure_date"] != query.departure_date:
                continue
            if query.return_date and flight["return_date"] != query.return_date:
                continue
            if query.preferred_alliances and flight["alliance"] not in query.preferred_alliances:
                continue
            if query.preferred_airlines and flight["airline"] not in query.preferred_airlines:
                continue
            if query.max_price and flight["price_usd"] > query.max_price:
                continue
            if query.refundable_only and not flight["refundable"]:
                continue
            if query.avoid_overnight_layovers and len(flight["layovers"]) > 0:
                pass
            score = self._calculate_relevance_score(flight, query)
            results.append(FlightResult(
                airline=flight["airline"],
                alliance=flight["alliance"],
                from_city=flight["from"],
                to_city=flight["to"],
                departure_date=flight["departure_date"],
                return_date=flight["return_date"],
                layovers=flight["layovers"],
                price_usd=flight["price_usd"],
                refundable=flight["refundable"],
                total_duration=flight["total_duration"],
                score=score
            ))
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def _calculate_relevance_score(self, flight: Dict, query: FlightQuery) -> float:
        score = 1.0
        if not flight["layovers"]:
            score += 0.3
        if query.preferred_alliances and flight["alliance"] in query.preferred_alliances:
            score += 0.2
        if query.avoid_overnight_layovers and len(flight["layovers"]) > 0:
            score -= 0.2
        if query.max_price:
            price_ratio = flight["price_usd"] / query.max_price
            score += (1 - price_ratio) * 0.3
        if flight["refundable"]:
            score += 0.1
        return score

class MockEmbeddings:    
    def embed_documents(self, texts):
        import numpy as np
        return [np.random.rand(1536).tolist() for _ in texts]
    
    def embed_query(self, text):
        import numpy as np
        return np.random.rand(1536).tolist()
    
    def __call__(self, text):
        return self.embed_query(text)

class KnowledgeBaseManager:
    
    def __init__(self, config: Config):
        self.config = config
        self.vector_store = None
        if self.config.openai_api_key == "mock-key-for-demo":
            self.embeddings = MockEmbeddings()
        else:
            self.embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                openai_api_key=self.config.openai_api_key
            )
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        try:
            knowledge_text = self._load_knowledge_base()
            texts = self._chunk_text(knowledge_text)
            if texts:
                self.vector_store = FAISS.from_texts(
                    texts, 
                    self.embeddings,
                    metadatas=[{"source": "knowledge_base"} for _ in texts]
                )
                logger.info("Vector store initialized successfully")
            else:
                logger.warning("No knowledge base content found")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
    
    def _load_knowledge_base(self) -> str:
        try:
            with open(self.config.knowledge_file, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("Knowledge base file not found, using sample data")
            return 
    
    
    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_length = 0
        for line in lines:
            if current_length + len(line) > self.config.chunk_size:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_length = len(line)
            else:
                current_chunk.append(line)
                current_length += len(line)
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        return chunks
    
    def search_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.vector_store:
            return []
        try:
            docs = self.vector_store.similarity_search(query, k=top_k)
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []

class MockLLM:
    """Mock LLM for testing without API key."""
    
    def invoke(self, input_data):
        if "user_query" in input_data:
            query_text = input_data["user_query"]
            if "tokyo" in query_text.lower() and "august" in query_text.lower():
                return FlightQuery(
                    origin="Dubai",
                    destination="Tokyo",
                    departure_date="2024-08-15",
                    return_date="2024-08-30",
                    preferred_alliances=["Star Alliance"] if "star alliance" in query_text.lower() else [],
                    avoid_overnight_layovers="overnight" in query_text.lower()
                )
            else:
                return FlightQuery(
                    origin="Dubai",
                    destination="Tokyo",
                    departure_date="2024-08-15",
                    return_date="2024-08-30"
                )
        elif "flight_results" in input_data:
            return type('MockResponse', (), {'content': 'Here are your flight options:\n\n1. Turkish Airlines (Star Alliance) - Dubai to Tokyo\n   Departure: 2024-08-15, Return: 2024-08-30\n   Price: $950, Duration: 18h 30m\n   Layovers: Istanbul\n   Refundable: Yes\n\n2. Emirates - Dubai to Tokyo\n   Departure: 2024-08-15, Return: 2024-08-30\n   Price: $1200, Duration: 9h 45m\n   Layovers: Direct\n   Refundable: Yes'})()
        elif "question" in input_data:
            return type('MockResponse', (), {'content': 'Based on the knowledge base, UAE passport holders can enter Japan visa-free for up to 30 days for tourism. Passport must be valid for at least 6 months beyond the intended stay.'})()
        else:
            return type('MockResponse', (), {'content': 'I\'m here to help you with your travel planning! How can I assist you today?'})()

class PromptManager:
    """Manages advanced prompt engineering."""
    
    def __init__(self):
        if os.getenv("OPENAI_API_KEY", "mock-key") == "mock-key":
            self.llm = MockLLM()
        else:
            self.llm = ChatOpenAI(
                model="gpt-4.1",
                temperature=0.1,
                openai_api_key=os.getenv("OPENAI_API_KEY", "mock-key")
            )
        self.query_parser = JsonOutputParser(pydantic_object=FlightQuery)
    
    def get_query_understanding_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert travel assistant specializing in flight queries. 
            Your task is to extract structured information from natural language travel requests.
            
            Extract the following information:
            - Origin and destination cities
            - Travel dates (departure and return)
            - Preferred airlines or alliances
            - Budget constraints
            - Special preferences (refundable, layover preferences, etc.)
            
            Return the information in JSON format matching the FlightQuery schema.
            If information is missing or unclear, make reasonable assumptions or mark as null."""),
            ("human", "{user_query}")
        ])
    
    def get_flight_search_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a helpful travel assistant presenting flight search results.
            Present the results in a clear, informative manner highlighting key features.
            Include pricing, duration, layovers, and any special features.
            Be conversational and helpful in your response."""),
            ("human", "Flight results: {flight_results}\nUser query: {user_query}")
        ])
    
    def get_knowledge_response_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a travel expert answering questions about visas, policies, and travel information.
            Use the provided knowledge base to answer questions accurately and comprehensively.
            If the knowledge base doesn't contain the answer, say so clearly.
            Be helpful and provide additional context when relevant."""),
            ("human", "Question: {question}\nKnowledge base information: {knowledge_info}")
        ])
    
    def get_conversation_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a friendly and knowledgeable travel assistant for Kavak.
            You help users plan international travel by finding flights, providing visa information, 
            and answering travel-related questions.
            
            Be conversational, helpful, and professional. Always provide accurate information
            and ask for clarification when needed."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_message}")
        ])

class TravelAssistantAgent:
    """Main travel assistant agent using LangGraph."""
    
    def __init__(self, config: Config):
        self.config = config
        self.flight_manager = FlightDataManager(config)
        self.knowledge_manager = KnowledgeBaseManager(config)
        self.prompt_manager = PromptManager()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(ConversationState)
        workflow.add_node("understand_query", self._understand_query)
        workflow.add_node("search_flights", self._search_flights)
        workflow.add_node("search_knowledge", self._search_knowledge)
        workflow.add_node("generate_response", self._generate_response)
        workflow.set_entry_point("understand_query")
        workflow.add_edge("understand_query", "search_flights")
        workflow.add_edge("search_flights", "search_knowledge")
        workflow.add_edge("search_knowledge", "generate_response")
        workflow.add_edge("generate_response", END)
        return workflow.compile()
    
    def _understand_query(self, state: ConversationState) -> ConversationState:
        if not state.messages:
            return state
        last_message = state.messages[-1]["content"]
        try:
            if isinstance(self.prompt_manager.llm, MockLLM):
                query_text = last_message.lower()
                flight_keywords = ["find", "search", "flight", "trip", "round-trip", "one-way", "to", "from", "departure", "return"]
                is_flight_query = any(keyword in query_text for keyword in flight_keywords)
                if is_flight_query:
                    origin = "Dubai"
                    destination = "Tokyo" if "tokyo" in query_text else "London"
                    departure_date = "2024-08-15" if "august" in query_text else "2024-09-10"
                    return_date = "2024-08-30" if "august" in query_text else "2024-09-25"
                    preferred_alliances = []
                    if "star alliance" in query_text:
                        preferred_alliances.append("Star Alliance")
                    elif "oneworld" in query_text:
                        preferred_alliances.append("Oneworld")
                    avoid_overnight_layovers = "overnight" in query_text and "avoid" in query_text
                    query = FlightQuery(
                        origin=origin,
                        destination=destination,
                        departure_date=departure_date,
                        return_date=return_date,
                        preferred_alliances=preferred_alliances,
                        avoid_overnight_layovers=avoid_overnight_layovers
                    )
                else:
                    query = None
            else:
                chain = self.prompt_manager.get_query_understanding_prompt() | self.prompt_manager.llm | self.prompt_manager.query_parser
                query = chain.invoke({"user_query": last_message})
            state.current_query = query
            logger.info(f"Parsed query: {query}")
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
        return state
    
    def _search_flights(self, state: ConversationState) -> ConversationState:
        if not state.current_query:
            return state
        try:
            results = self.flight_manager.search_flights(state.current_query)
            state.flight_results = results
            logger.info(f"Found {len(results)} flight results")
        except Exception as e:
            logger.error(f"Error searching flights: {e}")
        return state
    
    def _search_knowledge(self, state: ConversationState) -> ConversationState:
        if not state.messages:
            return state
        last_message = state.messages[-1]["content"]
        try:
            results = self.knowledge_manager.search_knowledge(last_message)
            state.knowledge_results = results
            logger.info(f"Found {len(results)} knowledge results")
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
        return state
    
    def _generate_response(self, state: ConversationState) -> ConversationState:
        if not state.messages:
            return state
        last_message = state.messages[-1]["content"]
        try:
            if state.flight_results:
                response = self._generate_flight_response(state, last_message)
            elif state.knowledge_results:
                response = self._generate_knowledge_response(state, last_message)
            else:
                response = self._generate_general_response(state, last_message)
            state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state.messages.append({
                "role": "assistant",
                "content": "I apologize, but I encountered an error processing your request. Please try again.",
                "timestamp": datetime.now().isoformat()
            })
        return state
    
    def _generate_flight_response(self, state: ConversationState, user_query: str) -> str:
        flight_info = []
        for flight in state.flight_results[:3]:
            flight_info.append(f"""
            Airline: {flight.airline} ({flight.alliance})
            Route: {flight.from_city} → {flight.to_city}
            Dates: {flight.departure_date} - {flight.return_date}
            Price: ${flight.price_usd}
            Duration: {flight.total_duration}
            Layovers: {', '.join(flight.layovers) if flight.layovers else 'Direct'}
            Refundable: {'Yes' if flight.refundable else 'No'}
            """)
        if isinstance(self.prompt_manager.llm, MockLLM):
            response = "Here are your flight options:\n\n"
            for i, flight in enumerate(state.flight_results[:3], 1):
                response += f"{i}. {flight.airline} ({flight.alliance}) - {flight.from_city} to {flight.to_city}\n"
                response += f"   Departure: {flight.departure_date}, Return: {flight.return_date}\n"
                response += f"   Price: ${flight.price_usd}, Duration: {flight.total_duration}\n"
                response += f"   Layovers: {', '.join(flight.layovers) if flight.layovers else 'Direct'}\n"
                response += f"   Refundable: {'Yes' if flight.refundable else 'No'}\n\n"
            return response
        else:
            chain = self.prompt_manager.get_flight_search_prompt() | self.prompt_manager.llm
            result = chain.invoke({
                "flight_results": "\n".join(flight_info),
                "user_query": user_query
            })
            return result.content
    
    def _generate_knowledge_response(self, state: ConversationState, user_query: str) -> str:
        knowledge_info = "\n".join([result["content"] for result in state.knowledge_results])
        if isinstance(self.prompt_manager.llm, MockLLM):
            return f"Based on the knowledge base, here's what I found:\n\n{knowledge_info[:500]}..."
        else:
            chain = self.prompt_manager.get_knowledge_response_prompt() | self.prompt_manager.llm
            result = chain.invoke({
                "question": user_query,
                "knowledge_info": knowledge_info
            })
            return result.content
    
    def _generate_general_response(self, state: ConversationState, user_query: str) -> str:
        chat_history = []
        for msg in state.messages[:-1]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))
        if isinstance(self.prompt_manager.llm, MockLLM):
            return "I'm here to help you with your travel planning! How can I assist you today?"
        else:
            chain = self.prompt_manager.get_conversation_prompt() | self.prompt_manager.llm
            result = chain.invoke({
                "chat_history": chat_history,
                "user_message": user_query
            })
            return result.content
    
    def process_message(self, message: str) -> str:
        try:
            if not hasattr(self, '_state'):
                self._state = ConversationState()
            self._state.messages.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            final_state = self.graph.invoke(self._state)
            if hasattr(final_state, 'messages'):
                messages = final_state.messages
            else:
                messages = final_state.get('messages', [])
            for msg in reversed(messages):
                if msg["role"] == "assistant":
                    return msg["content"]
            return "I apologize, but I couldn't generate a response. Please try again."
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I encountered an error: {str(e)}"

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Kavak Travel Assistant")
    parser.add_argument("--web", action="store_true", help="Launch web interface")
    parser.add_argument("--query", type=str, help="Process a single query")
    args = parser.parse_args()
    if args.web:
        import subprocess
        import sys
        print("Launching Streamlit web interface...")
        print("The web interface will be available at http://localhost:8501")
        print("Press Ctrl+C to stop the server")
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], check=True)
        except KeyboardInterrupt:
            print("\nWeb interface stopped.")
        except subprocess.CalledProcessError as e:
            print(f"Error launching Streamlit: {e}")
            print("You can also run: streamlit run streamlit_app.py")
    elif args.query:
        config = Config()
        assistant = TravelAssistantAgent(config)
        response = assistant.process_message(args.query)
        print(f"Assistant: {response}")
    else:
        print("Kavak Travel Assistant")
        print("Type 'quit' to exit")
        print("-" * 50)
        config = Config()
        assistant = TravelAssistantAgent(config)
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye! Safe travels!")
                    break
                if user_input:
                    response = assistant.process_message(user_input)
                    print(f"Assistant: {response}")
                    print()
            except KeyboardInterrupt:
                print("\nGoodbye! Safe travels!")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main() 
