"""
================================================================================
🤖 MULTI-AGENT DEMO: Smart Research Assistant
================================================================================

This demo showcases multiple AI agents working together to answer complex queries.
Perfect for demonstrating RAG, Agents, and GenAI concepts visually.

ARCHITECTURE:
                    ┌─────────────────┐
                    │   SUPERVISOR    │  ← Orchestrates & routes queries
                    │     AGENT       │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │  RAG AGENT  │   │  WEB AGENT  │   │ MATH AGENT  │
    │             │   │             │   │             │
    │ Searches    │   │ Searches    │   │ Performs    │
    │ local       │   │ the         │   │ calculations│
    │ knowledge   │   │ internet    │   │ & analysis  │
    │ base        │   │             │   │             │
    └─────────────┘   └─────────────┘   └─────────────┘

VISUAL OUTPUT:
- Each agent's actions are displayed in colored boxes
- State transitions are clearly shown
- Final response is synthesized from all agents

Author: AI Demo
Date: April 2026
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import sys
from typing import TypedDict, Annotated, Sequence, Literal
from datetime import datetime

# ==============================================================================
# SSL CERTIFICATE FIX (for macOS / Corporate Networks)
# ==============================================================================
# This fixes the "SSL: CERTIFICATE_VERIFY_FAILED" error on macOS
# or when behind corporate proxies/firewalls
# NOTE: verify=False is acceptable for demos but not for production!

import ssl
import httpx

# Create a custom httpx client that bypasses SSL verification
# This is necessary when behind corporate proxies or on macOS with cert issues
custom_http_client = httpx.Client(verify=False)

# Suppress the InsecureRequestWarning for cleaner demo output
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Rich library for beautiful console output (visual demo!)
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box

# LangChain imports for LLM and tools
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# LangGraph for multi-agent orchestration
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

# Vector store for RAG
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# Web search tool (free, no API key needed!)
from langchain_community.tools import DuckDuckGoSearchRun

# For mathematical operations
import operator
import re

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Initialize Rich console for beautiful output
console = Console()

# Color scheme for different agents (for visual distinction)
AGENT_COLORS = {
    "supervisor": "bold magenta",
    "rag": "bold cyan",
    "web": "bold green", 
    "math": "bold yellow",
    "final": "bold white on blue"
}

# ==============================================================================
# STEP 1: INITIALIZE THE LLM (OpenAI GPT-4)
# ==============================================================================

def initialize_llm():
    """
    Initialize the OpenAI Language Model.
    
    We use GPT-4o-mini for cost efficiency, but you can change to 'gpt-4o' 
    for better reasoning (especially for complex routing decisions).
    
    Returns:
        ChatOpenAI: Configured LLM instance
    """
    console.print("\n[bold blue]🔧 Initializing OpenAI LLM...[/bold blue]")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[bold red]❌ ERROR: OPENAI_API_KEY not found![/bold red]")
        console.print("Please set it: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Cost-effective, fast model
        temperature=0,         # Low temperature for consistent outputs
        streaming=True,        # Enable streaming for real-time output
        http_client=custom_http_client  # Use custom client with SSL fix
    )
    
    console.print("[green]✓ LLM initialized successfully![/green]")
    return llm

# ==============================================================================
# STEP 2: BUILD THE RAG KNOWLEDGE BASE
# ==============================================================================

def build_knowledge_base():
    """
    Build a vector store from local documents for RAG.
    
    RAG (Retrieval Augmented Generation) Flow:
    1. Load documents from knowledge_base/ folder
    2. Split them into chunks
    3. Create embeddings using OpenAI
    4. Store in ChromaDB vector database
    
    Returns:
        Chroma: Vector store for similarity search
    """
    console.print("\n[bold blue]📚 Building RAG Knowledge Base...[/bold blue]")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(script_dir, "knowledge_base")
    
    # Check if knowledge base folder exists
    if not os.path.exists(kb_path):
        console.print(f"[yellow]⚠ Knowledge base folder not found at {kb_path}[/yellow]")
        console.print("[yellow]Creating sample knowledge base...[/yellow]")
        os.makedirs(kb_path, exist_ok=True)
        return None
    
    # Load all .txt files from knowledge_base folder
    try:
        loader = DirectoryLoader(
            kb_path,
            glob="**/*.txt",           # Load all .txt files
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = loader.load()
        
        if not documents:
            console.print("[yellow]⚠ No documents found in knowledge base[/yellow]")
            return None
            
        console.print(f"[cyan]  → Loaded {len(documents)} documents[/cyan]")
        
        # Split documents into smaller chunks for better retrieval
        # Chunk size of 500 chars with 50 char overlap for context continuity
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        console.print(f"[cyan]  → Created {len(splits)} text chunks[/cyan]")
        
        # Create embeddings and vector store
        # Embeddings convert text to numerical vectors for similarity search
        embeddings = OpenAIEmbeddings(http_client=custom_http_client)
        
        # Use a local persistent directory for ChromaDB
        persist_directory = os.path.join(script_dir, ".chroma_db")
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name="demo_knowledge_base",
            persist_directory=persist_directory  # Store locally, not client-server
        )
        
        console.print("[green]✓ Knowledge base ready![/green]")
        return vectorstore
        
    except Exception as e:
        console.print(f"[red]❌ Error building knowledge base: {e}[/red]")
        return None

# ==============================================================================
# STEP 3: DEFINE THE GRAPH STATE
# ==============================================================================

# This TypedDict defines what information flows between agents
# Think of it as the "shared memory" or "blackboard" that all agents can read/write

class AgentState(TypedDict):
    """
    State that flows through the multi-agent graph.
    
    Attributes:
        messages: List of all messages in the conversation
        query: The original user query
        rag_response: Response from the RAG agent
        web_response: Response from the Web Search agent  
        math_response: Response from the Math agent
        next_agent: Which agent should process next
        final_response: The synthesized final answer
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    rag_response: str
    web_response: str
    math_response: str
    next_agent: str
    final_response: str

# ==============================================================================
# STEP 4: CREATE INDIVIDUAL AGENTS
# ==============================================================================

def create_rag_agent(llm, vectorstore):
    """
    Create the RAG (Retrieval Augmented Generation) Agent.
    
    This agent:
    1. Receives a query
    2. Searches the local knowledge base for relevant info
    3. Uses the retrieved context to generate an answer
    
    Args:
        llm: The language model
        vectorstore: The vector database with embedded documents
        
    Returns:
        function: Agent function that processes queries
    """
    
    def rag_agent(state: AgentState) -> dict:
        """RAG Agent: Searches local knowledge base"""
        
        query = state["query"]
        
        # Visual feedback - show agent is working
        console.print(Panel(
            f"[cyan]🔍 Searching knowledge base for:[/cyan]\n\"{query}\"",
            title="📚 RAG AGENT",
            border_style=AGENT_COLORS["rag"],
            box=box.DOUBLE
        ))
        
        if vectorstore is None:
            response = "Knowledge base is not available. Cannot search local documents."
            console.print(f"[yellow]  ⚠ {response}[/yellow]")
            return {"rag_response": response}
        
        # Perform similarity search in vector store
        # k=3 means retrieve top 3 most relevant chunks
        docs = vectorstore.similarity_search(query, k=3)
        
        if not docs:
            response = "No relevant information found in the knowledge base."
            console.print(f"[yellow]  ⚠ {response}[/yellow]")
            return {"rag_response": response}
        
        # Show retrieved documents
        console.print(f"[cyan]  → Found {len(docs)} relevant chunks[/cyan]")
        
        # Combine retrieved chunks into context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt with retrieved context
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on 
            the provided context from our knowledge base. Be concise and accurate.
            If the context doesn't contain relevant information, say so.
            
            CONTEXT FROM KNOWLEDGE BASE:
            {context}"""),
            ("human", "{query}")
        ])
        
        # Generate response using LLM with context
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"context": context, "query": query})
        
        # Display the response
        console.print(Panel(
            Markdown(response),
            title="📚 RAG Agent Response",
            border_style="cyan"
        ))
        
        return {"rag_response": response}
    
    return rag_agent


def create_web_agent(llm):
    """
    Create the Web Search Agent.
    
    This agent:
    1. Receives a query
    2. Searches the internet using DuckDuckGo
    3. Synthesizes findings into a response
    
    Uses DuckDuckGo because it's FREE and doesn't need an API key!
    
    Args:
        llm: The language model
        
    Returns:
        function: Agent function that processes queries
    """
    
    # Initialize DuckDuckGo search tool
    search_tool = DuckDuckGoSearchRun()
    
    def web_agent(state: AgentState) -> dict:
        """Web Agent: Searches the internet"""
        
        query = state["query"]
        
        # Visual feedback
        console.print(Panel(
            f"[green]🌐 Searching the web for:[/green]\n\"{query}\"",
            title="🌍 WEB AGENT",
            border_style=AGENT_COLORS["web"],
            box=box.DOUBLE
        ))
        
        try:
            # Perform web search
            search_results = search_tool.invoke(query)
            console.print(f"[green]  → Retrieved web results[/green]")
            
            # Show raw search results (truncated for display)
            if len(search_results) > 300:
                display_results = search_results[:300] + "..."
            else:
                display_results = search_results
            console.print(f"[dim]  Raw results: {display_results}[/dim]")
            
            # Use LLM to synthesize search results
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a web research assistant. Synthesize the following 
                search results into a clear, concise answer. Include relevant facts and 
                cite sources when possible.
                
                SEARCH RESULTS:
                {search_results}"""),
                ("human", "{query}")
            ])
            
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({"search_results": search_results, "query": query})
            
        except Exception as e:
            response = f"Web search encountered an error: {str(e)}"
            console.print(f"[red]  ❌ {response}[/red]")
            return {"web_response": response}
        
        # Display the response
        console.print(Panel(
            Markdown(response),
            title="🌍 Web Agent Response",
            border_style="green"
        ))
        
        return {"web_response": response}
    
    return web_agent


def create_math_agent(llm):
    """
    Create the Math/Calculator Agent.
    
    This agent:
    1. Receives a query involving calculations
    2. Extracts numbers and operations
    3. Performs calculations or analysis
    
    Args:
        llm: The language model
        
    Returns:
        function: Agent function that processes queries
    """
    
    def math_agent(state: AgentState) -> dict:
        """Math Agent: Performs calculations and analysis"""
        
        query = state["query"]
        
        # Visual feedback
        console.print(Panel(
            f"[yellow]🔢 Analyzing mathematical query:[/yellow]\n\"{query}\"",
            title="🧮 MATH AGENT",
            border_style=AGENT_COLORS["math"],
            box=box.DOUBLE
        ))
        
        # Use LLM for mathematical reasoning
        # GPT-4 is quite good at math, but for production you might want
        # to use specialized tools like Wolfram Alpha
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a mathematical assistant. Solve the given problem 
            step by step. Show your work clearly. Handle:
            - Arithmetic calculations
            - Percentages and ratios
            - Unit conversions
            - Statistical analysis
            - Basic algebra
            
            Always verify your calculations and explain each step."""),
            ("human", "{query}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"query": query})
        
        # Display the response
        console.print(Panel(
            Markdown(response),
            title="🧮 Math Agent Response",
            border_style="yellow"
        ))
        
        return {"math_response": response}
    
    return math_agent

# ==============================================================================
# STEP 5: CREATE THE SUPERVISOR AGENT
# ==============================================================================

def create_supervisor(llm):
    """
    Create the Supervisor Agent that orchestrates other agents.
    
    The supervisor:
    1. Analyzes the incoming query
    2. Decides which agent(s) should handle it
    3. Routes the query appropriately
    
    This is the "brain" that coordinates the multi-agent system.
    
    Args:
        llm: The language model
        
    Returns:
        function: Supervisor function that routes queries
    """
    
    def supervisor(state: AgentState) -> dict:
        """Supervisor: Routes queries to appropriate agents"""
        
        query = state["query"]
        
        # Visual feedback
        console.print(Panel(
            f"[magenta]🎯 Analyzing query to determine best agent(s):[/magenta]\n\"{query}\"",
            title="🎭 SUPERVISOR AGENT",
            border_style=AGENT_COLORS["supervisor"],
            box=box.DOUBLE
        ))
        
        # Prompt for supervisor to decide routing
        routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a supervisor that routes queries to specialized agents.
            
            Available agents:
            1. RAG - For questions about company policies, internal documents, product info
            2. WEB - For current events, real-time data, external information
            3. MATH - For calculations, statistics, numerical analysis
            4. ALL - For complex queries requiring multiple perspectives
            
            Analyze the query and respond with ONLY ONE of: RAG, WEB, MATH, or ALL
            
            Examples:
            - "What is our company vacation policy?" → RAG
            - "What is the current stock price of Apple?" → WEB
            - "Calculate 15% of 2500" → MATH
            - "Compare our product pricing with market trends" → ALL
            """),
            ("human", "{query}")
        ])
        
        chain = routing_prompt | llm | StrOutputParser()
        decision = chain.invoke({"query": query}).strip().upper()
        
        # Clean up decision (extract just the agent name)
        if "RAG" in decision:
            decision = "RAG"
        elif "WEB" in decision:
            decision = "WEB"
        elif "MATH" in decision:
            decision = "MATH"
        else:
            decision = "ALL"
        
        # Display the routing decision with cool visual
        console.print(f"\n[bold magenta]  📍 ROUTING DECISION: {decision}[/bold magenta]")
        
        if decision == "ALL":
            console.print("[dim]  → Will consult all agents for comprehensive answer[/dim]")
        else:
            console.print(f"[dim]  → Query will be handled by {decision} agent[/dim]")
        
        console.print()  # Add spacing
        
        return {"next_agent": decision}
    
    return supervisor

# ==============================================================================
# STEP 6: CREATE THE FINAL SYNTHESIZER
# ==============================================================================

def create_synthesizer(llm):
    """
    Create the Final Synthesizer that combines all agent responses.
    
    This agent:
    1. Collects responses from all agents that were consulted
    2. Synthesizes them into a coherent final answer
    3. Removes redundancy and highlights key points
    
    Args:
        llm: The language model
        
    Returns:
        function: Synthesizer function
    """
    
    def synthesizer(state: AgentState) -> dict:
        """Synthesize all agent responses into final answer"""
        
        # Visual feedback
        console.print(Panel(
            "[white]📝 Synthesizing responses from all agents...[/white]",
            title="🎯 FINAL SYNTHESIS",
            border_style=AGENT_COLORS["final"],
            box=box.DOUBLE
        ))
        
        # Collect all responses
        responses = []
        
        if state.get("rag_response"):
            responses.append(f"**From Knowledge Base (RAG):**\n{state['rag_response']}")
            
        if state.get("web_response"):
            responses.append(f"**From Web Search:**\n{state['web_response']}")
            
        if state.get("math_response"):
            responses.append(f"**From Mathematical Analysis:**\n{state['math_response']}")
        
        if not responses:
            final = "I couldn't gather information from any agent. Please try rephrasing your query."
        elif len(responses) == 1:
            # Only one agent responded, use that directly
            final = responses[0]
        else:
            # Multiple agents responded, synthesize
            combined = "\n\n".join(responses)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a synthesis assistant. Combine the following 
                agent responses into a single, coherent answer. Remove redundancy,
                highlight key points, and ensure the response flows naturally.
                
                AGENT RESPONSES:
                {responses}"""),
                ("human", "Please provide a synthesized response to: {query}")
            ])
            
            chain = prompt | llm | StrOutputParser()
            final = chain.invoke({"responses": combined, "query": state["query"]})
        
        return {"final_response": final}
    
    return synthesizer

# ==============================================================================
# STEP 7: BUILD THE MULTI-AGENT GRAPH
# ==============================================================================

def build_agent_graph(llm, vectorstore):
    """
    Build the LangGraph that orchestrates all agents.
    
    Graph Structure:
    
        START
          │
          ▼
      ┌─────────┐
      │Supervisor│ ──────────────────────────┐
      └────┬────┘                            │
           │ (routes to appropriate agent)   │
           ├──────────┬──────────┬───────────┤
           │          │          │           │
           ▼          ▼          ▼           ▼
        ┌─────┐   ┌─────┐   ┌─────┐      (ALL)
        │ RAG │   │ WEB │   │MATH │         │
        └──┬──┘   └──┬──┘   └──┬──┘         │
           │          │          │           │
           └──────────┴──────────┴───────────┘
                               │
                               ▼
                       ┌─────────────┐
                       │ Synthesizer │
                       └──────┬──────┘
                              │
                              ▼
                            END
    
    Args:
        llm: The language model
        vectorstore: The vector database
        
    Returns:
        CompiledGraph: The compiled multi-agent graph
    """
    
    console.print("\n[bold blue]🔨 Building Multi-Agent Graph...[/bold blue]")
    
    # Create all agents
    supervisor = create_supervisor(llm)
    rag_agent = create_rag_agent(llm, vectorstore)
    web_agent = create_web_agent(llm)
    math_agent = create_math_agent(llm)
    synthesizer = create_synthesizer(llm)
    
    # Initialize the StateGraph with our state schema
    workflow = StateGraph(AgentState)
    
    # -------------------------------------------------------------------------
    # Add nodes (each agent is a node in the graph)
    # -------------------------------------------------------------------------
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("web_agent", web_agent)
    workflow.add_node("math_agent", math_agent)
    workflow.add_node("synthesizer", synthesizer)
    
    # -------------------------------------------------------------------------
    # Define routing logic (conditional edges)
    # -------------------------------------------------------------------------
    
    def route_query(state: AgentState) -> str:
        """Determine next node based on supervisor's decision"""
        decision = state.get("next_agent", "ALL")
        
        if decision == "RAG":
            return "rag_only"
        elif decision == "WEB":
            return "web_only"
        elif decision == "MATH":
            return "math_only"
        else:  # ALL
            return "all_agents"
    
    # Add conditional edge from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_query,
        {
            "rag_only": "rag_agent",
            "web_only": "web_agent",
            "math_only": "math_agent",
            "all_agents": "rag_agent"  # Start with RAG when ALL
        }
    )
    
    # -------------------------------------------------------------------------
    # For "ALL" path: Chain agents together
    # -------------------------------------------------------------------------
    
    def check_if_all(state: AgentState) -> str:
        """Check if we need to continue to more agents"""
        if state.get("next_agent") == "ALL":
            if not state.get("web_response"):
                return "continue_to_web"
            elif not state.get("math_response"):
                return "continue_to_math"
        return "go_to_synthesizer"
    
    # RAG can either continue to web (if ALL) or go to synthesizer
    workflow.add_conditional_edges(
        "rag_agent",
        check_if_all,
        {
            "continue_to_web": "web_agent",
            "go_to_synthesizer": "synthesizer"
        }
    )
    
    # Web can either continue to math (if ALL) or go to synthesizer
    workflow.add_conditional_edges(
        "web_agent",
        check_if_all,
        {
            "continue_to_math": "math_agent",
            "go_to_synthesizer": "synthesizer"
        }
    )
    
    # Math always goes to synthesizer
    workflow.add_edge("math_agent", "synthesizer")
    
    # Synthesizer ends the graph
    workflow.add_edge("synthesizer", END)
    
    # -------------------------------------------------------------------------
    # Set entry point
    # -------------------------------------------------------------------------
    workflow.add_edge(START, "supervisor")
    
    # Compile the graph
    graph = workflow.compile()
    
    console.print("[green]✓ Multi-Agent Graph compiled successfully![/green]")
    
    return graph

# ==============================================================================
# STEP 8: VISUALIZATION HELPER
# ==============================================================================

def print_graph_visual():
    """Print a visual representation of the agent graph"""
    
    graph_visual = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                    MULTI-AGENT SYSTEM ARCHITECTURE                 ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║                         ┌──────────────┐                          ║
    ║                         │   👤 USER    │                          ║
    ║                         │    QUERY     │                          ║
    ║                         └──────┬───────┘                          ║
    ║                                │                                   ║
    ║                                ▼                                   ║
    ║                    ┌───────────────────────┐                      ║
    ║                    │   🎭 SUPERVISOR       │                      ║
    ║                    │   Routes queries to   │                      ║
    ║                    │   appropriate agents  │                      ║
    ║                    └───────────┬───────────┘                      ║
    ║                                │                                   ║
    ║            ┌───────────────────┼───────────────────┐              ║
    ║            │                   │                   │              ║
    ║            ▼                   ▼                   ▼              ║
    ║    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        ║
    ║    │  📚 RAG      │   │  🌍 WEB      │   │  🧮 MATH     │        ║
    ║    │  AGENT       │   │  AGENT       │   │  AGENT       │        ║
    ║    │              │   │              │   │              │        ║
    ║    │ Searches     │   │ Searches     │   │ Performs     │        ║
    ║    │ local docs   │   │ internet     │   │ calculations │        ║
    ║    └──────────────┘   └──────────────┘   └──────────────┘        ║
    ║            │                   │                   │              ║
    ║            └───────────────────┼───────────────────┘              ║
    ║                                │                                   ║
    ║                                ▼                                   ║
    ║                    ┌───────────────────────┐                      ║
    ║                    │   🎯 SYNTHESIZER      │                      ║
    ║                    │   Combines responses  │                      ║
    ║                    │   into final answer   │                      ║
    ║                    └───────────┬───────────┘                      ║
    ║                                │                                   ║
    ║                                ▼                                   ║
    ║                         ┌──────────────┐                          ║
    ║                         │  ✨ FINAL    │                          ║
    ║                         │   RESPONSE   │                          ║
    ║                         └──────────────┘                          ║
    ║                                                                    ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    console.print(graph_visual, style="cyan")

# ==============================================================================
# STEP 9: RUN THE MULTI-AGENT SYSTEM
# ==============================================================================

def run_query(graph, query: str):
    """
    Execute a query through the multi-agent system.
    
    Args:
        graph: The compiled LangGraph
        query: User's question
    """
    
    console.print("\n")
    console.print("═" * 70, style="bold blue")
    console.print(f"[bold blue]📨 NEW QUERY:[/bold blue] {query}")
    console.print("═" * 70, style="bold blue")
    console.print()
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "rag_response": "",
        "web_response": "",
        "math_response": "",
        "next_agent": "",
        "final_response": ""
    }
    
    # Run the graph
    try:
        # Invoke the graph with initial state
        result = graph.invoke(initial_state)
        
        # Display final response
        console.print("═" * 70, style="bold green")
        console.print(Panel(
            Markdown(result.get("final_response", "No response generated")),
            title="✨ FINAL RESPONSE",
            border_style="bold green",
            box=box.DOUBLE
        ))
        console.print("═" * 70, style="bold green")
        
        return result
        
    except Exception as e:
        console.print(f"[bold red]❌ Error running query: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return None

# ==============================================================================
# STEP 10: MAIN FUNCTION - DEMO ENTRY POINT
# ==============================================================================

def main():
    """
    Main function - Entry point for the demo
    
    This function:
    1. Displays the architecture
    2. Initializes all components
    3. Runs demo queries
    4. Provides an interactive mode
    """
    
    # Welcome banner
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]🤖 MULTI-AGENT DEMO: Smart Research Assistant[/bold cyan]\n\n"
        "This demo showcases multiple AI agents working together:\n"
        "• 📚 RAG Agent - Searches local knowledge base\n"
        "• 🌍 Web Agent - Searches the internet\n"
        "• 🧮 Math Agent - Performs calculations\n"
        "• 🎭 Supervisor - Routes queries intelligently",
        title="Welcome",
        border_style="bold blue"
    ))
    
    # Show architecture
    print_graph_visual()
    
    # Initialize components
    llm = initialize_llm()
    vectorstore = build_knowledge_base()
    graph = build_agent_graph(llm, vectorstore)
    
    # Demo queries to showcase different agent flows
    demo_queries = [
        # This will route to RAG agent
        "What are TechCorp's core values and vacation policy?",
        
        # This will route to WEB agent  
        "What is the latest news about artificial intelligence?",
        
        # This will route to MATH agent
        "Calculate the total revenue if we sell 150 units at $299 each with a 15% discount",
        
        # This will route to ALL agents
        "Compare our company's AI product features with current market trends and calculate potential market share"
    ]
    
    # Wait for user confirmation before starting demo queries
    console.print()
    console.input("[bold cyan]Press Enter to start the demo queries...[/bold cyan]")
    
    console.print("\n[bold yellow]🎬 RUNNING DEMO QUERIES[/bold yellow]\n")
    console.print("[dim]These queries demonstrate different routing paths:[/dim]\n")
    
    # Run demo queries
    for i, query in enumerate(demo_queries, 1):
        console.print(f"\n[bold]Demo Query {i}/{len(demo_queries)}[/bold]")
        
        result = run_query(graph, query)
        
        if i < len(demo_queries):
            console.print("[dim]─" * 70 + "[/dim]")
            console.input("[bold cyan]Press Enter to continue to next query...[/bold cyan]")
    
    # Interactive mode
    console.print("\n")
    console.print(Panel(
        "[bold green]🎮 INTERACTIVE MODE[/bold green]\n\n"
        "Now you can try your own queries!\n"
        "Type 'quit' or 'exit' to end the demo.",
        border_style="green"
    ))
    
    while True:
        console.print()
        query = console.input("[bold cyan]Your query: [/bold cyan]")
        
        if query.lower() in ['quit', 'exit', 'q']:
            console.print("\n[bold blue]👋 Thanks for watching the demo![/bold blue]\n")
            break
        
        if query.strip():
            run_query(graph, query)

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
