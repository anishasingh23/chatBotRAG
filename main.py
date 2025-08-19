# app.py
import streamlit as st
import os
from typing import List, Dict, Any
import uuid
from datetime import datetime
import io
import re
import base64
import tempfile
import json
import math

# Import document processing libraries
import pypdf
from docx import Document as DocxDocument
import pandas as pd
import plotly.graph_objects as go

# Import vector store
import chromadb
from chromadb.utils import embedding_functions

# Import Groq
import groq

# Import environment variables
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="DocuChat - Advanced RAG Chat System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = None
if "collection" not in st.session_state:
    st.session_state.collection = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = False
if "documents" not in st.session_state:
    st.session_state.documents = []
if "research_mode" not in st.session_state:
    st.session_state.research_mode = False
if "current_files" not in st.session_state:
    st.session_state.current_files = []
if "show_upload" not in st.session_state:
    st.session_state.show_upload = True
if "file_previews" not in st.session_state:
    st.session_state.file_previews = {}
if "document_summaries" not in st.session_state:
    st.session_state.document_summaries = {}
if "knowledge_graph" not in st.session_state:
    st.session_state.knowledge_graph = {"nodes": [], "edges": []}
if "selected_chat" not in st.session_state:
    st.session_state.selected_chat = None
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())
if "show_graph" not in st.session_state:
    st.session_state.show_graph = False

# Get API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Custom CSS for better styling with fixed colors
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem !important;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #1f77b4;
        margin-bottom: 2rem;
    }
    .stButton button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #135e96;
    }
    .process-btn {
        background-color: #28a745 !important;
    }
    .process-btn:hover {
        background-color: #1e7e34 !important;
    }
    .clear-btn {
        background-color: #dc3545 !important;
    }
    .clear-btn:hover {
        background-color: #bd2130 !important;
    }
    .chat-message-user {
        background-color: #e6f7ff;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
        color: #333 !important;
    }
    .chat-message-assistant {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #28a745;
        color: #333 !important;
    }
    .status-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #6c757d;
        margin-bottom: 1rem;
        color: #333 !important;
    }
    .preview-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        border: 1px solid #ddd;
    }
    .summary-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        border-left: 4px solid #28a745;
    }
    .chat-session {
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        border-radius: 5px;
        cursor: pointer;
    }
    .chat-session:hover {
        background-color: #f0f0f0;
    }
    .chat-session.active {
        background-color: #e6f7ff;
        border-left: 4px solid #1f77b4;
    }
    /* Ensure text is visible in all containers */
    .stMarkdown, .stText, .stCode {
        color: #333 !important;
    }
    /* Fix for chat input */
    .stChatInput {
        position: fixed;
        bottom: 2rem;
        width: 70%;
        z-index: 100;
    }
    .graph-controls {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to extract text from different file types
def extract_text_from_file(uploaded_file) -> str:
    """Extract text from various file types"""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    text = ""
    
    try:
        if file_extension == ".pdf":
            pdf_reader = pypdf.PdfReader(uploaded_file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"--- PAGE {page_num + 1} ---\n{page_text}\n\n"
        
        elif file_extension in [".docx", ".doc"]:
            doc = DocxDocument(io.BytesIO(uploaded_file.getvalue()))
            for para_num, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
        
        elif file_extension == ".txt":
            text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(uploaded_file)
            text = df.to_string()
        
        elif file_extension == ".csv":
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    text = df.to_string()
                    break
                except UnicodeDecodeError:
                    continue
            if not text:
                uploaded_file.seek(0)
                text = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return ""
        
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text if text else ""
    except Exception as e:
        st.error(f"Error extracting text from {uploaded_file.name}: {str(e)}")
        return ""

# Function to split text into chunks with better project detection
def split_text_into_chunks(text: str, chunk_size: int = 600, chunk_overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks with better boundary detection for resumes"""
    if not text:
        return []
    
    project_patterns = [
        r'(?i)(?:projects?|work experience|experience|portfolio)[:\s]*\n',
        r'(?i)\b(?:project|experience)\b.*\n',
        r'(?i)^[A-Z][A-Za-z\s]+:.*$',
    ]
    
    chunks = []
    
    for pattern in project_patterns:
        sections = re.split(pattern, text)
        if len(sections) > 1:
            for i, section in enumerate(sections):
                if i > 0 and section.strip():
                    if len(section) > chunk_size * 2:
                        sub_chunks = split_by_sentences(section, chunk_size, chunk_overlap)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(section.strip())
            if chunks:
                return chunks
    
    return split_by_sentences(text, chunk_size, chunk_overlap)

def split_by_sentences(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text by sentences with overlap"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_length = len(sentence)
        
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            overlap_start = max(0, len(current_chunk) - 3)
            current_chunk = current_chunk[overlap_start:]
            current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
        
        current_chunk.append(sentence)
        current_length += sentence_length + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Function to process documents
def process_documents(texts: List[str], metadata: List[dict]) -> List[dict]:
    """Process documents into chunks"""
    chunks = []
    
    for i, text in enumerate(texts):
        if not text or len(text.strip()) < 50:
            continue
            
        text_chunks = split_text_into_chunks(text)
        for j, chunk in enumerate(text_chunks):
            if len(chunk) > 30:
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        **metadata[i], 
                        "chunk_id": j,
                        "chunk_length": len(chunk),
                        "word_count": len(chunk.split()),
                        "is_resume": "resume" in metadata[i]["source"].lower() or "cv" in metadata[i]["source"].lower()
                    }
                })
    
    return chunks

# Function to initialize ChromaDB
def initialize_chromadb():
    """Initialize ChromaDB client and collection"""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        return client, embedding_func
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {str(e)}")
        return None, None

# Function to create a fresh collection
def create_fresh_collection(client, embedding_func, collection_name="documents"):
    """Create a fresh collection, deleting any existing one"""
    try:
        try:
            client.delete_collection(collection_name)
        except:
            pass
        
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_func,
            metadata={"created": datetime.now().isoformat()}
        )
        return collection
    except Exception as e:
        st.error(f"Error creating collection: {str(e)}")
        return None

# Function to add documents to ChromaDB
def add_to_chromadb(chunks: List[dict], collection):
    """Add document chunks to ChromaDB"""
    try:
        if not chunks:
            st.error("No valid text chunks to add to database.")
            return False
            
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]
        
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        return True
    except Exception as e:
        st.error(f"Error adding to ChromaDB: {str(e)}")
        return False

# Function to query ChromaDB with enhanced retrieval for resumes
def query_chromadb(query: str, collection, k: int = 8):
    """Query ChromaDB for similar documents with enhanced retrieval"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        resume_terms = ["project", "experience", "work", "portfolio", "skill"]
        if any(term in query.lower() for term in resume_terms):
            project_results = collection.query(
                query_texts=["projects experience work portfolio skills"],
                n_results=min(k, 4),
                include=["documents", "metadatas", "distances"]
            )
            
            if project_results and project_results['documents']:
                combined_docs = results['documents'][0] if results['documents'] else []
                combined_metas = results['metadatas'][0] if results['metadatas'] else []
                
                for doc, meta in zip(project_results['documents'][0], project_results['metadatas'][0]):
                    if doc not in combined_docs:
                        combined_docs.append(doc)
                        combined_metas.append(meta)
                
                results['documents'] = [combined_docs]
                results['metadatas'] = [combined_metas]
        
        return results
    except Exception as e:
        st.error(f"Error querying ChromaDB: {str(e)}")
        return None

# Function to generate response using Groq API with enhanced resume handling
def generate_groq_response(query: str, context: str, research_mode: bool = False) -> str:
    """Generate response using Groq API with enhanced resume handling"""
    try:
        if not GROQ_API_KEY:
            return "Error: GROQ_API_KEY not found. Please check your .env file."
        
        client = groq.Groq(api_key=GROQ_API_KEY)
        
        if any(term in query.lower() for term in ["project", "experience", "work", "portfolio"]):
            prompt = f"""
            Analyze the following resume content and provide a comprehensive response to the query.
            
            CONTEXT FROM RESUME:
            {context}
            
            QUERY:
            {query}
            
            Please:
            1. Extract ALL projects, experiences, or work items mentioned
            2. Provide complete details for each item
            3. If multiple items exist, list them all with clear separation
            4. Include technologies, durations, and achievements for each project
            5. Be thorough and don't miss any details from the context
            
            If the context doesn't contain relevant information, say so clearly.
            """
        elif research_mode:
            prompt = f"""
            Based EXCLUSIVELY on the following context, provide a comprehensive analysis with detailed explanations and evidence. 
            If the context doesn't contain relevant information, say so clearly.
            
            CONTEXT:
            {context}
            
            QUESTION:
            {query}
            
            Provide a thorough, well-structured response with supporting evidence from the context.
            """
        else:
            prompt = f"""
            Based EXCLUSIVELY on the following context, answer the question concisely.
            If the context doesn't contain relevant information, say "I don't have information about this in the provided documents."
            
            CONTEXT:
            {context}
            
            QUESTION:
            {query}
            
            Provide a clear and direct answer based only on the context.
            """
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts complete information from resumes and documents. When asked about projects or experiences, list ALL items found in the context with full details."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=2048
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Function to generate document summary
def generate_document_summary(text: str, filename: str) -> str:
    """Generate a summary for a document"""
    try:
        if not GROQ_API_KEY:
            return "API key not available for summarization."
        
        client = groq.Groq(api_key=GROQ_API_KEY)
        
        # Truncate very long texts to avoid token limits
        if len(text) > 12000:
            text = text[:12000] + "... [document truncated for summarization]"
        
        prompt = f"""
        Please provide a concise summary of the following document titled "{filename}".
        Focus on the key points, main topics, and important information.
        
        DOCUMENT CONTENT:
        {text}
        
        SUMMARY:
        """
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that creates concise, informative summaries of documents."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=512
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Function to extract entities for knowledge graph
def extract_entities(text: str) -> List[str]:
    """Extract key entities from text for knowledge graph"""
    # Simple entity extraction using patterns
    entities = set()
    
    # Extract potential names (title case words)
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    for word in words:
        if len(word) > 2 and word.lower() not in ['the', 'and', 'for', 'with', 'this', 'that']:
            entities.add(word)
    
    # Extract potential organizations (all caps or title case with length)
    orgs = re.findall(r'\b[A-Z][A-Za-z&]+\b', text)
    for org in orgs:
        if len(org) > 3 and org not in entities:
            entities.add(org)
    
    # Extract skills (common tech terms)
    tech_terms = [
        'python', 'java', 'javascript', 'html', 'css', 'react', 'node', 'sql', 
        'nosql', 'aws', 'azure', 'docker', 'kubernetes', 'machine learning',
        'ai', 'data analysis', 'web development', 'cloud computing'
    ]
    
    for term in tech_terms:
        if term in text.lower():
            entities.add(term.title())
    
    return list(entities)[:15]  # Limit to top 15 entities

# Function to create knowledge graph
def create_knowledge_graph(chunks: List[dict]) -> Dict[str, Any]:
    """Create a knowledge graph from document chunks"""
    nodes = []
    edges = []
    
    # Extract entities from each chunk
    for chunk in chunks:
        chunk_text = chunk["text"]
        source = chunk["metadata"]["source"]
        entities = extract_entities(chunk_text)
        
        # Add document as a node
        doc_node_id = f"doc_{source}"
        if not any(node['id'] == doc_node_id for node in nodes):
            nodes.append({
                "id": doc_node_id,
                "label": source,
                "type": "document",
                "size": 20
            })
        
        # Add entities and connect to document
        for entity in entities:
            entity_id = f"ent_{entity}"
            if not any(node['id'] == entity_id for node in nodes):
                nodes.append({
                    "id": entity_id,
                    "label": entity,
                    "type": "entity",
                    "size": 15
                })
            
            # Add edge between document and entity
            edges.append({
                "source": doc_node_id,
                "target": entity_id,
                "value": 5  # Connection strength
            })
    
    return {"nodes": nodes, "edges": edges}

# Function to visualize knowledge graph
def visualize_knowledge_graph(graph_data: Dict[str, Any]):
    """Visualize the knowledge graph using Plotly"""
    if not graph_data["nodes"]:
        st.info("No entities found to create a knowledge graph.")
        return
    
    # Create node positions in a circular layout
    num_nodes = len(graph_data["nodes"])
    radius = 150
    positions = {}
    
    for i, node in enumerate(graph_data["nodes"]):
        angle = 2 * math.pi * i / num_nodes
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        positions[node["id"]] = (x, y)
    
    # Create edge traces
    edge_traces = []
    for edge in graph_data["edges"]:
        x0, y0 = positions[edge["source"]]
        x1, y1 = positions[edge["target"]]
        
        edge_trace = go.Scatter(
            x=[x0, x1, None], 
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=1, color='#888'),
            hoverinfo='none'
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in graph_data["nodes"]:
        x, y = positions[node["id"]]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node["label"])
        node_size.append(node.get("size", 10))
        
        if node["type"] == "document":
            node_color.append("#1f77b4")  # Blue for documents
        else:
            node_color.append("#ff7f0e")  # Orange for entities
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=2, color='darkgray')
        )
    )   
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace],
                   layout=go.Layout(
                       title='Knowledge Graph',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=500
                   ))
    
    st.plotly_chart(fig, use_container_width=True)

# Function to create file preview
def create_file_preview(uploaded_file) -> str:
    """Create a text preview of a file"""
    try:
        text = extract_text_from_file(uploaded_file)
        if not text:
            return "No text content available for preview."
        
        # Create a preview (first 500 characters)
        preview = text[:500]
        if len(text) > 500:
            preview += "..."
            
        return preview
    except Exception as e:
        return f"Error creating preview: {str(e)}"

# Function to handle user input
def handle_userinput(user_question: str, research_mode: bool = False):
    """Process user input and generate response"""
    if st.session_state.collection is None or st.session_state.collection.count() == 0:
        st.error("Please upload and process documents first.")
        return ""
    
    is_resume_query = any(term in user_question.lower() for term in ["project", "experience", "work", "portfolio", "skill"])
    k = 10 if is_resume_query else (8 if research_mode else 6)
    
    results = query_chromadb(user_question, st.session_state.collection, k=k)
    
    if not results or not results.get('documents') or not results['documents'][0]:
        st.error("No relevant context found in the documents. Please try a different question.")
        return ""
    
    context_parts = []
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        source = metadata.get('source', 'Unknown')
        context_parts.append(f"[From {source}]:\n{doc}")
    
    context = "\n\n".join(context_parts)
    
    with st.spinner("Analyzing documents..." if is_resume_query else "Thinking..."):
        response = generate_groq_response(user_question, context, research_mode)
    
    # Add to current chat history
    current_history = st.session_state.chat_sessions.get(st.session_state.current_session_id, [])
    current_history.append((user_question, response, context))
    st.session_state.chat_sessions[st.session_state.current_session_id] = current_history
    
    # Update knowledge graph with new query context
    if research_mode:
        update_knowledge_graph_with_query(user_question, response)
    
    return response

# Function to update knowledge graph with query context
def update_knowledge_graph_with_query(query: str, response: str):
    """Update knowledge graph with information from query and response"""
    # Extract entities from query and response
    query_entities = extract_entities(query)
    response_entities = extract_entities(response)
    
    all_entities = set(query_entities + response_entities)
    
    # Add query as a special node
    query_node_id = f"query_{hash(query)}"
    if not any(node['id'] == query_node_id for node in st.session_state.knowledge_graph["nodes"]):
        st.session_state.knowledge_graph["nodes"].append({
            "id": query_node_id,
            "label": f"Query: {query[:30]}...",
            "type": "query",
            "size": 15
        })
    
    # Connect query to entities
    for entity in all_entities:
        entity_id = f"ent_{entity}"
        
        # Add entity if not exists
        if not any(node['id'] == entity_id for node in st.session_state.knowledge_graph["nodes"]):
            st.session_state.knowledge_graph["nodes"].append({
                "id": entity_id,
                "label": entity,
                "type": "entity",
                "size": 10
            })
        
        # Add edge between query and entity
        edge_exists = any(
            edge['source'] == query_node_id and edge['target'] == entity_id 
            or edge['source'] == entity_id and edge['target'] == query_node_id
            for edge in st.session_state.knowledge_graph["edges"]
        )
        
        if not edge_exists:
            st.session_state.knowledge_graph["edges"].append({
                "source": query_node_id,
                "target": entity_id,
                "value": 3
            })

# Function to reset chat
def reset_chat():
    """Reset the chat history for current session"""
    st.session_state.chat_sessions[st.session_state.current_session_id] = []

# Function to create a new chat session
def create_new_chat_session():
    """Create a new chat session"""
    new_session_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_session_id] = []
    st.session_state.current_session_id = new_session_id
    return new_session_id

# Function to switch chat session
def switch_chat_session(session_id: str):
    """Switch to a different chat session"""
    st.session_state.current_session_id = session_id

# Function to display chat messages
def display_chat_messages():
    """Display chat messages in a clean, modern format"""
    current_history = st.session_state.chat_sessions.get(st.session_state.current_session_id, [])
    
    for i, (question, answer, context) in enumerate(current_history):
        # User message
        st.markdown(f"""
        <div class="chat-message-user">
            <strong>👤 You:</strong> {question}
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant message
        st.markdown(f"""
        <div class="chat-message-assistant">
            <strong>🤖 Assistant:</strong> {answer}
        </div>
        """, unsafe_allow_html=True)

# Function to display document previews
def display_document_previews():
    """Display document preview thumbnails"""
    if not st.session_state.file_previews:
        return
    
    st.markdown("### 📄 Document Previews")
    
    for filename, preview in st.session_state.file_previews.items():
        with st.expander(f"📋 {filename}"):
            st.markdown(f"""
            <div class="preview-container">
                <pre>{preview}</pre>
            </div>
            """, unsafe_allow_html=True)
            
            # Show summary if available
            if filename in st.session_state.document_summaries:
                st.markdown(f"""
                <div class="summary-box">
                    <strong>📝 Summary:</strong><br>
                    {st.session_state.document_summaries[filename]}
                </div>
                """, unsafe_allow_html=True)

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">DocuChat 🤖</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Chat with your documents using AI-powered search</p>', unsafe_allow_html=True)
    
    # Check for API key
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found. Please create a .env file with your API key.")
        st.info("""
        Create a file named `.env` in your project directory with:
        ```
        GROQ_API_KEY=your_api_key_here
        ```
        """)
        return
    
    # Initialize ChromaDB client
    if st.session_state.chroma_client is None:
        st.session_state.chroma_client, embedding_func = initialize_chromadb()
        st.session_state.embedding_func = embedding_func
    
    # Upload section
    if st.session_state.show_upload or not st.session_state.processed_docs:
        with st.container():
            st.markdown('<div class="upload-container">', unsafe_allow_html=True)
            st.markdown("### 📁 Upload Your Documents")
            
            uploaded_files = st.file_uploader(
                "Add files",
                type=["pdf", "docx", "txt", "xlsx", "csv"],
                accept_multiple_files=True,
                key="file_uploader",
                help="Supported formats: PDF, Word, Text, Excel, CSV"
            )
            
            # Create previews for uploaded files
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.file_previews:
                        preview = create_file_preview(uploaded_file)
                        st.session_state.file_previews[uploaded_file.name] = preview
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                if st.button("📤 Process Documents", type="primary", use_container_width=True):
                    if uploaded_files:
                        with st.spinner("Processing documents..."):
                            st.session_state.collection = create_fresh_collection(
                                st.session_state.chroma_client, 
                                st.session_state.embedding_func
                            )
                            
                            all_texts = []
                            all_metadata = []
                            
                            for uploaded_file in uploaded_files:
                                text = extract_text_from_file(uploaded_file)
                                if text and len(text.strip()) > 100:
                                    all_texts.append(text)
                                    all_metadata.append({
                                        "source": uploaded_file.name,
                                        "upload_time": datetime.now().isoformat(),
                                        "file_size": len(uploaded_file.getvalue())
                                    })
                                    
                                    # Generate summary for each document
                                    if uploaded_file.name not in st.session_state.document_summaries:
                                        with st.spinner(f"Generating summary for {uploaded_file.name}..."):
                                            summary = generate_document_summary(text, uploaded_file.name)
                                            st.session_state.document_summaries[uploaded_file.name] = summary
                            
                            if all_texts:
                                chunks = process_documents(all_texts, all_metadata)
                                
                                if chunks:
                                    if add_to_chromadb(chunks, st.session_state.collection):
                                        # Create knowledge graph
                                        st.session_state.knowledge_graph = create_knowledge_graph(chunks)
                                        
                                        st.session_state.processed_docs = True
                                        st.session_state.documents = all_texts
                                        st.session_state.current_files = [f.name for f in uploaded_files]
                                        st.session_state.show_upload = False
                                        create_new_chat_session()
                                        st.success(f"✅ Processed {len(all_texts)} document(s) with {len(chunks)} chunks!")
                                        st.rerun()
                            else:
                                st.error("Failed to extract sufficient text from any documents.")
                    else:
                        st.warning("Please upload at least one document.")
            
            with col2:
                if st.session_state.processed_docs:
                    if st.button("🔄 Process New Files", use_container_width=True):
                        st.session_state.show_upload = True
                        st.rerun()
            
            with col3:
                if st.session_state.processed_docs:
                    if st.button("🗑️ Clear All", use_container_width=True):
                        if st.session_state.collection:
                            try:
                                st.session_state.chroma_client.delete_collection("documents")
                                st.session_state.collection = None
                            except:
                                pass
                            st.session_state.processed_docs = False
                            st.session_state.documents = []
                            st.session_state.current_files = []
                            st.session_state.file_previews = {}
                            st.session_state.document_summaries = {}
                            st.session_state.knowledge_graph = {"nodes": [], "edges": []}
                            st.session_state.show_upload = True
                            create_new_chat_session()
                            st.success("Database cleared!")
                            st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat interface
    if st.session_state.processed_docs:
        # Create two columns for main content and sidebar
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Status bar
            with st.container():
                st.markdown('<div class="status-box">', unsafe_allow_html=True)
                col1a, col2a, col3a = st.columns(3)
                with col1a:
                    st.write(f"**Documents:** {len(st.session_state.current_files)}")
                with col2a:
                    count = st.session_state.collection.count() if st.session_state.collection else 0
                    st.write(f"**Chunks:** {count}")
                with col3a:
                    mode = "🔍 Deep Research" if st.session_state.research_mode else "💬 Standard Chat"
                    st.write(f"**Mode:** {mode}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Toggle for research mode
            st.session_state.research_mode = st.toggle("Deep Research Mode", 
                                                      help="In-depth analysis with comprehensive responses")
            
            # Toggle for knowledge graph
            st.session_state.show_graph = st.toggle("Show Knowledge Graph", 
                                                   help="Visualize relationships between concepts in your documents")
            
            # Display knowledge graph if enabled
            if st.session_state.show_graph and st.session_state.knowledge_graph["nodes"]:
                visualize_knowledge_graph(st.session_state.knowledge_graph)
            
            # Display document previews
            display_document_previews()
            
            # Display chat messages
            display_chat_messages()
            
            # User input
            user_question = st.chat_input("Ask a question about your documents...")
            
            if user_question:
                # Add user question to chat immediately
                response = handle_userinput(user_question, st.session_state.research_mode)
                if response:
                    st.rerun()
        
        with col2:
            # Sidebar for chat history management and additional controls
            st.markdown("### 💬 Chat Sessions")
            
            # Button to create new chat
            if st.button("➕ New Chat", use_container_width=True):
                create_new_chat_session()
                st.rerun()
            
            # Display chat sessions
            for session_id in list(st.session_state.chat_sessions.keys()):
                session_history = st.session_state.chat_sessions[session_id]
                session_name = f"Chat {session_id[:8]}"
                if session_history:
                    session_name = f"Chat {session_id[:8]}: {session_history[0][0][:20]}..."
                
                is_active = session_id == st.session_state.current_session_id
                session_class = "chat-session active" if is_active else "chat-session"
                
                st.markdown(f'<div class="{session_class}" onclick="switchSession(\'{session_id}\')">{session_name}</div>', 
                           unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### ⚙️ Settings")
            
            if st.button("🧹 Clear Current Chat", use_container_width=True):
                reset_chat()
                st.rerun()
            
            st.markdown("---")
            st.markdown("### 💡 Tips")
            st.info("""
            - Ask about **projects**, **experience**, or **skills** in resumes
            - Use **specific questions** for better answers
            - Enable **Deep Research** for comprehensive analysis
            - Enable **Knowledge Graph** to visualize relationships
            - Supported formats: PDF, DOCX, TXT, XLSX, CSV
            """)
            
            # JavaScript for session switching
            st.markdown("""
            <script>
            function switchSession(sessionId) {
                window.location.href = window.location.href + "?session=" + sessionId;
            }
            </script>
            """, unsafe_allow_html=True)
            
            # Handle session switching from URL parameter
            query_params = st.experimental_get_query_params()
            if "session" in query_params:
                session_id = query_params["session"][0]
                if session_id in st.session_state.chat_sessions:
                    switch_chat_session(session_id)
                    st.experimental_set_query_params()
                    st.rerun()

if __name__ == "__main__":
    main()