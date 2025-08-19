# app.py
import streamlit as st
import os
from typing import List
import uuid
from datetime import datetime
import io
import re

# Import document processing libraries
import pypdf
from docx import Document as DocxDocument
import pandas as pd

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
    page_title="RAG Chat System",
    page_icon="📚",
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

# Get API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Function to extract text from different file types
def extract_text_from_file(uploaded_file) -> str:
    """Extract text from various file types"""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    text = ""
    
    try:
        if file_extension == ".pdf":
            pdf_reader = pypdf.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        elif file_extension in [".docx", ".doc"]:
            doc = DocxDocument(io.BytesIO(uploaded_file.getvalue()))
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
        
        elif file_extension == ".txt":
            text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(uploaded_file)
            text = df.to_string()
        
        elif file_extension == ".csv":
            # Try different encodings for CSV
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    text = df.to_string()
                    break
                except UnicodeDecodeError:
                    continue
            if not text:
                # Fallback: read as text
                uploaded_file.seek(0)
                text = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return ""
        
        # Clean up text - remove excessive whitespace but preserve structure
        text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with single space
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Replace multiple newlines with double newline
        text = text.strip()
        
        return text if text else ""
    except Exception as e:
        st.error(f"Error extracting text from {uploaded_file.name}: {str(e)}")
        return ""

# Function to split text into chunks
def split_text_into_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks with better boundary detection"""
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Determine end position
        end = start + chunk_size
        
        if end >= text_length:
            chunks.append(text[start:].strip())
            break
        
        # Look for good breaking points (prioritize paragraph breaks, then sentences)
        break_points = [
            text.rfind('\n\n', start, end),  # Paragraph break
            text.rfind('. ', start, end),    # Sentence end with space
            text.rfind('? ', start, end),    # Question end
            text.rfind('! ', start, end),    # Exclamation end
            text.rfind('\n', start, end),    # Line break
            text.rfind('; ', start, end),    # Semicolon
            text.rfind(': ', start, end),    # Colon
            text.rfind(', ', start, end),    # Comma
            text.rfind(' ', start, end),     # Space
        ]
        
        # Find the best break point
        break_point = -1
        for bp in break_points:
            if bp > start + chunk_size // 2:  # Ensure we don't make chunks too small
                break_point = bp
                break
        
        # If no good break point found, just break at the chunk size
        if break_point == -1:
            break_point = end
        else:
            # Adjust for the break point character(s)
            if text[break_point:break_point+2] == '\n\n':
                break_point += 2
            elif break_point < text_length - 1 and text[break_point+1] == ' ':
                break_point += 2
            else:
                break_point += 1
        
        chunks.append(text[start:break_point].strip())
        start = break_point - chunk_overlap
        if start < 0:
            start = 0
    
    # Filter out empty chunks
    return [chunk for chunk in chunks if chunk]

# Function to process documents
def process_documents(texts: List[str], metadata: List[dict]) -> List[dict]:
    """Process documents into chunks"""
    chunks = []
    
    for i, text in enumerate(texts):
        if not text or len(text.strip()) < 50:  # Skip very short texts
            continue
            
        text_chunks = split_text_into_chunks(text)
        for j, chunk in enumerate(text_chunks):
            if len(chunk) > 50:  # Only add meaningful chunks
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        **metadata[i], 
                        "chunk_id": j,
                        "chunk_length": len(chunk),
                        "word_count": len(chunk.split())
                    }
                })
    
    return chunks

# Function to initialize ChromaDB
def initialize_chromadb():
    """Initialize ChromaDB client and collection"""
    try:
        # Initialize Chroma client with new API
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create embedding function
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
        # Delete existing collection if it exists
        try:
            client.delete_collection(collection_name)
        except:
            pass  # Collection didn't exist
        
        # Create new collection
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

# Function to query ChromaDB
def query_chromadb(query: str, collection, k: int = 4):
    """Query ChromaDB for similar documents"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        return results
    except Exception as e:
        st.error(f"Error querying ChromaDB: {str(e)}")
        return None

# Function to generate response using Groq API
def generate_groq_response(query: str, context: str, research_mode: bool = False) -> str:
    """Generate response using Groq API"""
    try:
        # Check if API key is available
        if not GROQ_API_KEY:
            return "Error: GROQ_API_KEY not found. Please check your .env file."
        
        # Initialize Groq client
        client = groq.Groq(api_key=GROQ_API_KEY)
        
        # Prepare prompt based on mode
        if research_mode:
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
        
        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based ONLY on the provided context. If the context doesn't contain the answer, say so clearly."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=1024
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Function to handle user input
def handle_userinput(user_question: str, research_mode: bool = False):
    """Process user input and generate response"""
    if st.session_state.collection is None or st.session_state.collection.count() == 0:
        st.error("Please upload and process documents first.")
        return
    
    # Query ChromaDB for relevant context
    results = query_chromadb(user_question, st.session_state.collection, k=6 if research_mode else 4)
    
    if not results or not results.get('documents') or not results['documents'][0]:
        st.error("No relevant context found in the documents. Please try a different question.")
        return
    
    # Combine context from retrieved documents with source information
    context_parts = []
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        source = metadata.get('source', 'Unknown')
        context_parts.append(f"[From {source}]:\n{doc}")
    
    context = "\n\n".join(context_parts)
    
    # Generate response
    with st.spinner("Thinking..."):
        response = generate_groq_response(user_question, context, research_mode)
    
    # Update chat history
    st.session_state.chat_history.append((user_question, response, context))
    
    # Display response
    with st.chat_message("assistant"):
        st.markdown(response)
        
        # Show source documents in an expander
        with st.expander("Source Context"):
            if len(context) > 3000:
                st.text(context[:3000] + "...")
            else:
                st.text(context)

# Function to reset chat
def reset_chat():
    """Reset the chat history"""
    st.session_state.chat_history = []

# Main application
def main():
    st.title("📚 RAG-Based Chat System")
    st.markdown("Upload documents, ask questions, and get AI-powered answers based on your content.")
    
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
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header("📁 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose documents",
            type=["pdf", "docx", "txt", "xlsx", "csv"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        # Process documents button
        if st.button("Process Documents", type="primary"):
            if uploaded_files:
                # Check if files are different from previous upload
                current_file_names = [f.name for f in uploaded_files]
                if (st.session_state.processed_docs and 
                    set(current_file_names) == set(st.session_state.current_files)):
                    st.info("Same documents already processed. Using existing database.")
                else:
                    with st.spinner("Processing documents..."):
                        # Create a fresh collection
                        st.session_state.collection = create_fresh_collection(
                            st.session_state.chroma_client, 
                            st.session_state.embedding_func
                        )
                        
                        # Extract text from files
                        all_texts = []
                        all_metadata = []
                        
                        for uploaded_file in uploaded_files:
                            with st.spinner(f"Processing {uploaded_file.name}..."):
                                text = extract_text_from_file(uploaded_file)
                                if text and len(text.strip()) > 100:  # Reasonable minimum text
                                    all_texts.append(text)
                                    all_metadata.append({
                                        "source": uploaded_file.name,
                                        "upload_time": datetime.now().isoformat(),
                                        "file_size": len(uploaded_file.getvalue())
                                    })
                                    st.success(f"✓ {uploaded_file.name} ({len(text)} chars)")
                                else:
                                    st.warning(f"⚠️ {uploaded_file.name} - too little text extracted")
                        
                        if all_texts:
                            # Process documents into chunks
                            chunks = process_documents(all_texts, all_metadata)
                            
                            if chunks:
                                # Add to ChromaDB
                                if add_to_chromadb(chunks, st.session_state.collection):
                                    st.session_state.processed_docs = True
                                    st.session_state.documents = all_texts
                                    st.session_state.current_files = current_file_names
                                    st.success(f"✅ Processed {len(all_texts)} document(s) with {len(chunks)} chunks!")
                                    reset_chat()
                            else:
                                st.error("No valid text chunks could be created from the documents.")
                        else:
                            st.error("Failed to extract sufficient text from any documents.")
            else:
                st.warning("Please upload at least one document.")
        
        st.divider()
        
        # Research mode toggle
        st.session_state.research_mode = st.toggle("Deep Research Mode", 
                                                  help="In-depth analysis with comprehensive responses")
        
        # Clear chat button
        if st.button("Clear Chat"):
            reset_chat()
            st.rerun()
        
        # Clear database button
        if st.button("Clear Database"):
            if st.session_state.collection:
                try:
                    st.session_state.chroma_client.delete_collection("documents")
                    st.session_state.collection = None
                except:
                    pass
                st.session_state.processed_docs = False
                st.session_state.documents = []
                st.session_state.current_files = []
                reset_chat()
                st.success("Database cleared!")
        
        # Display document information if processed
        if st.session_state.processed_docs and st.session_state.collection:
            st.divider()
            st.subheader("Processed Documents")
            count = st.session_state.collection.count()
            st.write(f"Chunks in database: {count}")
            for i, file_name in enumerate(st.session_state.current_files):
                st.caption(f"Document {i+1}: {file_name}")

    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display chat history
        st.subheader("💬 Chat")
        
        if not st.session_state.processed_docs:
            st.info("👆 Upload and process documents in the sidebar to start chatting")
        
        # Display chat messages
        for i, (question, answer, context) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(f"**You:** {question}")
            with st.chat_message("assistant"):
                st.markdown(answer)
        
        # User input
        if st.session_state.processed_docs:
            user_question = st.chat_input("Ask a question about your documents...")
            if user_question:
                with st.chat_message("user"):
                    st.markdown(f"**You:** {user_question}")
                handle_userinput(user_question, st.session_state.research_mode)
    
    with col2:
        st.subheader("ℹ️ System Information")
        
        if st.session_state.processed_docs:
            st.success("✅ Documents processed and ready for queries")
            if st.session_state.collection:
                count = st.session_state.collection.count()
                st.write(f"Chunks in database: {count}")
                st.write(f"Documents: {len(st.session_state.current_files)}")
        else:
            st.info("Please upload and process documents to start chatting")
        
        if st.session_state.research_mode:
            st.info("🔍 Deep Research Mode: ON")
        else:
            st.info("💬 Standard Chat Mode: ON")
        
        st.divider()
        st.subheader("📊 Usage Tips")
        st.markdown("""
        - Upload documents and click **Process Documents**
        - Each time you upload new files, the system creates a fresh database
        - Use **Deep Research Mode** for detailed analysis
        - Ask specific questions for better answers
        - **Clear Database** to start with new documents
        - Supported formats: PDF, DOCX, TXT, XLSX, CSV
        """)

if __name__ == "__main__":
    main()