
# main.py - FastAPI Backend with Semantic Cache, ChromaDB, and MCP Integration

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import asyncio
import hashlib
import json
import os
from datetime import datetime
import logging
from contextlib import asynccontextmanager

# Core dependencies
import chromadb
from chromadb.config import Settings
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
import base64

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class ProcessRequest(BaseModel):
    service_id: str
    prompt: str
    user_id: Optional[str] = "default_user"

class ProcessResponse(BaseModel):
    service_id: str
    prompt: str
    response: str
    cached: bool
    processing_time: float
    metadata: Dict[str, Any]

class DownloadRequest(BaseModel):
    content: str
    filename: Optional[str] = "report"

# Global Variables
semantic_cache = {}
chroma_client = None
embedding_model = None
mcp_tools = {}

# Semantic Cache Implementation
class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.cache = {}
        self.embeddings = {}

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using sentence transformers"""
        global embedding_model
        if embedding_model is None:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return embedding_model.encode(text)

    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def get(self, prompt: str, service_id: str) -> Optional[Dict]:
        """Get cached response for semantically similar prompt"""
        query_embedding = self._generate_embedding(f"{service_id}:{prompt}")

        for cache_key, cache_data in self.cache.items():
            if cache_key.startswith(f"{service_id}:"):
                cached_embedding = self.embeddings.get(cache_key)
                if cached_embedding is not None:
                    similarity = self._calculate_similarity(query_embedding, cached_embedding)
                    if similarity >= self.similarity_threshold:
                        logger.info(f"Cache hit with similarity: {similarity:.3f}")
                        return cache_data

        return None

    def set(self, prompt: str, service_id: str, response: Dict):
        """Cache response with its embedding"""
        cache_key = f"{service_id}:{prompt}"
        query_embedding = self._generate_embedding(cache_key)

        self.cache[cache_key] = response
        self.embeddings[cache_key] = query_embedding
        logger.info(f"Cached response for service: {service_id}")

# ChromaDB Integration
class ChromaDBManager:
    def __init__(self):
        self.client = None
        self.collection = None

    async def initialize(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Use in-memory client for demo (replace with persistent in production)
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(
                name="ai_service_responses",
                metadata={"description": "AI service responses and metadata"}
            )
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")

    async def store_response(self, service_id: str, prompt: str, response: str, metadata: Dict):
        """Store response and metadata in ChromaDB"""
        if self.collection is None:
            return

        try:
            doc_id = hashlib.md5(f"{service_id}:{prompt}".encode()).hexdigest()
            self.collection.add(
                documents=[response],
                metadatas=[{**metadata, "service_id": service_id, "prompt": prompt}],
                ids=[doc_id]
            )
            logger.info(f"Stored response in ChromaDB: {doc_id}")
        except Exception as e:
            logger.error(f"Failed to store in ChromaDB: {e}")

# MCP Server Integration (Mock Implementation)
class MCPServerManager:
    def __init__(self):
        self.tools = {
            "semantic_search": self._semantic_search_tool,
            "text_analysis": self._text_analysis_tool,
            "document_qa": self._document_qa_tool,
            "data_extraction": self._data_extraction_tool,
            "sentiment_analysis": self._sentiment_analysis_tool,
            "summarization": self._summarization_tool,
            "translation": self._translation_tool,
            "content_generation": self._content_generation_tool
        }

    async def call_tool(self, service_id: str, prompt: str) -> Dict[str, Any]:
        """Call appropriate MCP tool based on service ID"""
        tool = self.tools.get(service_id)
        if tool:
            return await tool(prompt)
        else:
            return {"error": f"No tool found for service: {service_id}"}

    async def _semantic_search_tool(self, prompt: str) -> Dict[str, Any]:
        """Mock semantic search implementation"""
        return {
            "response": f"Semantic search results for: '{prompt}'. Found 5 relevant documents with high similarity scores.",
            "metadata": {"tool": "semantic_search", "documents_found": 5, "avg_similarity": 0.87}
        }

    async def _text_analysis_tool(self, prompt: str) -> Dict[str, Any]:
        """Mock text analysis implementation"""
        return {
            "response": f"Text analysis of: '{prompt}'. Detected language: English. Complexity: Medium. Key entities extracted.",
            "metadata": {"tool": "text_analysis", "language": "en", "complexity": "medium", "entities": 3}
        }

    async def _document_qa_tool(self, prompt: str) -> Dict[str, Any]:
        """Mock document Q&A implementation"""
        return {
            "response": f"Answer to '{prompt}': Based on the document analysis, the key information indicates comprehensive coverage of the topic with high confidence.",
            "metadata": {"tool": "document_qa", "confidence": 0.92, "sources": 2}
        }

    async def _data_extraction_tool(self, prompt: str) -> Dict[str, Any]:
        """Mock data extraction implementation"""
        return {
            "response": f"Extracted structured data from: '{prompt}'. Identified key-value pairs and relationships.",
            "metadata": {"tool": "data_extraction", "extracted_fields": 8, "confidence": 0.89}
        }

    async def _sentiment_analysis_tool(self, prompt: str) -> Dict[str, Any]:
        """Mock sentiment analysis implementation"""
        return {
            "response": f"Sentiment analysis of: '{prompt}'. Overall sentiment: Positive (0.73). High confidence assessment.",
            "metadata": {"tool": "sentiment_analysis", "sentiment": "positive", "score": 0.73, "confidence": 0.91}
        }

    async def _summarization_tool(self, prompt: str) -> Dict[str, Any]:
        """Mock summarization implementation"""
        return {
            "response": f"Summary of: '{prompt}'. Key points extracted and condensed into concise overview maintaining essential information.",
            "metadata": {"tool": "summarization", "compression_ratio": 0.25, "key_points": 4}
        }

    async def _translation_tool(self, prompt: str) -> Dict[str, Any]:
        """Mock translation implementation"""
        return {
            "response": f"Translation of: '{prompt}'. Translated to target language with high accuracy and context preservation.",
            "metadata": {"tool": "translation", "source_lang": "auto", "target_lang": "en", "confidence": 0.94}
        }

    async def _content_generation_tool(self, prompt: str) -> Dict[str, Any]:
        """Mock content generation implementation"""
        return {
            "response": f"Generated content for: '{prompt}'. Created engaging, relevant content tailored to specifications with creative elements.",
            "metadata": {"tool": "content_generation", "word_count": 150, "creativity_score": 0.88}
        }

# PDF Generation
class PDFGenerator:
    @staticmethod
    def create_report(content: str, metadata: Dict[str, Any], filename: str = "report") -> str:
        """Generate PDF report from content and metadata"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        story.append(Paragraph("AI Service Report", title_style))
        story.append(Spacer(1, 12))

        # Metadata Table
        if metadata:
            story.append(Paragraph("Service Information", styles['Heading2']))
            metadata_data = [[key.replace('_', ' ').title(), str(value)] for key, value in metadata.items()]
            metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(metadata_table)
            story.append(Spacer(1, 12))

        # Content
        story.append(Paragraph("Response Content", styles['Heading2']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(content, styles['Normal']))

        # Footer
        story.append(Spacer(1, 24))
        footer_text = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Paragraph(footer_text, styles['Normal']))

        doc.build(story)

        # Save to file
        buffer.seek(0)
        file_path = f"/tmp/{filename}.pdf"
        with open(file_path, 'wb') as f:
            f.write(buffer.read())

        return file_path

# Global instances
semantic_cache_instance = SemanticCache()
chromadb_manager = ChromaDBManager()
mcp_manager = MCPServerManager()

# FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting AI Service Dashboard Backend...")
    await chromadb_manager.initialize()
    logger.info("Backend initialization complete")
    yield
    # Shutdown
    logger.info("Shutting down backend...")

# FastAPI App
app = FastAPI(
    title="AI Service Dashboard Backend",
    description="Backend API for AI Service Dashboard with Semantic Cache, ChromaDB, and MCP Integration",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/")
async def root():
    return {"message": "AI Service Dashboard Backend", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "semantic_cache": len(semantic_cache_instance.cache),
            "chromadb": "connected" if chromadb_manager.client else "disconnected",
            "mcp_tools": len(mcp_manager.tools)
        }
    }

@app.post("/api/process", response_model=ProcessResponse)
async def process_request(request: ProcessRequest):
    """Process user request through semantic cache and MCP tools"""
    start_time = datetime.now()

    try:
        # Check semantic cache first
        cached_response = semantic_cache_instance.get(request.prompt, request.service_id)

        if cached_response:
            # Cache hit
            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessResponse(
                service_id=request.service_id,
                prompt=request.prompt,
                response=cached_response["response"],
                cached=True,
                processing_time=processing_time,
                metadata=cached_response.get("metadata", {})
            )

        # Cache miss - call MCP tool
        mcp_result = await mcp_manager.call_tool(request.service_id, request.prompt)

        if "error" in mcp_result:
            raise HTTPException(status_code=400, detail=mcp_result["error"])

        # Prepare response data
        response_data = {
            "response": mcp_result["response"],
            "metadata": mcp_result.get("metadata", {})
        }

        # Cache the response
        semantic_cache_instance.set(request.prompt, request.service_id, response_data)

        # Store in ChromaDB
        await chromadb_manager.store_response(
            service_id=request.service_id,
            prompt=request.prompt,
            response=mcp_result["response"],
            metadata={
                **mcp_result.get("metadata", {}),
                "user_id": request.user_id,
                "timestamp": datetime.now().isoformat()
            }
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return ProcessResponse(
            service_id=request.service_id,
            prompt=request.prompt,
            response=mcp_result["response"],
            cached=False,
            processing_time=processing_time,
            metadata=mcp_result.get("metadata", {})
        )

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/download")
async def download_pdf(request: DownloadRequest):
    """Generate and download PDF report"""
    try:
        # Extract metadata from content if available
        metadata = {
            "content_length": len(request.content),
            "generated_at": datetime.now().isoformat(),
            "filename": request.filename
        }

        # Generate PDF
        pdf_path = PDFGenerator.create_report(
            content=request.content,
            metadata=metadata,
            filename=request.filename
        )

        return FileResponse(
            path=pdf_path,
            filename=f"{request.filename}.pdf",
            media_type="application/pdf"
        )

    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")

@app.get("/api/services")
async def get_services():
    """Get available AI services"""
    services = [
        {"id": "semantic_search", "name": "Semantic Search", "type": "search"},
        {"id": "text_analysis", "name": "Text Analysis", "type": "nlp"},
        {"id": "document_qa", "name": "Document Q&A", "type": "qa"},
        {"id": "data_extraction", "name": "Data Extraction", "type": "extraction"},
        {"id": "sentiment_analysis", "name": "Sentiment Analysis", "type": "sentiment"},
        {"id": "summarization", "name": "Text Summarization", "type": "summarization"},
        {"id": "translation", "name": "Translation", "type": "translation"},
        {"id": "content_generation", "name": "Content Generation", "type": "generation"}
    ]
    return {"services": services}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
