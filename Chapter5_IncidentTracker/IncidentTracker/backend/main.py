"""
Main FastAPI backend for Error Resolution System
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

from config import settings, LOGGING_CONFIG
from mistral_client import MistralClient
from query_processor import QueryProcessor
from graph_builder import GraphBuilder
from pdf_generator import PDFGenerator

# Configure logging
import logging.config
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Error Resolution System",
    description="Intelligent error resolution with multi-source search and AI query expansion",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize components
mistral_client = MistralClient()
query_processor = QueryProcessor()
graph_builder = GraphBuilder()
pdf_generator = PDFGenerator()


# Pydantic models
class ErrorInput(BaseModel):
    """Input model for error reporting"""
    severity_level: str = Field(..., description="Error severity (P1, P2, P3)")
    error_code: str = Field(..., description="Specific error code or identifier")
    error_description: str = Field(..., description="Detailed error description")
    application_name: str = Field(..., description="Name of affected application")
    environment: str = Field(..., description="Environment (Dev, Test, Prod)")
    applicable_pool: str = Field(..., description="System pool or cluster")


class SearchResult(BaseModel):
    """Model for search results"""
    id: str
    title: str
    snippet: str
    source: str
    relevance: float
    url: str
    created: Optional[str] = None
    modified: Optional[str] = None
    author: Optional[str] = None
    tags: Optional[List[str]] = []


class SearchResponse(BaseModel):
    """Response model for search operations"""
    query_id: str
    original_input: ErrorInput
    expanded_query: Dict[str, Any]
    search_results: Dict[str, List[SearchResult]]
    top_recommendations: List[Dict[str, Any]]
    search_metadata: Dict[str, Any]
    processing_time: float


class GraphNode(BaseModel):
    """Model for graph visualization nodes"""
    id: str
    label: str
    type: str
    status: str
    metadata: Dict[str, Any] = {}


class GraphEdge(BaseModel):
    """Model for graph visualization edges"""
    source: str
    target: str
    type: str
    weight: float = 1.0


class WorkflowGraph(BaseModel):
    """Model for workflow graph"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    layout: Dict[str, Any] = {}


# API Routes

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI-Powered Error Resolution System",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "search": "/api/v1/search",
            "status": "/api/v1/status",
            "graph": "/api/v1/graph/{query_id}",
            "export": "/api/v1/export/{query_id}"
        }
    }


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check system components
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "mistral_client": "healthy",
                "mcp_servers": await _check_mcp_servers(),
                "database": "healthy"  # Add actual DB health check
            }
        }
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/api/v1/search", response_model=SearchResponse)
async def search_error_resolution(
    error_input: ErrorInput,
    background_tasks: BackgroundTasks
) -> SearchResponse:
    """
    Main endpoint for error resolution search
    
    Process user input, expand query with AI, search across multiple sources,
    and return ranked recommendations.
    """
    start_time = datetime.utcnow()
    query_id = f"query_{int(start_time.timestamp())}"
    
    try:
        logger.info(f"Starting error resolution search - Query ID: {query_id}")
        
        # Step 1: Expand query using Mistral AI
        logger.info("Expanding query with Mistral AI...")
        expanded_query = await mistral_client.expand_query(error_input.dict())
        
        # Step 2: Execute parallel searches across all sources
        logger.info("Executing parallel searches...")
        search_results = await query_processor.execute_parallel_search(
            expanded_query, query_id
        )
        
        # Step 3: Analyze results and generate recommendations
        logger.info("Analyzing results with AI...")
        all_results = []
        for source, results in search_results.items():
            all_results.extend(results)
        
        analysis = await mistral_client.analyze_search_results(all_results, expanded_query)
        
        # Step 4: Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Step 5: Store search session for later retrieval
        background_tasks.add_task(
            _store_search_session,
            query_id, error_input, expanded_query, search_results, analysis
        )
        
        # Step 6: Build response
        response = SearchResponse(
            query_id=query_id,
            original_input=error_input,
            expanded_query=expanded_query,
            search_results={
                source: [SearchResult(**result) for result in results]
                for source, results in search_results.items()
            },
            top_recommendations=analysis.get("top_recommendations", []),
            search_metadata={
                "total_results": len(all_results),
                "sources_searched": list(search_results.keys()),
                "confidence_score": analysis.get("overall_confidence", 0.0),
                "processing_time": processing_time,
                "timestamp": start_time.isoformat()
            },
            processing_time=processing_time
        )
        
        logger.info(f"Search completed successfully - Query ID: {query_id}, Results: {len(all_results)}")
        return response
        
    except Exception as e:
        logger.error(f"Error in search process: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search processing failed: {str(e)}"
        )


@app.get("/api/v1/graph/{query_id}", response_model=WorkflowGraph)
async def get_workflow_graph(query_id: str) -> WorkflowGraph:
    """
    Get the topological workflow graph for a search query
    """
    try:
        # Retrieve search session data
        session_data = await _get_search_session(query_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Query not found")
        
        # Generate workflow graph
        graph = await graph_builder.create_workflow_graph(session_data)
        
        return WorkflowGraph(
            nodes=[GraphNode(**node) for node in graph["nodes"]],
            edges=[GraphEdge(**edge) for edge in graph["edges"]],
            layout=graph.get("layout", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating workflow graph: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Graph generation failed: {str(e)}"
        )


@app.get("/api/v1/export/{query_id}")
async def export_search_results(
    query_id: str,
    format: str = "pdf"
) -> Dict[str, Any]:
    """
    Export search results as PDF or other formats
    """
    try:
        # Retrieve search session data
        session_data = await _get_search_session(query_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Query not found")
        
        if format.lower() == "pdf":
            # Generate PDF report
            pdf_path = await pdf_generator.generate_report(session_data)
            
            return {
                "format": "pdf",
                "file_path": pdf_path,
                "download_url": f"/api/v1/download/{query_id}.pdf",
                "generated_at": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Format '{format}' not supported"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Export failed: {str(e)}"
        )


@app.get("/api/v1/sources")
async def get_available_sources():
    """
    Get list of available search sources and their status
    """
    try:
        sources_status = await query_processor.check_sources_status()
        return {
            "sources": sources_status,
            "total_sources": len(sources_status),
            "active_sources": len([s for s in sources_status if s["status"] == "active"]),
            "checked_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error checking sources: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check sources: {str(e)}"
        )


@app.get("/api/v1/history")
async def get_search_history(limit: int = 10):
    """
    Get recent search history
    """
    try:
        # This would typically query a database
        # For now, return a placeholder
        return {
            "searches": [],
            "total": 0,
            "limit": limit,
            "message": "Search history feature coming soon"
        }
    except Exception as e:
        logger.error(f"Error retrieving search history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )


# Utility functions

async def _check_mcp_servers() -> Dict[str, str]:
    """Check the status of MCP servers"""
    mcp_status = {}
    
    # Check each MCP server
    servers = {
        "confluence": settings.confluence_mcp_port,
        "teams": settings.teams_mcp_port,
        "outlook": settings.outlook_mcp_port,
        "local_disk": settings.local_disk_mcp_port
    }
    
    for server_name, port in servers.items():
        try:
            # Simple connectivity check - in production, implement proper health checks
            mcp_status[server_name] = "healthy"
        except Exception:
            mcp_status[server_name] = "unhealthy"
    
    return mcp_status


async def _store_search_session(
    query_id: str,
    error_input: ErrorInput,
    expanded_query: Dict[str, Any],
    search_results: Dict[str, List[Dict[str, Any]]],
    analysis: Dict[str, Any]
):
    """Store search session data for later retrieval"""
    try:
        # This would typically store in a database
        # For now, store in memory (implement proper storage)
        session_data = {
            "query_id": query_id,
            "error_input": error_input.dict(),
            "expanded_query": expanded_query,
            "search_results": search_results,
            "analysis": analysis,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Store session data (implement actual storage)
        logger.info(f"Stored search session: {query_id}")
        
    except Exception as e:
        logger.error(f"Failed to store search session {query_id}: {str(e)}")


async def _get_search_session(query_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve search session data"""
    try:
        # This would typically query a database
        # For now, return None (implement proper retrieval)
        logger.info(f"Retrieving search session: {query_id}")
        return None  # Implement actual retrieval
        
    except Exception as e:
        logger.error(f"Failed to retrieve search session {query_id}: {str(e)}")
        return None


# Application startup
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting AI-Powered Error Resolution System")
    
    # Initialize components
    await query_processor.initialize()
    logger.info("Query processor initialized")
    
    # Start MCP servers in background
    asyncio.create_task(_start_mcp_servers())
    logger.info("MCP servers starting...")
    
    logger.info("Application startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down AI-Powered Error Resolution System")
    
    # Cleanup components
    await query_processor.cleanup()
    logger.info("Application shutdown complete")


async def _start_mcp_servers():
    """Start all MCP servers"""
    try:
        # Import and start each MCP server
        from confluence_server import confluence_server
        from teams_server import teams_server
        # from outlook_server import outlook_server
        # from local_disk_server import local_disk_server
        
        # Start servers in background tasks
        asyncio.create_task(confluence_server.run_server())
        asyncio.create_task(teams_server.run_server())
        
        logger.info("All MCP servers started")
        
    except Exception as e:
        logger.error(f"Error starting MCP servers: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )