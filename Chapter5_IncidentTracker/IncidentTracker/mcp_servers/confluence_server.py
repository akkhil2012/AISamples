"""
MCP Server for Confluence Search Integration
"""
import asyncio
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP
from atlassian import Confluence
from config import settings
import logging
import re
from urllib.parse import quote

logger = logging.getLogger(__name__)


class ConfluenceMCPServer:
    """MCP Server for Confluence search operations"""
    
    def __init__(self):
        self.mcp = FastMCP("Confluence Search Server")
        self.confluence_client = None
        self._setup_tools()
        self._setup_resources()
        
    def _setup_confluence_client(self):
        """Initialize Confluence client"""
        try:
            self.confluence_client = Confluence(
                url=settings.confluence_url,
                username=settings.confluence_username,
                password=settings.confluence_api_token,
                cloud=True
            )
            logger.info("Confluence client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Confluence client: {str(e)}")
            self.confluence_client = None
    
    def _setup_tools(self):
        """Setup MCP tools for Confluence operations"""
        
        @self.mcp.tool()
        async def search_confluence(
            query: str,
            space_key: str = None,
            content_type: str = "page",
            limit: int = 25
        ) -> List[Dict[str, Any]]:
            """
            Search Confluence for content matching the query
            
            Args:
                query: Search query string
                space_key: Specific space to search (optional)
                content_type: Type of content to search (page, blogpost, comment)
                limit: Maximum number of results to return
                
            Returns:
                List of search results with metadata
            """
            if not self.confluence_client:
                self._setup_confluence_client()
                
            if not self.confluence_client:
                return [{"error": "Confluence client not available"}]
            
            try:
                # Build CQL (Confluence Query Language) query
                cql_query = self._build_cql_query(query, space_key, content_type)
                
                # Execute search
                results = self.confluence_client.cql(cql_query, limit=limit)
                
                # Process and format results
                formatted_results = []
                for result in results.get('results', []):
                    formatted_result = self._format_confluence_result(result)
                    formatted_results.append(formatted_result)
                
                logger.info(f"Found {len(formatted_results)} Confluence results for query: {query}")
                return formatted_results
                
            except Exception as e:
                logger.error(f"Confluence search error: {str(e)}")
                return [{"error": f"Search failed: {str(e)}"}]
        
        @self.mcp.tool()
        async def get_confluence_page(page_id: str) -> Dict[str, Any]:
            """
            Retrieve full content of a Confluence page
            
            Args:
                page_id: ID of the page to retrieve
                
            Returns:
                Page content and metadata
            """
            if not self.confluence_client:
                self._setup_confluence_client()
                
            if not self.confluence_client:
                return {"error": "Confluence client not available"}
            
            try:
                # Get page content
                page = self.confluence_client.get_page_by_id(
                    page_id, 
                    expand="body.storage,version,space,ancestors"
                )
                
                # Format page data
                formatted_page = {
                    "id": page.get("id"),
                    "title": page.get("title"),
                    "type": page.get("type"),
                    "space": {
                        "key": page.get("space", {}).get("key"),
                        "name": page.get("space", {}).get("name")
                    },
                    "version": page.get("version", {}).get("number"),
                    "created": page.get("history", {}).get("createdDate"),
                    "modified": page.get("version", {}).get("when"),
                    "author": page.get("version", {}).get("by", {}).get("displayName"),
                    "url": f"{settings.confluence_url}/wiki{page.get('_links', {}).get('webui', '')}",
                    "content": self._extract_page_content(page),
                    "source": "confluence"
                }
                
                return formatted_page
                
            except Exception as e:
                logger.error(f"Error retrieving Confluence page {page_id}: {str(e)}")
                return {"error": f"Failed to retrieve page: {str(e)}"}
        
        @self.mcp.tool()
        async def search_confluence_spaces(query: str = "") -> List[Dict[str, Any]]:
            """
            List available Confluence spaces
            
            Args:
                query: Optional query to filter spaces
                
            Returns:
                List of available spaces
            """
            if not self.confluence_client:
                self._setup_confluence_client()
                
            if not self.confluence_client:
                return [{"error": "Confluence client not available"}]
            
            try:
                spaces = self.confluence_client.get_all_spaces(
                    space_type="global",
                    limit=100
                )
                
                formatted_spaces = []
                for space in spaces.get('results', []):
                    if not query or query.lower() in space.get('name', '').lower():
                        formatted_spaces.append({
                            "key": space.get("key"),
                            "name": space.get("name"),
                            "description": space.get("description", {}).get("plain", {}).get("value", ""),
                            "type": space.get("type"),
                            "url": f"{settings.confluence_url}/wiki{space.get('_links', {}).get('webui', '')}"
                        })
                
                return formatted_spaces
                
            except Exception as e:
                logger.error(f"Error listing Confluence spaces: {str(e)}")
                return [{"error": f"Failed to list spaces: {str(e)}"}]
    
    def _setup_resources(self):
        """Setup MCP resources for Confluence data"""
        
        @self.mcp.resource("confluence://spaces")
        async def confluence_spaces() -> str:
            """Get list of all available Confluence spaces"""
            spaces = await self.search_confluence_spaces()
            return str(spaces)
        
        @self.mcp.resource("confluence://recent/{space_key}")
        async def recent_pages(space_key: str) -> str:
            """Get recent pages from a specific space"""
            if not self.confluence_client:
                self._setup_confluence_client()
                
            if not self.confluence_client:
                return "Confluence client not available"
            
            try:
                # Get recent content from space
                results = self.confluence_client.get_all_pages_from_space(
                    space_key, 
                    expand="version,space",
                    limit=20
                )
                
                recent_pages = []
                for page in results:
                    recent_pages.append({
                        "id": page.get("id"),
                        "title": page.get("title"),
                        "modified": page.get("version", {}).get("when"),
                        "url": f"{settings.confluence_url}/wiki{page.get('_links', {}).get('webui', '')}"
                    })
                
                return str(recent_pages)
                
            except Exception as e:
                logger.error(f"Error getting recent pages from {space_key}: {str(e)}")
                return f"Error: {str(e)}"
    
    def _build_cql_query(self, query: str, space_key: str = None, content_type: str = "page") -> str:
        """Build Confluence Query Language (CQL) query"""
        
        # Escape special characters in query
        escaped_query = re.sub(r'[^\w\s]', ' ', query)
        
        # Build base query
        cql_parts = []
        
        # Add content type filter
        if content_type:
            cql_parts.append(f"type = {content_type}")
        
        # Add space filter
        if space_key:
            cql_parts.append(f"space = {space_key}")
        
        # Add text search
        if escaped_query.strip():
            # Use text search for better relevance
            cql_parts.append(f"text ~ \"{escaped_query}\"")
        
        # Combine parts
        cql_query = " AND ".join(cql_parts)
        
        # Add ordering
        cql_query += " ORDER BY lastModified DESC"
        
        logger.debug(f"Built CQL query: {cql_query}")
        return cql_query
    
    def _format_confluence_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single Confluence search result"""
        
        # Extract content snippet
        excerpt = result.get("excerpt", "")
        if not excerpt:
            # Try to get content from body
            body = result.get("body", {})
            if isinstance(body, dict):
                excerpt = body.get("view", {}).get("value", "")[:300]
        
        # Clean HTML from excerpt
        clean_excerpt = re.sub(r'<[^>]+>', '', excerpt).strip()
        
        formatted_result = {
            "id": result.get("id"),
            "title": result.get("title", "Untitled"),
            "type": result.get("type", "page"),
            "snippet": clean_excerpt[:500] if clean_excerpt else "No content preview available",
            "url": f"{settings.confluence_url}/wiki{result.get('_links', {}).get('webui', '')}",
            "space": {
                "key": result.get("space", {}).get("key"),
                "name": result.get("space", {}).get("name")
            },
            "created": result.get("history", {}).get("createdDate"),
            "modified": result.get("version", {}).get("when"),
            "author": result.get("version", {}).get("by", {}).get("displayName"),
            "source": "confluence",
            "relevance": self._calculate_relevance(result)
        }
        
        return formatted_result
    
    def _extract_page_content(self, page: Dict[str, Any]) -> str:
        """Extract readable content from Confluence page"""
        
        # Try to get storage format first
        body = page.get("body", {})
        
        if "storage" in body:
            content = body["storage"].get("value", "")
        elif "view" in body:
            content = body["view"].get("value", "")
        else:
            content = ""
        
        # Clean HTML tags for better readability
        clean_content = re.sub(r'<[^>]+>', ' ', content)
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
        
        return clean_content
    
    def _calculate_relevance(self, result: Dict[str, Any]) -> float:
        """Calculate relevance score for search result"""
        
        score = 0.5  # Base score
        
        # Boost score based on result type
        if result.get("type") == "page":
            score += 0.2
        
        # Boost recent content
        version = result.get("version", {})
        if version.get("when"):
            # Simple recency boost (can be improved)
            score += 0.1
        
        # Boost if title contains search terms (simplified)
        title = result.get("title", "").lower()
        if any(term in title for term in ["error", "troubleshoot", "issue", "problem"]):
            score += 0.2
        
        return min(score, 1.0)
    
    async def run_server(self):
        """Run the MCP server"""
        try:
            logger.info(f"Starting Confluence MCP server on port {settings.confluence_mcp_port}")
            await self.mcp.run(transport="stdio")
        except Exception as e:
            logger.error(f"Error running Confluence MCP server: {str(e)}")


# Server instance
confluence_server = ConfluenceMCPServer()


if __name__ == "__main__":
    asyncio.run(confluence_server.run_server())