"""
Query processor for orchestrating searches across multiple MCP servers
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
import aiohttp
import json
from datetime import datetime
from config import settings, SearchSource

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Orchestrates search queries across multiple MCP servers"""
    
    def __init__(self):
        self.mcp_clients = {}
        self.source_status = {}
        
    async def initialize(self):
        """Initialize MCP client connections"""
        try:
            # Initialize connections to each MCP server
            mcp_servers = {
                SearchSource.CONFLUENCE: f"http://localhost:{settings.confluence_mcp_port}",
                SearchSource.TEAMS: f"http://localhost:{settings.teams_mcp_port}",
                SearchSource.OUTLOOK: f"http://localhost:{settings.outlook_mcp_port}",
                SearchSource.LOCAL_DISK: f"http://localhost:{settings.local_disk_mcp_port}"
            }
            
            for source, url in mcp_servers.items():
                try:
                    # Create MCP client connection (simplified - actual MCP client would be more complex)
                    self.mcp_clients[source] = MCPClient(source, url)
                    self.source_status[source] = "active"
                    logger.info(f"Initialized MCP client for {source}")
                except Exception as e:
                    logger.error(f"Failed to initialize {source} MCP client: {str(e)}")
                    self.source_status[source] = "inactive"
            
            logger.info(f"Query processor initialized with {len(self.mcp_clients)} active sources")
            
        except Exception as e:
            logger.error(f"Failed to initialize query processor: {str(e)}")
    
    async def execute_parallel_search(
        self, 
        expanded_query: Dict[str, Any], 
        query_id: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Execute searches across all available sources in parallel
        
        Args:
            expanded_query: AI-expanded query with search terms and strategies
            query_id: Unique identifier for this search session
            
        Returns:
            Dictionary with results from each source
        """
        logger.info(f"Starting parallel search for query: {query_id}")
        
        # Create search tasks for each active source
        search_tasks = {}
        
        for source, client in self.mcp_clients.items():
            if self.source_status.get(source) == "active":
                task = asyncio.create_task(
                    self._search_source(source, client, expanded_query, query_id)
                )
                search_tasks[source] = task
        
        # Execute all searches concurrently with timeout
        results = {}
        try:
            # Wait for all searches with timeout
            completed_tasks = await asyncio.wait_for(
                asyncio.gather(*search_tasks.values(), return_exceptions=True),
                timeout=settings.search_timeout_seconds
            )
            
            # Process results
            for source, result in zip(search_tasks.keys(), completed_tasks):
                if isinstance(result, Exception):
                    logger.error(f"Search failed for {source}: {str(result)}")
                    results[source] = [{"error": str(result)}]
                    self.source_status[source] = "error"
                else:
                    results[source] = result or []
                    logger.info(f"Found {len(results[source])} results from {source}")
                    
        except asyncio.TimeoutError:
            logger.warning(f"Search timeout after {settings.search_timeout_seconds}s")
            # Handle partial results
            for source, task in search_tasks.items():
                if task.done():
                    try:
                        results[source] = await task
                    except Exception as e:
                        results[source] = [{"error": str(e)}]
                else:
                    results[source] = [{"error": "Search timeout"}]
                    task.cancel()
        
        logger.info(f"Parallel search completed for {query_id}. Sources: {list(results.keys())}")
        return results
    
    async def _search_source(
        self, 
        source: str, 
        client: 'MCPClient', 
        expanded_query: Dict[str, Any], 
        query_id: str
    ) -> List[Dict[str, Any]]:
        """Search a specific source using its MCP client"""
        
        try:
            logger.debug(f"Searching {source} for query {query_id}")
            
            # Extract search strategy for this source
            search_strategy = expanded_query.get("search_strategy", {})
            source_strategy = search_strategy.get(source, "")
            
            # Get search terms
            all_terms = expanded_query.get("all_terms", [])
            primary_keywords = expanded_query.get("primary_keywords", [])
            
            # Create search query based on source type
            if source == SearchSource.CONFLUENCE:
                results = await self._search_confluence(
                    client, all_terms, primary_keywords, source_strategy
                )
            elif source == SearchSource.TEAMS:
                results = await self._search_teams(
                    client, all_terms, primary_keywords, source_strategy
                )
            elif source == SearchSource.OUTLOOK:
                results = await self._search_outlook(
                    client, all_terms, primary_keywords, source_strategy
                )
            elif source == SearchSource.LOCAL_DISK:
                results = await self._search_local_disk(
                    client, all_terms, primary_keywords, source_strategy
                )
            else:
                results = []
            
            # Add metadata to results
            for result in results:
                result["query_id"] = query_id
                result["searched_at"] = datetime.utcnow().isoformat()
                if "source" not in result:
                    result["source"] = source
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching {source}: {str(e)}")
            return [{"error": str(e), "source": source}]
    
    async def _search_confluence(
        self, 
        client: 'MCPClient', 
        all_terms: List[str], 
        primary_keywords: List[str],
        strategy: str
    ) -> List[Dict[str, Any]]:
        """Search Confluence using MCP client"""
        
        # Build search query from terms
        query = " OR ".join(primary_keywords[:5]) if primary_keywords else " ".join(all_terms[:10])
        
        # Call Confluence MCP server
        results = await client.call_tool(
            "search_confluence",
            {
                "query": query,
                "limit": settings.max_search_results
            }
        )
        
        return results or []
    
    async def _search_teams(
        self,
        client: 'MCPClient',
        all_terms: List[str],
        primary_keywords: List[str],
        strategy: str
    ) -> List[Dict[str, Any]]:
        """Search Teams using MCP client"""
        
        query = " ".join(primary_keywords[:5]) if primary_keywords else " ".join(all_terms[:10])
        
        results = await client.call_tool(
            "search_teams_messages",
            {
                "query": query,
                "limit": settings.max_search_results
            }
        )
        
        return results or []
    
    async def _search_outlook(
        self,
        client: 'MCPClient',
        all_terms: List[str],
        primary_keywords: List[str],
        strategy: str
    ) -> List[Dict[str, Any]]:
        """Search Outlook using MCP client"""
        
        query = " ".join(primary_keywords[:5]) if primary_keywords else " ".join(all_terms[:10])
        
        results = await client.call_tool(
            "search_outlook_emails",
            {
                "query": query,
                "limit": settings.max_search_results
            }
        )
        
        return results or []
    
    async def _search_local_disk(
        self,
        client: 'MCPClient',
        all_terms: List[str],
        primary_keywords: List[str],
        strategy: str
    ) -> List[Dict[str, Any]]:
        """Search local disk using MCP client"""
        
        results = await client.call_tool(
            "search_local_files",
            {
                "terms": all_terms,
                "primary_terms": primary_keywords,
                "file_extensions": settings.file_extensions_list,
                "search_paths": settings.search_paths_list,
                "limit": settings.max_search_results
            }
        )
        
        return results or []
    
    async def check_sources_status(self) -> List[Dict[str, Any]]:
        """Check the status of all search sources"""
        
        sources_status = []
        
        for source in SearchSource.all():
            status = self.source_status.get(source, "unknown")
            client = self.mcp_clients.get(source)
            
            source_info = {
                "name": source,
                "display_name": SearchSource.get_display_name(source),
                "status": status,
                "available": client is not None,
                "last_checked": datetime.utcnow().isoformat()
            }
            
            # Try to ping the source
            if client:
                try:
                    health_status = await client.health_check()
                    source_info["status"] = "active" if health_status else "inactive"
                except Exception:
                    source_info["status"] = "error"
            
            sources_status.append(source_info)
        
        return sources_status
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up query processor")
        
        for source, client in self.mcp_clients.items():
            try:
                await client.close()
            except Exception as e:
                logger.error(f"Error closing {source} client: {str(e)}")
        
        self.mcp_clients.clear()
        self.source_status.clear()


class MCPClient:
    """Simplified MCP client for communication with MCP servers"""
    
    def __init__(self, source: str, base_url: str):
        self.source = source
        self.base_url = base_url
        self.session = None
        
    async def _get_session(self):
        """Get or create HTTP session"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call an MCP tool"""
        
        try:
            session = await self._get_session()
            
            # Create MCP request
            mcp_request = {
                "jsonrpc": "2.0",
                "id": f"req_{datetime.now().timestamp()}",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": parameters
                }
            }
            
            # Send request to MCP server
            async with session.post(
                f"{self.base_url}/mcp",
                json=mcp_request,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    if "result" in data:
                        return data["result"].get("content", [])
                    elif "error" in data:
                        logger.error(f"MCP error from {self.source}: {data['error']}")
                        return [{"error": data["error"]["message"]}]
                    else:
                        return []
                else:
                    error_text = await response.text()
                    logger.error(f"HTTP error from {self.source}: {response.status} - {error_text}")
                    return [{"error": f"HTTP {response.status}: {error_text}"}]
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout calling {self.source} MCP server")
            return [{"error": "Request timeout"}]
        except Exception as e:
            logger.error(f"Error calling {self.source} MCP server: {str(e)}")
            return [{"error": str(e)}]
    
    async def health_check(self) -> bool:
        """Check if MCP server is healthy"""
        try:
            session = await self._get_session()
            
            async with session.get(
                f"{self.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
                
        except Exception:
            return False
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None