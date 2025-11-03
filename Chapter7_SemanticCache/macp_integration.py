
# mcp_integration.py - Advanced MCP Server Integration Examples

import asyncio
import json
import httpx
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

@dataclass
class MCPTool:
    """Represents an MCP tool with its configuration"""
    name: str
    description: str
    endpoint: str
    parameters: Dict[str, Any]
    auth_config: Optional[Dict[str, str]] = None

class MCPServerInterface(ABC):
    """Abstract interface for MCP server implementations"""

    @abstractmethod
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool with given parameters"""
        pass

    @abstractmethod
    async def list_tools(self) -> List[MCPTool]:
        """List available tools"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the MCP server is healthy"""
        pass

class PropertyDataMCPServer(MCPServerInterface):
    """
    Example MCP Server for proprietary data sources
    This would connect to your internal APIs, databases, or data lakes
    """

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0
        )

        # Define available tools
        self.tools = {
            "search_documents": MCPTool(
                name="search_documents",
                description="Search internal document repository",
                endpoint=f"{base_url}/api/documents/search",
                parameters={
                    "query": {"type": "string", "required": True},
                    "limit": {"type": "integer", "default": 10},
                    "filters": {"type": "object", "required": False}
                }
            ),
            "get_customer_data": MCPTool(
                name="get_customer_data",
                description="Retrieve customer information from CRM",
                endpoint=f"{base_url}/api/customers",
                parameters={
                    "customer_id": {"type": "string", "required": True},
                    "include_history": {"type": "boolean", "default": False}
                }
            ),
            "analyze_sales_data": MCPTool(
                name="analyze_sales_data",
                description="Analyze sales performance data",
                endpoint=f"{base_url}/api/analytics/sales",
                parameters={
                    "start_date": {"type": "string", "required": True},
                    "end_date": {"type": "string", "required": True},
                    "region": {"type": "string", "required": False}
                }
            )
        }

    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a proprietary data tool"""
        tool = self.tools.get(tool_name)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found"}

        try:
            # Validate parameters
            validated_params = self._validate_parameters(tool, parameters)

            # Make API call to proprietary system
            response = await self.client.post(
                tool.endpoint,
                json=validated_params
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "data": data,
                    "tool": tool_name,
                    "metadata": {
                        "response_time": response.elapsed.total_seconds(),
                        "status_code": response.status_code,
                        "data_source": "proprietary"
                    }
                }
            else:
                return {
                    "error": f"API call failed with status {response.status_code}",
                    "details": response.text
                }

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return {"error": str(e)}

    async def list_tools(self) -> List[MCPTool]:
        """List available proprietary tools"""
        return list(self.tools.values())

    async def health_check(self) -> bool:
        """Check proprietary system health"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False

    def _validate_parameters(self, tool: MCPTool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool parameters"""
        validated = {}

        for param_name, param_config in tool.parameters.items():
            if param_config.get("required", False) and param_name not in parameters:
                raise ValueError(f"Required parameter '{param_name}' missing")

            if param_name in parameters:
                validated[param_name] = parameters[param_name]
            elif "default" in param_config:
                validated[param_name] = param_config["default"]

        return validated

class ExternalAPIMCPServer(MCPServerInterface):
    """
    Example MCP Server for external APIs (e.g., weather, news, stock data)
    """

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

        self.tools = {
            "get_weather": MCPTool(
                name="get_weather",
                description="Get weather information for a location",
                endpoint="https://api.openweathermap.org/data/2.5/weather",
                parameters={
                    "location": {"type": "string", "required": True},
                    "units": {"type": "string", "default": "metric"}
                }
            ),
            "get_news": MCPTool(
                name="get_news",
                description="Get latest news articles",
                endpoint="https://newsapi.org/v2/top-headlines",
                parameters={
                    "category": {"type": "string", "default": "general"},
                    "country": {"type": "string", "default": "us"},
                    "limit": {"type": "integer", "default": 5}
                }
            ),
            "get_stock_data": MCPTool(
                name="get_stock_data",
                description="Get stock price information",
                endpoint="https://api.finnhub.io/api/v1/quote",
                parameters={
                    "symbol": {"type": "string", "required": True}
                }
            )
        }

    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call external API tool"""
        # Implementation would depend on specific API requirements
        # This is a mock implementation

        tool = self.tools.get(tool_name)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found"}

        # Mock responses for demonstration
        mock_responses = {
            "get_weather": {
                "temperature": "22°C",
                "condition": "Partly Cloudy",
                "humidity": "65%",
                "location": parameters.get("location", "Unknown")
            },
            "get_news": {
                "articles": [
                    {"title": "Breaking News 1", "source": "News Source 1"},
                    {"title": "Breaking News 2", "source": "News Source 2"}
                ],
                "category": parameters.get("category", "general")
            },
            "get_stock_data": {
                "symbol": parameters.get("symbol", "UNKNOWN"),
                "price": 150.25,
                "change": "+2.15",
                "change_percent": "+1.45%"
            }
        }

        return {
            "success": True,
            "data": mock_responses.get(tool_name, {}),
            "tool": tool_name,
            "metadata": {
                "source": "external_api",
                "mock": True  # Remove in real implementation
            }
        }

    async def list_tools(self) -> List[MCPTool]:
        """List available external API tools"""
        return list(self.tools.values())

    async def health_check(self) -> bool:
        """Check external API health"""
        return True  # Mock implementation

class MCPServerManager:
    """
    Enhanced MCP Server Manager with multiple server support
    """

    def __init__(self):
        self.servers: Dict[str, MCPServerInterface] = {}
        self.tool_routing: Dict[str, str] = {}  # tool_name -> server_name

    def register_server(self, name: str, server: MCPServerInterface):
        """Register an MCP server"""
        self.servers[name] = server
        logger.info(f"Registered MCP server: {name}")

    async def initialize_servers(self):
        """Initialize all registered servers and build tool routing"""
        for name, server in self.servers.items():
            try:
                tools = await server.list_tools()
                for tool in tools:
                    self.tool_routing[tool.name] = name
                logger.info(f"Initialized server '{name}' with {len(tools)} tools")
            except Exception as e:
                logger.error(f"Failed to initialize server '{name}': {e}")

    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Route tool call to appropriate server"""
        server_name = self.tool_routing.get(tool_name)
        if not server_name:
            return {"error": f"No server found for tool '{tool_name}'"}

        server = self.servers.get(server_name)
        if not server:
            return {"error": f"Server '{server_name}' not available"}

        try:
            result = await server.call_tool(tool_name, parameters)
            # Add server information to metadata
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"]["server"] = server_name
            return result
        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}' on server '{server_name}': {e}")
            return {"error": str(e)}

    async def list_all_tools(self) -> Dict[str, List[MCPTool]]:
        """List tools from all servers"""
        all_tools = {}
        for name, server in self.servers.items():
            try:
                tools = await server.list_tools()
                all_tools[name] = tools
            except Exception as e:
                logger.error(f"Error listing tools from server '{name}': {e}")
                all_tools[name] = []
        return all_tools

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all registered servers"""
        health_status = {}
        for name, server in self.servers.items():
            try:
                health_status[name] = await server.health_check()
            except Exception as e:
                logger.error(f"Health check failed for server '{name}': {e}")
                health_status[name] = False
        return health_status

# Example usage and setup
async def setup_mcp_servers():
    """Setup example MCP servers"""
    manager = MCPServerManager()

    # Register proprietary data server
    proprietary_server = PropertyDataMCPServer(
        base_url="https://your-internal-api.company.com",
        api_key="your-api-key"
    )
    manager.register_server("proprietary", proprietary_server)

    # Register external API server
    external_server = ExternalAPIMCPServer()
    manager.register_server("external", external_server)

    # Initialize all servers
    await manager.initialize_servers()

    return manager

# Integration with FastAPI backend
class EnhancedMCPIntegration:
    """Enhanced MCP integration for the FastAPI backend"""

    def __init__(self):
        self.manager = None
        self.initialized = False

    async def initialize(self):
        """Initialize MCP integration"""
        if not self.initialized:
            self.manager = await setup_mcp_servers()
            self.initialized = True
            logger.info("MCP integration initialized")

    async def process_request(self, service_id: str, prompt: str) -> Dict[str, Any]:
        """Process request through MCP servers"""
        if not self.initialized:
            await self.initialize()

        # Map service_id to tool_name and extract parameters from prompt
        tool_mapping = {
            "semantic_search": ("search_documents", {"query": prompt}),
            "text_analysis": ("analyze_text", {"text": prompt}),
            "document_qa": ("search_documents", {"query": prompt}),
            "data_extraction": ("extract_data", {"content": prompt}),
            "sentiment_analysis": ("analyze_sentiment", {"text": prompt}),
            "summarization": ("summarize_text", {"text": prompt}),
            "translation": ("translate_text", {"text": prompt, "target_lang": "en"}),
            "content_generation": ("generate_content", {"prompt": prompt})
        }

        tool_name, parameters = tool_mapping.get(service_id, (service_id, {"input": prompt}))

        # Call the appropriate tool
        result = await self.manager.call_tool(tool_name, parameters)

        return result

    async def get_server_status(self) -> Dict[str, Any]:
        """Get status of all MCP servers"""
        if not self.initialized:
            return {"status": "not_initialized"}

        health_status = await self.manager.health_check_all()
        tools = await self.manager.list_all_tools()

        return {
            "status": "initialized",
            "servers": health_status,
            "tools": {server: len(tool_list) for server, tool_list in tools.items()}
        }

# Example configuration for production
MCP_CONFIG = {
    "proprietary_servers": [
        {
            "name": "crm_server",
            "type": "PropertyDataMCPServer",
            "config": {
                "base_url": "https://crm-api.company.com",
                "api_key": "${CRM_API_KEY}"
            }
        },
        {
            "name": "data_lake_server",
            "type": "PropertyDataMCPServer", 
            "config": {
                "base_url": "https://data-lake-api.company.com",
                "api_key": "${DATA_LAKE_API_KEY}"
            }
        }
    ],
    "external_servers": [
        {
            "name": "weather_server",
            "type": "ExternalAPIMCPServer",
            "config": {
                "api_keys": {
                    "weather": "${WEATHER_API_KEY}",
                    "news": "${NEWS_API_KEY}",
                    "stocks": "${STOCKS_API_KEY}"
                }
            }
        }
    ]
}
