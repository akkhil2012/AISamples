"""
MCP Server for Microsoft Teams Search Integration
"""
import asyncio
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP
import aiohttp
import msal
from config import settings
import logging
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TeamsMCPServer:
    """MCP Server for Microsoft Teams search operations"""
    
    def __init__(self):
        self.mcp = FastMCP("Teams Search Server")
        self.access_token = None
        self.token_expires_at = None
        self._setup_tools()
        self._setup_resources()
        
    async def _get_access_token(self) -> Optional[str]:
        """Get Microsoft Graph access token"""
        
        # Check if current token is still valid
        if (self.access_token and self.token_expires_at and 
            datetime.now() < self.token_expires_at - timedelta(minutes=5)):
            return self.access_token
        
        try:
            # Create MSAL app
            app = msal.ConfidentialClientApplication(
                settings.microsoft_client_id,
                authority=settings.microsoft_authority_url,
                client_credential=settings.microsoft_client_secret,
            )
            
            # Acquire token for Microsoft Graph
            result = app.acquire_token_for_client(
                scopes=["https://graph.microsoft.com/.default"]
            )
            
            if "access_token" in result:
                self.access_token = result["access_token"]
                # Set expiration time (default 1 hour)
                expires_in = result.get("expires_in", 3600)
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                logger.info("Successfully acquired Microsoft Graph access token")
                return self.access_token
            else:
                logger.error(f"Failed to acquire token: {result.get('error_description')}")
                return None
                
        except Exception as e:
            logger.error(f"Error acquiring Microsoft Graph token: {str(e)}")
            return None
    
    def _setup_tools(self):
        """Setup MCP tools for Teams operations"""
        
        @self.mcp.tool()
        async def search_teams_messages(
            query: str,
            team_id: str = None,
            channel_id: str = None,
            from_date: str = None,
            limit: int = 25
        ) -> List[Dict[str, Any]]:
            """
            Search Microsoft Teams messages
            
            Args:
                query: Search query string
                team_id: Specific team to search (optional)
                channel_id: Specific channel to search (optional)
                from_date: Search from this date (YYYY-MM-DD format)
                limit: Maximum number of results to return
                
            Returns:
                List of Teams messages matching the query
            """
            access_token = await self._get_access_token()
            if not access_token:
                return [{"error": "Failed to authenticate with Microsoft Graph"}]
            
            try:
                # Build search request
                search_results = await self._execute_teams_search(
                    access_token, query, team_id, channel_id, from_date, limit
                )
                
                logger.info(f"Found {len(search_results)} Teams messages for query: {query}")
                return search_results
                
            except Exception as e:
                logger.error(f"Teams search error: {str(e)}")
                return [{"error": f"Search failed: {str(e)}"}]
        
        @self.mcp.tool()
        async def get_teams_list() -> List[Dict[str, Any]]:
            """
            Get list of available Teams
            
            Returns:
                List of Teams the user has access to
            """
            access_token = await self._get_access_token()
            if not access_token:
                return [{"error": "Failed to authenticate with Microsoft Graph"}]
            
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json"
                    }
                    
                    # Get teams list
                    url = "https://graph.microsoft.com/v1.0/me/joinedTeams"
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            teams = []
                            
                            for team in data.get("value", []):
                                teams.append({
                                    "id": team.get("id"),
                                    "name": team.get("displayName"),
                                    "description": team.get("description", ""),
                                    "archived": team.get("isArchived", False),
                                    "created": team.get("createdDateTime"),
                                    "source": "teams"
                                })
                            
                            return teams
                        else:
                            error_msg = await response.text()
                            logger.error(f"Failed to get teams list: {error_msg}")
                            return [{"error": f"Failed to get teams: {error_msg}"}]
                            
            except Exception as e:
                logger.error(f"Error getting teams list: {str(e)}")
                return [{"error": f"Failed to get teams: {str(e)}"}]
        
        @self.mcp.tool()
        async def get_team_channels(team_id: str) -> List[Dict[str, Any]]:
            """
            Get channels for a specific team
            
            Args:
                team_id: ID of the team
                
            Returns:
                List of channels in the team
            """
            access_token = await self._get_access_token()
            if not access_token:
                return [{"error": "Failed to authenticate with Microsoft Graph"}]
            
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json"
                    }
                    
                    # Get channels for team
                    url = f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels"
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            channels = []
                            
                            for channel in data.get("value", []):
                                channels.append({
                                    "id": channel.get("id"),
                                    "name": channel.get("displayName"),
                                    "description": channel.get("description", ""),
                                    "type": channel.get("membershipType"),
                                    "created": channel.get("createdDateTime"),
                                    "webUrl": channel.get("webUrl"),
                                    "source": "teams"
                                })
                            
                            return channels
                        else:
                            error_msg = await response.text()
                            logger.error(f"Failed to get channels for team {team_id}: {error_msg}")
                            return [{"error": f"Failed to get channels: {error_msg}"}]
                            
            except Exception as e:
                logger.error(f"Error getting channels for team {team_id}: {str(e)}")
                return [{"error": f"Failed to get channels: {str(e)}"}]
        
        @self.mcp.tool()
        async def get_channel_messages(
            team_id: str,
            channel_id: str,
            limit: int = 50
        ) -> List[Dict[str, Any]]:
            """
            Get recent messages from a specific channel
            
            Args:
                team_id: ID of the team
                channel_id: ID of the channel
                limit: Maximum number of messages to return
                
            Returns:
                List of recent messages from the channel
            """
            access_token = await self._get_access_token()
            if not access_token:
                return [{"error": "Failed to authenticate with Microsoft Graph"}]
            
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json"
                    }
                    
                    # Get messages from channel
                    url = f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages"
                    params = {"$top": limit, "$orderby": "createdDateTime desc"}
                    
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            messages = []
                            
                            for message in data.get("value", []):
                                formatted_message = self._format_teams_message(message, team_id, channel_id)
                                messages.append(formatted_message)
                            
                            return messages
                        else:
                            error_msg = await response.text()
                            logger.error(f"Failed to get messages: {error_msg}")
                            return [{"error": f"Failed to get messages: {error_msg}"}]
                            
            except Exception as e:
                logger.error(f"Error getting channel messages: {str(e)}")
                return [{"error": f"Failed to get messages: {str(e)}"}]
    
    def _setup_resources(self):
        """Setup MCP resources for Teams data"""
        
        @self.mcp.resource("teams://my-teams")
        async def my_teams() -> str:
            """Get list of teams I'm a member of"""
            teams = await self.get_teams_list()
            return json.dumps(teams, indent=2)
        
        @self.mcp.resource("teams://recent-activity")
        async def recent_activity() -> str:
            """Get recent Teams activity"""
            # This is a simplified version - in practice you'd aggregate from multiple channels
            access_token = await self._get_access_token()
            if not access_token:
                return "Authentication failed"
            
            return json.dumps({
                "message": "Recent activity would be aggregated from multiple teams/channels",
                "status": "authenticated"
            }, indent=2)
    
    async def _execute_teams_search(
        self,
        access_token: str,
        query: str,
        team_id: str = None,
        channel_id: str = None,
        from_date: str = None,
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """Execute Microsoft Graph search for Teams messages"""
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                }
                
                # Build search request body
                search_request = {
                    "requests": [
                        {
                            "entityTypes": ["chatMessage"],
                            "query": {
                                "queryString": query
                            },
                            "from": 0,
                            "size": limit
                        }
                    ]
                }
                
                # Add filters if specified
                if team_id or channel_id or from_date:
                    filters = []
                    if team_id:
                        filters.append(f"parentId:{team_id}")
                    if from_date:
                        filters.append(f"createdDateTime>={from_date}")
                    
                    if filters:
                        search_request["requests"][0]["query"]["queryString"] += f" AND {' AND '.join(filters)}"
                
                # Execute search
                url = "https://graph.microsoft.com/v1.0/search/query"
                async with session.post(url, headers=headers, json=search_request) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        
                        for search_response in data.get("value", []):
                            for hit in search_response.get("hitsContainers", []):
                                for message_hit in hit.get("hits", []):
                                    formatted_message = self._format_search_result(message_hit)
                                    results.append(formatted_message)
                        
                        return results
                    else:
                        error_msg = await response.text()
                        logger.error(f"Teams search failed: {error_msg}")
                        return [{"error": f"Search failed: {error_msg}"}]
                        
        except Exception as e:
            logger.error(f"Error executing Teams search: {str(e)}")
            return [{"error": f"Search execution failed: {str(e)}"}]
    
    def _format_teams_message(self, message: Dict[str, Any], team_id: str, channel_id: str) -> Dict[str, Any]:
        """Format a Teams message for consistent output"""
        
        # Extract message content
        body = message.get("body", {})
        content = body.get("content", "") if body else ""
        
        # Clean HTML from content
        import re
        clean_content = re.sub(r'<[^>]+>', '', content).strip()
        
        # Extract author information
        from_user = message.get("from", {}).get("user", {})
        
        formatted_message = {
            "id": message.get("id"),
            "content": clean_content,
            "snippet": clean_content[:300] if clean_content else "No content",
            "created": message.get("createdDateTime"),
            "modified": message.get("lastModifiedDateTime"),
            "author": {
                "name": from_user.get("displayName", "Unknown"),
                "email": from_user.get("userPrincipalName", "")
            },
            "team_id": team_id,
            "channel_id": channel_id,
            "message_type": message.get("messageType", "message"),
            "importance": message.get("importance", "normal"),
            "webUrl": message.get("webUrl", ""),
            "source": "teams",
            "relevance": self._calculate_message_relevance(message, clean_content)
        }
        
        return formatted_message
    
    def _format_search_result(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        """Format a search result hit"""
        
        resource = hit.get("resource", {})
        summary = hit.get("summary", "")
        
        # Extract content
        body = resource.get("body", {})
        content = body.get("content", "") if body else ""
        
        # Clean content
        import re
        clean_content = re.sub(r'<[^>]+>', '', content).strip()
        
        formatted_result = {
            "id": resource.get("id"),
            "content": clean_content,
            "snippet": summary or clean_content[:300],
            "created": resource.get("createdDateTime"),
            "author": {
                "name": resource.get("from", {}).get("user", {}).get("displayName", "Unknown")
            },
            "webUrl": resource.get("webUrl", ""),
            "source": "teams",
            "relevance": hit.get("rank", 0.5) / 100.0  # Convert rank to 0-1 scale
        }
        
        return formatted_result
    
    def _calculate_message_relevance(self, message: Dict[str, Any], content: str) -> float:
        """Calculate relevance score for a Teams message"""
        
        score = 0.5  # Base score
        
        # Boost based on message importance
        importance = message.get("importance", "normal")
        if importance == "high":
            score += 0.2
        elif importance == "urgent":
            score += 0.3
        
        # Boost if content contains error-related terms
        error_terms = ["error", "issue", "problem", "bug", "failure", "exception", "crash"]
        content_lower = content.lower()
        matching_terms = sum(1 for term in error_terms if term in content_lower)
        score += min(matching_terms * 0.1, 0.3)
        
        # Boost recent messages
        created = message.get("createdDateTime")
        if created:
            # Simple recency boost (can be improved with actual date parsing)
            score += 0.1
        
        return min(score, 1.0)
    
    async def run_server(self):
        """Run the MCP server"""
        try:
            logger.info(f"Starting Teams MCP server on port {settings.teams_mcp_port}")
            await self.mcp.run(transport="stdio")
        except Exception as e:
            logger.error(f"Error running Teams MCP server: {str(e)}")


# Server instance
teams_server = TeamsMCPServer()


if __name__ == "__main__":
    asyncio.run(teams_server.run_server())