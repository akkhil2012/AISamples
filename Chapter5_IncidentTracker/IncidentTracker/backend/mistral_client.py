"""
Mistral AI client for query processing and expansion
"""
import asyncio
from typing import List, Dict, Any, Optional
from mistralai import Mistral
from config import settings
import logging

logger = logging.getLogger(__name__)


class MistralClient:
    """Client for interacting with Mistral AI for query processing"""
    
    def __init__(self):
        self.client = Mistral(api_key=settings.mistral_api_key)
        self.model = settings.mistral_model
        
    async def expand_query(self, error_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expand user query using Mistral AI for better search results
        
        Args:
            error_input: Dictionary containing error details
            
        Returns:
            Dictionary with expanded query terms and search strategy
        """
        try:
            # Create the expansion prompt
            prompt = self._create_expansion_prompt(error_input)
            
            # Call Mistral API
            response = await self._call_mistral_async(prompt)
            
            # Parse and structure the response
            expanded_query = self._parse_expansion_response(response, error_input)
            
            logger.info(f"Successfully expanded query for error: {error_input.get('error_code', 'Unknown')}")
            return expanded_query
            
        except Exception as e:
            logger.error(f"Error expanding query: {str(e)}")
            # Fallback to basic expansion
            return self._create_fallback_expansion(error_input)
    
    def _create_expansion_prompt(self, error_input: Dict[str, Any]) -> str:
        """Create a detailed prompt for query expansion"""
        
        prompt = f"""
You are an expert system administrator and error resolution specialist. Your task is to analyze an error report and expand the search query to find the most relevant solutions.

**Error Details:**
- Severity Level: {error_input.get('severity_level', 'Unknown')}
- Error Code: {error_input.get('error_code', 'Not provided')}
- Error Description: {error_input.get('error_description', 'Not provided')}
- Application Name: {error_input.get('application_name', 'Not provided')}
- Environment: {error_input.get('environment', 'Unknown')}
- Applicable Pool: {error_input.get('applicable_pool', 'Not provided')}

**Your Task:**
1. Generate comprehensive search terms that would help find relevant documentation, discussions, and solutions
2. Include technical synonyms, related error codes, and common troubleshooting terms
3. Consider different ways this error might be discussed (formal documentation vs. informal chat)
4. Include terms for different data sources (Confluence pages, Teams chats, email threads, local files)

**Output Format (JSON):**
{{
    "primary_keywords": ["list of main search terms"],
    "technical_terms": ["technical synonyms and related terms"],
    "contextual_terms": ["environment and application specific terms"],
    "troubleshooting_terms": ["common resolution and debugging terms"],
    "search_strategy": {{
        "confluence": "specific search approach for Confluence",
        "teams": "specific search approach for Teams",
        "outlook": "specific search approach for Outlook",
        "local_files": "specific search approach for local files"
    }},
    "priority_sources": ["ranked list of most likely sources"],
    "expanded_description": "enhanced error description for better matching"
}}

Generate a comprehensive search expansion that maximizes the chance of finding relevant solutions.
"""
        return prompt
    
    async def _call_mistral_async(self, prompt: str) -> str:
        """Make async call to Mistral API"""
        try:
            response = await asyncio.to_thread(
                self.client.chat.complete,
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert system administrator specialized in error resolution and technical search optimization."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Mistral API call failed: {str(e)}")
            raise
    
    def _parse_expansion_response(self, response: str, original_input: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Mistral response and structure the expanded query"""
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_response = json.loads(json_match.group())
            else:
                # Fallback parsing
                parsed_response = self._extract_terms_from_text(response)
            
            # Ensure all required fields exist
            expanded_query = {
                "original_input": original_input,
                "primary_keywords": parsed_response.get("primary_keywords", []),
                "technical_terms": parsed_response.get("technical_terms", []),
                "contextual_terms": parsed_response.get("contextual_terms", []),
                "troubleshooting_terms": parsed_response.get("troubleshooting_terms", []),
                "search_strategy": parsed_response.get("search_strategy", {}),
                "priority_sources": parsed_response.get("priority_sources", ["confluence", "teams", "outlook", "local_files"]),
                "expanded_description": parsed_response.get("expanded_description", original_input.get("error_description", "")),
                "all_terms": []
            }
            
            # Combine all terms for easy searching
            expanded_query["all_terms"] = (
                expanded_query["primary_keywords"] +
                expanded_query["technical_terms"] +
                expanded_query["contextual_terms"] +
                expanded_query["troubleshooting_terms"]
            )
            
            return expanded_query
            
        except Exception as e:
            logger.error(f"Error parsing Mistral response: {str(e)}")
            return self._create_fallback_expansion(original_input)
    
    def _extract_terms_from_text(self, text: str) -> Dict[str, Any]:
        """Fallback method to extract terms from unstructured text"""
        import re
        
        # Basic extraction logic
        terms = re.findall(r'\b[A-Za-z][A-Za-z0-9_-]+\b', text)
        
        return {
            "primary_keywords": terms[:10],
            "technical_terms": [],
            "contextual_terms": [],
            "troubleshooting_terms": ["troubleshoot", "debug", "fix", "resolve", "solution"],
            "search_strategy": {},
            "priority_sources": ["confluence", "teams", "outlook", "local_files"],
            "expanded_description": text[:500]
        }
    
    def _create_fallback_expansion(self, error_input: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic expansion if AI fails"""
        
        error_code = error_input.get("error_code", "")
        error_description = error_input.get("error_description", "")
        application = error_input.get("application_name", "")
        environment = error_input.get("environment", "")
        
        # Basic keyword extraction
        basic_terms = []
        if error_code:
            basic_terms.append(error_code)
        if application:
            basic_terms.append(application)
        if environment:
            basic_terms.append(environment)
        
        # Add common troubleshooting terms
        troubleshooting_terms = [
            "error", "issue", "problem", "fix", "resolve", "solution",
            "troubleshoot", "debug", "workaround", "patch"
        ]
        
        return {
            "original_input": error_input,
            "primary_keywords": basic_terms,
            "technical_terms": [error_code] if error_code else [],
            "contextual_terms": [application, environment],
            "troubleshooting_terms": troubleshooting_terms,
            "search_strategy": {
                "confluence": f"Search for {error_code} OR {application} troubleshooting",
                "teams": f"Look for discussions about {error_code} or {application} issues",
                "outlook": f"Find email threads mentioning {error_code} resolution",
                "local_files": f"Search files for {error_code} runbooks or documentation"
            },
            "priority_sources": ["confluence", "teams", "outlook", "local_files"],
            "expanded_description": error_description,
            "all_terms": basic_terms + troubleshooting_terms
        }
    
    async def analyze_search_results(self, results: List[Dict[str, Any]], original_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze search results and provide recommendations
        
        Args:
            results: List of search results from all sources
            original_query: Original expanded query
            
        Returns:
            Analysis with top recommendations and insights
        """
        try:
            # Create analysis prompt
            prompt = self._create_analysis_prompt(results, original_query)
            
            # Call Mistral API
            response = await self._call_mistral_async(prompt)
            
            # Parse and return analysis
            return self._parse_analysis_response(response, results)
            
        except Exception as e:
            logger.error(f"Error analyzing search results: {str(e)}")
            return self._create_fallback_analysis(results)
    
    def _create_analysis_prompt(self, results: List[Dict[str, Any]], original_query: Dict[str, Any]) -> str:
        """Create prompt for analyzing search results"""
        
        # Summarize results for the prompt
        results_summary = []
        for i, result in enumerate(results[:20]):  # Limit to top 20 results
            results_summary.append({
                "index": i + 1,
                "source": result.get("source", "Unknown"),
                "title": result.get("title", "No title"),
                "snippet": result.get("snippet", "")[:200],  # Truncate snippet
                "relevance": result.get("relevance", 0.0)
            })
        
        prompt = f"""
You are an expert technical analyst specializing in error resolution. Analyze the following search results to identify the top 3 most promising solutions.

**Original Error:**
- Error Code: {original_query['original_input'].get('error_code', 'Unknown')}
- Description: {original_query['original_input'].get('error_description', 'Not provided')}
- Application: {original_query['original_input'].get('application_name', 'Unknown')}
- Environment: {original_query['original_input'].get('environment', 'Unknown')}

**Search Results:**
{str(results_summary)}

**Your Task:**
1. Identify the 3 most relevant and promising solutions
2. Rank them by likelihood of success and completeness
3. Provide reasoning for each recommendation
4. Identify any patterns or common themes

**Output Format (JSON):**
{{
    "top_recommendations": [
        {{
            "rank": 1,
            "source_index": "index from search results",
            "confidence_score": "0.0-1.0",
            "reasoning": "why this is the best solution",
            "action_required": "what the user should do",
            "estimated_resolution_time": "time estimate"
        }}
    ],
    "common_patterns": ["list of recurring themes or solutions"],
    "additional_investigation": ["areas that need more research"],
    "prevention_recommendations": ["how to prevent this error in future"],
    "overall_confidence": "0.0-1.0 confidence in finding a solution"
}}

Focus on actionable, practical solutions that directly address the reported error.
"""
        return prompt
    
    def _parse_analysis_response(self, response: str, original_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse the analysis response from Mistral"""
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                return self._create_fallback_analysis(original_results)
            
            # Enhance recommendations with original result data
            enhanced_recommendations = []
            for rec in analysis.get("top_recommendations", []):
                source_index = rec.get("source_index")
                if source_index and source_index <= len(original_results):
                    original_result = original_results[source_index - 1]
                    enhanced_rec = {**rec, **original_result}
                    enhanced_recommendations.append(enhanced_rec)
            
            analysis["top_recommendations"] = enhanced_recommendations
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing analysis response: {str(e)}")
            return self._create_fallback_analysis(original_results)
    
    def _create_fallback_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create fallback analysis if AI analysis fails"""
        
        # Sort results by relevance score
        sorted_results = sorted(results, key=lambda x: x.get("relevance", 0.0), reverse=True)
        
        top_recommendations = []
        for i, result in enumerate(sorted_results[:3]):
            top_recommendations.append({
                "rank": i + 1,
                "confidence_score": result.get("relevance", 0.5),
                "reasoning": f"High relevance score from {result.get('source', 'unknown')} source",
                "action_required": "Review the provided documentation or discussion",
                "estimated_resolution_time": "30-60 minutes",
                **result
            })
        
        return {
            "top_recommendations": top_recommendations,
            "common_patterns": ["Multiple sources available", "Documentation exists"],
            "additional_investigation": ["Check for recent updates", "Verify environment-specific factors"],
            "prevention_recommendations": ["Monitor application logs", "Update documentation"],
            "overall_confidence": 0.7 if results else 0.2
        }