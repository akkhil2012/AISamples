import os
from textwrap import dedent
from typing import List
from google import genai
from google.genai import types

from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

from deep_knowledge_creator.neo4j_populator import Neo4jPopulator

load_dotenv(find_dotenv())


class CypherQueries(BaseModel):
    queries: List[str]

class ConvertEntityRelationJsonToCypher:
    def __init__(self):
        self.populator = Neo4jPopulator("bolt://localhost:7687", "neo4j", "neo4j@123")

    async def extract_insert_cypher_script(self):
        system_prompt = dedent("""\
        You are an expert in converting given JSON data to Neo4j Cypher Graph code! The Graph has many 
        nodes & relations generate cypher queries accordingly. Generate all queries which will cover the 
        entire data provided to you. never leave the task in middle or never generate half completed queries
        
        Always give the script as Example:
        create
            (a:Entity {name: "physics"})
            (a:Entity {name: "mechanics"})
        
        Output Mandatory Rules:
        - Output only the Cypher query—no backticks, no explanations, no verbose text.
        - Always filter on relation_type if a semantic type is specified in the question.
        - Use toLower() for case-insensitive matching if needed.
        - Never include ``` in and other verbose in the output. Just give the final cypher statement in the output. 
        """)

        with open("test_output/extraction_results.json", "r") as f:
            entity_relations_json = f.read()
            print(entity_relations_json)

        user_query = dedent(f"""\
        given the data, your task is to generate the cypher query and hence a respond accordingly.
        -----------------------------------------------------------------------------------------
        {entity_relations_json}
        - Never include ``` in and other verbose in the output. Just give the final cypher statement in the output.
        IMPORTANT: always give your cypher script with CREATE followed by queries
        """)

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_query,
            config= types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
                system_instruction= system_prompt,
                response_schema=CypherQueries,
                temperature=0.3
            )
        )

        print(response.text)
        return response.text

    async def extract_search_cypher_script(self):
        system_prompt = dedent("""\
        You are an expert in converting given JSON data to Neo4j Cypher Search code! The Search has many 
        nodes & relations generate cypher queries accordingly. Generate search queries which will cover the 
        entire data provided to you. never leave the task in middle or never generate half completed queries
        
        Always give the script as Example:
            MATCH (p:Person)-[r:WORKS_FOR]->(c:Company)
            RETURN p.name, c.name, r.since
            
            MATCH (p:Person {name: "John"})-[r]-(other)
            RETURN type(r) as relationship_type, other
            
            MATCH (p:Person)-[:WORKS_FOR]->(c:Company)<-[:WORKS_FOR]-(colleague:Person)
            WHERE p.name = "John"
            RETURN colleague.name as colleagues
        
        Output Mandatory Rules:
        - Output only the Cypher search query—no backticks, no explanations, no verbose text.
        - Always filter on relation_type if a semantic type is specified in the question.
        - Use toLower() for case-insensitive matching if needed.
        - Never include ``` in and other verbose in the output. Just give the final cypher statement in the output. 
        """)

        with open("test_output/extraction_results.json", "r") as f:
            entity_relations_json = f.read()
            print(entity_relations_json)

        user_query = dedent(f"""\
        given the data, your task is to generate the cypher search query and hence a respond accordingly.
        -----------------------------------------------------------------------------------------
        {entity_relations_json}
        - Never include ``` in and other verbose in the output. Just give the final cypher statement in the output.
        IMPORTANT: always give your cypher search query.
        """)

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_query,
            config= types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
                system_instruction= system_prompt,
                response_schema=CypherQueries,
                temperature=0.3
            )
        )

        return response.text


    async def populate_graph(self, cypher_query: str):
        await self.populator.populate_graph(cypher_query)
