import asyncio

from deep_knowledge_creator.entity_relation_json_to_cypher_query import ConvertEntityRelationJsonToCypher
from deep_knowledge_creator.text_to_entity_level_graph_converter import content_extract
import langextract as lextract

async def main(input_text: str):
    result = await content_extract(input_text=input_text)
    # Save the results to a JSONL file
    lextract.io.save_annotated_documents([result], output_name="extraction_results.json")
    o = ConvertEntityRelationJsonToCypher()
    cypher_script = await o.extract_search_cypher_script()
    print(cypher_script)


if __name__ == "__main__":
    asyncio.run(main("What is the spectrum of antimicrobial activity of amoxicillin, and what are the evidence-based "
                     "indications for its clinical use across different patient populations?"))