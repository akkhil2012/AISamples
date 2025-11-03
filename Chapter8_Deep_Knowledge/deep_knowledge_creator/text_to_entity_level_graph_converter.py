import asyncio
import json
import os
from textwrap import dedent
import langextract as lextract
from dotenv import load_dotenv, find_dotenv
from typing import Any
import pandas as pd

from deep_knowledge_creator.entity_relation_json_to_cypher_query import ConvertEntityRelationJsonToCypher
from deep_knowledge_creator.ingest_to_qdrant import AsyncQdrantIngestion

load_dotenv(find_dotenv())
o = ConvertEntityRelationJsonToCypher()
qdrant = AsyncQdrantIngestion()

async def content_extract(input_text):
    # Entity-level graph extraction prompt for medical text
    # Medical entity-level graph extraction prompt for PubMedQA text
    prompt = dedent("""\
    Task Description
    You are a medical knowledge extraction specialist focused on building entity-level knowledge graphs from biomedical literature. Your task is to extract three specific components from medical text:
    
    entities -- All medical entities mentioned in the text (diseases, drugs, procedures, biomarkers, cell types, anatomical structures, etc.)
    relations -- All relationship types between entities (treats, causes, associated_with, increased_in, contraindicated_in, etc.)
    edges -- Specific entity-relationship-entity triples that represent factual medical relationships
    
    Component Categories:
    entities: Use for all medical terms as they appear in the text. Include diseases, conditions, drugs, biomarkers, cell types, anatomical structures, procedures, and measurements. Preserve exact medical terminology.
    
    relations: Use for all relationship types that connect entities. Include causative relationships (causes, leads_to), therapeutic relationships (treats, prevents), quantitative relationships (increased_in, decreased_in), associative relationships (associated_with, correlated_with), and contraindication relationships (contraindicated_in, avoided_in).
    
    edges: Use for specific factual relationships between two entities. Format as (entity1, relation, entity2). Only include relationships explicitly stated or strongly implied in the text.
    
    Critical Rules:
    - Use exact medical terminology as it appears in the text - do not paraphrase
    - Only extract relationships that are factually stated in the source text
    - Maintain medical accuracy and precision in all extractions
    - Focus on clinically relevant entities and relationships
    
    Professional Output Standards:
    All extracted entities must maintain medical terminology accuracy. Ensure that:
    - Medical terms are spelled correctly and consistently
    - Entity names match established medical nomenclature
    - Relationship types are medically appropriate
    - All extractions preserve the intended clinical meaning
    - Abbreviations are preserved as they appear in source text
    
    Required JSON Format
    Each final answer must be valid JSON with an array key "extractions". Each "extraction" is an object with:
    
    {
     "extraction_class": "entity" | "relation" | "edge",
     "extraction_text": "...",
     "attributes": {}
    }
    
    For entities: Include attributes like "type" (disease, drug, biomarker, etc.) and "scale" (entity)
    For relations: Include attributes like "relation_type" (causative, therapeutic, associative, etc.)
    For edges: Include attributes like "confidence" (high, medium, low) and "evidence_based" (true, false)""")

    # 2. Provide a high-quality example to guide the model for medical entity extraction
    examples = [
        lextract.data.ExampleData(
            text=(
                "ILC2s were increased in patients with CRSwNP and were associated with "
                "nasal polyps. ILC2s drive Th2 inflammation and are identified by "
                "CD45+ CD127+ CRTH2+ markers. Aspirin treats acute myocardial infarction "
                "but is contraindicated in patients with severe kidney disease."
            ),
            extractions=[
                lextract.data.Extraction(
                    extraction_class="entity",
                    extraction_text="ILC2s",
                    attributes={"type": "cell_type", "scale": "entity"},
                ),
                lextract.data.Extraction(
                    extraction_class="entity",
                    extraction_text="CRSwNP",
                    attributes={"type": "disease", "scale": "entity"},
                ),
                lextract.data.Extraction(
                    extraction_class="entity",
                    extraction_text="nasal polyps",
                    attributes={"type": "condition", "scale": "entity"},
                ),
                lextract.data.Extraction(
                    extraction_class="relation",
                    extraction_text="increased_in",
                    attributes={"relation_type": "quantitative_change"},
                ),
                lextract.data.Extraction(
                    extraction_class="relation",
                    extraction_text="associated_with",
                    attributes={"relation_type": "correlation"},
                ),
                lextract.data.Extraction(
                    extraction_class="edge",
                    extraction_text="(ILC2s, increased_in, CRSwNP)",
                    attributes={"confidence": "high", "evidence_based": "true"},
                ),
                lextract.data.Extraction(
                    extraction_class="edge",
                    extraction_text="(ILC2s, associated_with, nasal polyps)",
                    attributes={"confidence": "high", "evidence_based": "true"},
                ),
                lextract.data.Extraction(
                    extraction_class="edge",
                    extraction_text="(Aspirin, contraindicated_in, severe kidney disease)",
                    attributes={"confidence": "high", "safety_critical": "true"},
                ),
            ],
        )
    ]

    # 3. Run the extraction on your input text
    input_text = input_text.strip()
    _result: Any = lextract.extract(
        api_key=os.environ.get('GEMINI_API_KEY'),
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples,
        model_id="gemini-2.5-flash",
    )

    print(_result)
    return _result

async def save_and_visualize_content(result: Any, context:str, need_vis: bool = False):
    # Save the results to a JSONL file
    lextract.io.save_annotated_documents([result], output_name="extraction_results.json")
    # JSON to cypher
    cypher_script = await o.extract_insert_cypher_script()
    # execute cypher
    await o.populate_graph(cypher_query=cypher_script)
    # ingest to vector store (qdrant)
    await qdrant.insert("medical_research", [context])
    if need_vis:
        # Generate the interactive visualization from the file
        html_content = lextract.visualize("test_output/extraction_results.jsonl")
        with open("visualization.html", "w") as f:
            f.write(html_content)


async def format_rows(df):
    for _, row in df[:2].iterrows():
        # Extract context from JSON string
        context_obj = json.loads(row['context'].replace("'", '"'))
        full_context = ' '.join(context_obj['contexts'])

        # Create formatted string
        formatted = f"question: {row['question']}, context: {full_context}, answer: {row['long_answer']}"
        resp = await content_extract(input_text=formatted)
        await save_and_visualize_content(resp, context=f"context: {full_context}, answer: {row['long_answer']}")

    print("completed the data processing")

async def main():
    qdrant = AsyncQdrantIngestion()
    await qdrant.create_collection("medical_research")
    df = pd.read_csv("../data/pubmedqa.csv")
    await format_rows(df)

if __name__ == "__main__":
    asyncio.run(main())

