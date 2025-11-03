# Medical Knowledge Extraction and Graph Construction System

A sophisticated system for extracting medical entities, relationships, and knowledge from biomedical literature, creating knowledge graphs, and enabling semantic search through vector databases.

## 🎯 Overview

This project processes biomedical texts (particularly from PubMedQA dataset) to build comprehensive knowledge graphs that capture medical entities, their relationships, and contextual information. The system combines entity-level graph extraction with vector database storage for both structured and semantic querying capabilities.

## 🏗️ System Architecture

```
Medical Text Input
       ↓
┌────────────────────────────┐
│    Text to Entity Extraction                  │
│    (LangExtract + Gemini)                   │
└────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│   Entity-Relationship-Edge      │
│      JSON Generation            │
└─────────────────────────────────┘
       ↓
┌─────────────────┬───────────────┐
│  Neo4j Graph    │  Qdrant Vector │
│   Database      │   Database     │
│  (Cypher)       │ (Embeddings)   │
└─────────────────┴───────────────┘
       ↓
┌─────────────────────────────────┐
│   Query & Visualization         │
│     Interface                   │
└─────────────────────────────────┘
```

## 🔧 Core Components

### 1. Text to Entity Level Graph Converter (`text_to_entity_level_graph_converter.py`)
- **Purpose**: Main orchestrator for the extraction pipeline
- **Key Functions**:
  - `content_extract()`: Extracts medical entities, relations, and edges from text using Gemini 2.5 Flash
  - `save_and_visualize_content()`: Processes extracted data and populates both graph and vector databases
  - `format_rows()`: Processes PubMedQA CSV data in batches

**Medical Entity Categories**:
- **Entities**: Diseases, drugs, procedures, biomarkers, cell types, anatomical structures
- **Relations**: Causative (causes, leads_to), therapeutic (treats, prevents), quantitative (increased_in, decreased_in)
- **Edges**: Specific entity-relationship-entity triples with confidence scores

### 2. Qdrant Vector Database Integration (`ingest_to_qdrant.py`)
- **Purpose**: Vector storage and semantic search capabilities
- **Features**:
  - Medical research metadata extraction using LangExtract
  - Automatic embedding generation with SentenceTransformer
  - Research paper categorization (study type, medical domain, sample size)
  - Async operations for scalability

**Extracted Metadata Fields**:
- Research question
- Study type (prospective, retrospective, randomized, etc.)
- Medical domain (cardiology, neurology, surgery, etc.)
- Sample size and population studied
- Intervention type

### 3. Neo4j Knowledge Graph (`neo4j_populator.py`, `entity_relation_json_to_cypher_query.py`)
- **Purpose**: Structured graph database for complex relationship queries
- **Features**:
  - Automatic Cypher query generation from extracted JSON
  - Support for both CREATE and SEARCH operations
  - Medical terminology preservation
  - Relationship type classification

### 4. Search Playground (`search_playground.py`)
- **Purpose**: Interactive querying interface
- **Functionality**:
  - Real-time entity extraction from user queries
  - Dynamic Cypher query generation
  - Graph search capabilities

## 📋 Installation

### Prerequisites
- Python 3.8+
- Neo4j Database
- Qdrant Vector Database
- Google Gemini API Key

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd medical-knowledge-extraction

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your GEMINI_API_KEY to .env file

# Start Neo4j (default: bolt://localhost:7687)
# Start Qdrant (default: http://localhost:6333)
```

### Environment Configuration
```bash
# .env file
GEMINI_API_KEY=your_gemini_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j@123
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=th3s3cr3tk3y
```

## 🚀 Usage

### Processing Medical Literature
```python
import asyncio
from text_to_entity_level_graph_converter import main

# Process PubMedQA dataset
asyncio.run(main())
```

### Interactive Search
```python
from search_playground import main

# Query the knowledge graph
query = "What is the spectrum of antimicrobial activity of amoxicillin?"
asyncio.run(main(query))
```

### Manual Text Processing
```python
from text_to_entity_level_graph_converter import content_extract, save_and_visualize_content

# Extract from custom text
medical_text = "ILC2s were increased in patients with CRSwNP..."
result = await content_extract(medical_text)
await save_and_visualize_content(result, context=medical_text, need_vis=True)
```

## 📊 Data Flow

1. **Input Processing**: Medical texts from PubMedQA or custom sources
2. **Entity Extraction**: Uses Gemini 2.5 Flash with specialized medical prompts
3. **Knowledge Graph Construction**: Converts extracted entities to Cypher queries
4. **Vector Database Storage**: Creates embeddings for semantic search
5. **Visualization**: Generates interactive HTML visualizations

## 🔍 Query Capabilities

### Graph Queries (Neo4j)
- Find all entities related to a specific disease
- Trace treatment pathways and contraindications
- Identify research gaps in medical literature
- Map drug-disease relationships

### Semantic Search (Qdrant)
- Find similar research studies
- Identify papers by methodology or population
- Discover related medical domains
- Search by research outcomes

## 📈 Example Output

### Extracted Entities
```json
{
  "extraction_class": "entity",
  "extraction_text": "ILC2s",
  "attributes": {"type": "cell_type", "scale": "entity"}
}
```

### Relationships
```json
{
  "extraction_class": "edge",
  "extraction_text": "(ILC2s, increased_in, CRSwNP)",
  "attributes": {"confidence": "high", "evidence_based": "true"}
}
```

### Generated Cypher
```cypher
CREATE
  (a:Entity {name: "ILC2s", type: "cell_type"})
  (b:Entity {name: "CRSwNP", type: "disease"})
  (a)-[:INCREASED_IN {confidence: "high"}]->(b)
```

## 🎛️ Configuration

### Model Settings
- **Primary Model**: Gemini 2.5 Flash for speed and accuracy
- **Embedding Model**: all-MiniLM-L6-v2 for vector generation
- **Vector Dimension**: 384 (configurable)

### Processing Parameters
- **Batch Size**: 2 papers per batch (configurable)
- **Confidence Threshold**: High, Medium, Low classifications
- **Medical Domain Focus**: Optimized for clinical and research literature

## 🔧 Customization

### Adding New Medical Domains
1. Extend entity types in extraction prompt
2. Add domain-specific relationship types
3. Update metadata extraction categories
4. Modify visualization templates

### Custom Entity Types
```python
# Add new entity categories
entity_types = [
    "biomarker", "genetic_variant", "pathway", 
    "clinical_trial_phase", "medical_device"
]
```

## 🚧 Troubleshooting

### Common Issues
- **Neo4j Connection**: Verify database is running and credentials are correct
- **Qdrant Setup**: Ensure vector database is accessible at specified URL
- **API Limits**: Check Gemini API quotas and rate limits
- **Memory Usage**: Large datasets may require batch processing

### Performance Optimization
- Use async operations for concurrent processing
- Implement connection pooling for database operations
- Cache frequently accessed embeddings
- Optimize Cypher queries with proper indexing

## 📚 Dependencies

### Core Libraries
- `langextract==1.0.8`: Advanced text extraction with LLM integration
- `google-genai==1.30.0`: Google Gemini API client
- `neo4j`: Graph database driver
- `qdrant-client`: Vector database client
- `sentence-transformers`: Embedding generation

### Supporting Libraries
- `pandas`: Data manipulation
- `pydantic`: Data validation
- `python-dotenv`: Environment configuration
- `protobuf`: Serialization support

## 🎯 Future Enhancements

- [ ] Multi-language medical text support
- [ ] Real-time knowledge graph updates
- [ ] Advanced visualization dashboards
- [ ] Integration with medical ontologies (UMLS, SNOMED CT)
- [ ] Automated literature review generation
- [ ] Clinical decision support features

## 📄 License

This project is designed for research and educational purposes in medical informatics and knowledge extraction under MIT license.

## 🤝 Contributing

Contributions welcome! Please focus on:
- Medical domain expertise
- Performance optimizations
- New visualization features
- Extended database integrations

## 📞 Support

For technical issues or medical domain questions, please create detailed issues with:
- Input text samples
- Expected vs actual outputs
- System configuration details
- Error logs and stack traces