from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

import langextract as lx
from typing import List, Dict

class MedicalResearchExtractor:
    def __init__(self):
        self.lx = lx

        # Define examples for LangExtract based on medical research papers
        self.examples = [
            lx.data.ExampleData(
                text="""Does mammary stimulation test predict preterm birth in nulliparous women?,"{'contexts': ['This prospective clinical trial was designed to assess the ability of the mammary stimulation test to predict preterm birth in a private nulliparous population.', 'The mammary stimulation test was performed between 26 and 28 weeks gestation by 267 nulliparous women with singleton pregnancies.'], 'labels': ['OBJECTIVE', 'METHODS', 'RESULTS'], 'meshes': ['Adult', 'Cardiotocography', 'Female', 'Humans', 'Pregnancy']}""",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="research_question",
                        extraction_text="Does mammary stimulation test predict preterm birth in nulliparous women?",
                        attributes={}
                    ),
                    lx.data.Extraction(
                        extraction_class="study_type",
                        extraction_text="prospective clinical trial",
                        attributes={}
                    ),
                    lx.data.Extraction(
                        extraction_class="medical_domain",
                        extraction_text="obstetrics",
                        attributes={}
                    ),
                    lx.data.Extraction(
                        extraction_class="sample_size",
                        extraction_text="267",
                        attributes={}
                    ),
                    lx.data.Extraction(
                        extraction_class="population_studied",
                        extraction_text="nulliparous women with singleton pregnancies",
                        attributes={}
                    ),
                    lx.data.Extraction(
                        extraction_class="intervention_type",
                        extraction_text="diagnostic test",
                        attributes={}
                    )
                ]
            ),
            lx.data.ExampleData(
                text="""Is insulin regulation of hepatic glucose transporter protein impaired in chronic pancreatitis?,"{'contexts': ['Chronic pancreatitis is associated with diabetes mellitus or impaired glucose tolerance.', 'Normal rats, rats with chronic pancreatitis induced 12 to 16 weeks earlier by oleic acid injection into the pancreatic ducts, and sham-operated rats were studied.'], 'labels': ['OBJECTIVE', 'BACKGROUND', 'METHODS', 'RESULTS'], 'meshes': ['Animals', 'Chronic Disease', 'Glucagon', 'Insulin', 'Liver', 'Pancreatitis', 'Rats']}""",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="research_question",
                        extraction_text="Is insulin regulation of hepatic glucose transporter protein impaired in chronic pancreatitis?",
                        attributes={}
                    ),
                    lx.data.Extraction(
                        extraction_class="study_type",
                        extraction_text="experimental animal study",
                        attributes={}
                    ),
                    lx.data.Extraction(
                        extraction_class="medical_domain",
                        extraction_text="endocrinology",
                        attributes={}
                    ),
                    lx.data.Extraction(
                        extraction_class="population_studied",
                        extraction_text="rats with chronic pancreatitis",
                        attributes={}
                    ),
                    lx.data.Extraction(
                        extraction_class="intervention_type",
                        extraction_text="drug administration",
                        attributes={}
                    )
                ]
            ),
            lx.data.ExampleData(
                text="""Does transfusion significantly increase the risk for infection after splenic injury?,"{'contexts': ['To determine if splenectomy results in an increased risk for perioperative infection when analyzed against splenic repair', 'Data were collected retrospectively from hospital records and analyzed using stepwise multiple logistic regression.', 'All patients (n = 252) undergoing operation for traumatic splenic injury'], 'labels': ['OBJECTIVE', 'METHODS', 'METHODS', 'METHODS'], 'meshes': ['Adult', 'Female', 'Humans', 'Spleen', 'Splenectomy', 'Surgery']}""",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="research_question",
                        extraction_text="Does transfusion significantly increase the risk for infection after splenic injury?",
                        attributes={}
                    ),
                    lx.data.Extraction(
                        extraction_class="study_type",
                        extraction_text="retrospective study",
                        attributes={}
                    ),
                    lx.data.Extraction(
                        extraction_class="medical_domain",
                        extraction_text="surgery",
                        attributes={}
                    ),
                    lx.data.Extraction(
                        extraction_class="sample_size",
                        extraction_text="252",
                        attributes={}
                    ),
                    lx.data.Extraction(
                        extraction_class="population_studied",
                        extraction_text="patients undergoing operation for traumatic splenic injury",
                        attributes={}
                    ),
                    lx.data.Extraction(
                        extraction_class="intervention_type",
                        extraction_text="surgical procedure",
                        attributes={}
                    )
                ]
            )
        ]

    def extract_metadata(self, text: str) -> Dict:
        """Extract metadata from medical research text using LangExtract"""

        # Define what to extract
        labels = [
            "research_question",
            "study_type",
            "medical_domain",
            "sample_size",
            "population_studied",
            "intervention_type"
        ]

        # Instructions for extraction
        instructions = """
        Extract medical research metadata:
        - research_question: The main research question being investigated
        - study_type: Type of study (prospective, retrospective, randomized, experimental, observational, etc.)
        - medical_domain: Medical field (cardiology, neurology, surgery, obstetrics, endocrinology, etc.)
        - sample_size: Number of subjects/participants (just the number)
        - population_studied: Who was studied (patients, animals, specific patient groups)
        - intervention_type: What was tested (drug, procedure, device, diagnostic test, etc.)
        """

        # Extract using LangExtract
        extractions = self.lx.extract(
            text_or_documents=text,
            model_id="gemini-2.5-pro",
            examples=self.examples,
            prompt_description=instructions
        )

        # Convert to dictionary
        metadata = {}
        for extraction in extractions.extractions:
            metadata[extraction.extraction_class] = extraction.extraction_text

        return metadata

    def extract_from_list(self, texts: List[str]) -> List[Dict]:
        """Extract metadata from list of medical research texts"""
        return [self.extract_metadata(text) for text in texts]


class AsyncQdrantIngestion:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.client = AsyncQdrantClient(url="http://localhost:6333", api_key="th3s3cr3tk3y")
        self.encoder = SentenceTransformer(model_name)
        self.vector_size = self.encoder.get_sentence_embedding_dimension()
        self.extractor = MedicalResearchExtractor()

    async def create_collection(self, name):
        await self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
        )

    async def insert(self, collection_name, texts: List[str]):
        # Extract metadata for all texts
        metadata = []
        for text in texts:
            meta = self.extractor.extract_metadata(text)
            metadata.append(meta)

        # Generate embeddings
        vectors = self.encoder.encode(texts)

        # Create points
        points = []
        for i, (text, vector) in enumerate(zip(texts, vectors)):
            payload = {"text": text}
            if metadata and i < len(metadata):
                payload.update(metadata[i])

            points.append(PointStruct(id=i, vector=vector.tolist(), payload=payload))

        await self.client.upsert(collection_name=collection_name, points=points)

    async def close(self):
        """Close the async client connection"""
        await self.client.close()

# Usage

# if __name__ == "__main__":
#     extractor = MedicalResearchExtractor()
#
#     sample_text = """Does mammary stimulation test predict preterm birth in nulliparous women?,"{'contexts': ['This prospective clinical trial was designed to assess the ability of the mammary stimulation test to predict preterm birth in a private nulliparous population.', 'The mammary stimulation test was performed between 26 and 28 weeks gestation by 267 nulliparous women with singleton pregnancies.'], 'labels': ['OBJECTIVE', 'METHODS', 'RESULTS']}"""
#
#     metadata = extractor.extract_metadata(sample_text)
#     print("Extracted Metadata:", metadata)

# qdrant = QdrantIngestion()
# qdrant.create_collection("docs")
# qdrant.insert("docs", ["""Does mammary stimulation test predict preterm birth in nulliparous women?,"{'contexts': ['This prospective clinical trial was designed to assess the ability of the mammary stimulation test to predict preterm birth in a private nulliparous population.', 'The mammary stimulation test was performed between 26 and 28 weeks gestation by 267 nulliparous women with singleton pregnancies.'], 'labels': ['OBJECTIVE', 'METHODS', 'RESULTS']}"""])

