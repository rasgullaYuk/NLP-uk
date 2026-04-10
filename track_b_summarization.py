"""
Track B: RAG-Enabled Summarization Pipeline

Implements RAG (Retrieval-Augmented Generation) using:
- Amazon Titan Embeddings via Bedrock for vector embeddings
- FAISS indexing for semantic search
- Claude 3.5 Sonnet for role-based summary generation
- Validation layer with rule-based checks and JSON schema validation

Features:
- Document chunking by type (discharge summary, prescription, lab report)
- Role-based summaries (Clinician, Patient, Pharmacist)
- Medical knowledge retrieval for context augmentation
- Hallucination reduction techniques
- Performance target: <20s per document
"""

import json
import os
import re
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from bedrock_prompt_management import BedrockPromptManager
from hipaa_compliance import (
    build_phi_detection_summary,
    create_secure_client,
    detect_phi_entities,
    scrub_json_value,
    scrub_text_for_logs,
)
from track_b_validation import MedicalValidationEngine

# AWS Configuration
AWS_REGION = "us-east-1"
BEDROCK_CLAUDE_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"
BEDROCK_TITAN_EMBED_MODEL = "amazon.titan-embed-text-v1"
TRACK_B_TARGET_SECONDS = float(os.getenv("TRACK_B_TARGET_SECONDS", "15"))

# Paths
SUMMARY_OUTPUT_DIR = "track_b_outputs"
FAISS_INDEX_DIR = "faiss_indices"


class DocumentType(Enum):
    """Types of clinical documents."""
    DISCHARGE_SUMMARY = "discharge_summary"
    PRESCRIPTION = "prescription"
    LAB_REPORT = "lab_report"
    CLINICAL_NOTE = "clinical_note"
    RADIOLOGY_REPORT = "radiology_report"
    UNKNOWN = "unknown"


class SummaryRole(Enum):
    """Target audience roles for summaries."""
    CLINICIAN = "clinician"
    PATIENT = "patient"
    PHARMACIST = "pharmacist"


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    chunk_id: str
    text: str
    document_type: DocumentType
    section: str
    start_pos: int
    end_pos: int
    metadata: Dict


@dataclass
class SummaryOutput:
    """Structured summary output."""
    document_id: str
    role: SummaryRole
    summary: str
    key_points: List[str]
    medications: List[Dict]
    diagnoses: List[str]
    follow_up_actions: List[str]
    confidence_score: float
    validation_passed: bool
    validation_errors: List[str]
    validation_confidence_score: float
    hallucination_score: float
    ocr_deviation_flag: bool
    ocr_deviation_score: float
    ocr_deviation_details: List[Dict[str, Any]]
    validation_audit_log: List[Dict[str, Any]]
    auto_corrections: List[str]
    generation_time_ms: int
    prompt_tracking: Dict[str, Any]


class DocumentChunker:
    """
    Chunks clinical documents by type and section.
    """

    # Section patterns for different document types
    SECTION_PATTERNS = {
        DocumentType.DISCHARGE_SUMMARY: [
            r'(?i)(chief complaint|presenting complaint)',
            r'(?i)(history of present illness|hpi)',
            r'(?i)(past medical history|pmh)',
            r'(?i)(medications|current medications)',
            r'(?i)(physical examination|examination)',
            r'(?i)(assessment|impression|diagnosis)',
            r'(?i)(plan|treatment plan|management)',
            r'(?i)(discharge instructions|instructions)',
        ],
        DocumentType.PRESCRIPTION: [
            r'(?i)(patient information|patient details)',
            r'(?i)(medication|drug|prescription)',
            r'(?i)(dosage|dose)',
            r'(?i)(instructions|directions)',
            r'(?i)(warnings|contraindications)',
        ],
        DocumentType.LAB_REPORT: [
            r'(?i)(test name|investigation)',
            r'(?i)(result|value|finding)',
            r'(?i)(reference range|normal range)',
            r'(?i)(interpretation|comment)',
        ],
    }

    # Document type detection patterns
    TYPE_PATTERNS = {
        DocumentType.DISCHARGE_SUMMARY: [
            r'(?i)discharge\s+summar',
            r'(?i)hospital\s+discharge',
            r'(?i)inpatient\s+summar',
        ],
        DocumentType.PRESCRIPTION: [
            r'(?i)prescription',
            r'(?i)rx\s*:',
            r'(?i)medication\s+order',
        ],
        DocumentType.LAB_REPORT: [
            r'(?i)lab(oratory)?\s+report',
            r'(?i)blood\s+test',
            r'(?i)pathology\s+report',
        ],
        DocumentType.RADIOLOGY_REPORT: [
            r'(?i)radiology\s+report',
            r'(?i)x-ray\s+report',
            r'(?i)ct\s+scan',
            r'(?i)mri\s+report',
        ],
    }

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def detect_document_type(self, text: str) -> DocumentType:
        """Detects the type of clinical document."""
        for doc_type, patterns in self.TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text[:1000]):  # Check first 1000 chars
                    return doc_type
        return DocumentType.UNKNOWN

    def chunk_document(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Chunks a document by sections and size.

        Args:
            text: Full document text
            document_id: Unique document identifier

        Returns:
            List of DocumentChunk objects
        """
        doc_type = self.detect_document_type(text)
        chunks = []

        # Get section patterns for this document type
        patterns = self.SECTION_PATTERNS.get(doc_type, [])

        if patterns:
            # Split by sections
            sections = self._split_by_sections(text, patterns)
        else:
            # Fall back to fixed-size chunking
            sections = [("full_document", text)]

        for section_name, section_text in sections:
            # Further chunk large sections
            section_chunks = self._chunk_text(section_text)

            for i, chunk_text in enumerate(section_chunks):
                chunk_id = f"{document_id}_{section_name}_{i}"
                chunk_id = hashlib.md5(chunk_id.encode()).hexdigest()[:12]

                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    document_type=doc_type,
                    section=section_name,
                    start_pos=text.find(chunk_text),
                    end_pos=text.find(chunk_text) + len(chunk_text),
                    metadata={
                        'document_id': document_id,
                        'chunk_index': i,
                        'total_chunks': len(section_chunks)
                    }
                ))

        return chunks

    def _split_by_sections(self, text: str, patterns: List[str]) -> List[Tuple[str, str]]:
        """Splits text by section headers."""
        sections = []
        current_section = "header"
        current_text = ""

        lines = text.split('\n')

        for line in lines:
            matched = False
            for pattern in patterns:
                if re.search(pattern, line):
                    # Save current section
                    if current_text.strip():
                        sections.append((current_section, current_text.strip()))

                    # Start new section
                    current_section = re.sub(r'[^a-z_]', '_', line.lower().strip())[:30]
                    current_text = line + '\n'
                    matched = True
                    break

            if not matched:
                current_text += line + '\n'

        # Add last section
        if current_text.strip():
            sections.append((current_section, current_text.strip()))

        return sections if sections else [("full_document", text)]

    def _chunk_text(self, text: str) -> List[str]:
        """Chunks text into overlapping segments."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to end at sentence boundary
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1

            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap

        return chunks


class TitanEmbeddings:
    """
    Generates embeddings using Amazon Titan Embeddings via Bedrock.
    """

    def __init__(self):
        self.bedrock = create_secure_client(
            'bedrock-runtime',
            region_name=AWS_REGION
        )
        self.model_id = BEDROCK_TITAN_EMBED_MODEL
        self.embedding_dim = 1536  # Titan v1 dimension
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self.max_workers = max(1, int(os.getenv("TRACK_B_EMBED_WORKERS", "4")))

    def _embed_uncached(self, text: str) -> np.ndarray:
        text = text[:8000]
        body = json.dumps({"inputText": text})
        response = self.bedrock.invoke_model(modelId=self.model_id, body=body)
        result = json.loads(response['body'].read())
        return np.array(result['embedding'], dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generates embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            numpy array of embedding
        """
        cache_key = hashlib.sha256(text[:8000].encode("utf-8")).hexdigest()
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        embedding = self._embed_uncached(text)
        with self._cache_lock:
            self._cache[cache_key] = embedding
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            numpy array of embeddings (n_texts x embedding_dim)
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        unique_map: Dict[str, str] = {}
        for text in texts:
            key = hashlib.sha256(text[:8000].encode("utf-8")).hexdigest()
            unique_map.setdefault(key, text)

        missing: Dict[str, str] = {}
        with self._cache_lock:
            for key, text in unique_map.items():
                if key not in self._cache:
                    missing[key] = text

        if missing:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(missing))) as executor:
                futures = {executor.submit(self._embed_uncached, text): key for key, text in missing.items()}
                for future in as_completed(futures):
                    key = futures[future]
                    embedding = future.result()
                    with self._cache_lock:
                        self._cache[key] = embedding

        with self._cache_lock:
            ordered = [self._cache[hashlib.sha256(text[:8000].encode("utf-8")).hexdigest()] for text in texts]
        return np.vstack(ordered)


class FAISSIndex:
    """
    FAISS index for semantic search over document chunks and medical knowledge.
    """

    def __init__(self, embedding_dim: int = 1536):
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")

        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine for normalized vectors)
        self.chunks: List[DocumentChunk] = []
        self.medical_knowledge: List[Dict] = []

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Adds document chunks to the index."""
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def add_medical_knowledge(self, knowledge: List[Dict], embeddings: np.ndarray):
        """Adds external medical knowledge to the index."""
        self.index.add(embeddings)
        self.medical_knowledge.extend(knowledge)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Any, float]]:
        """
        Searches for most similar chunks.

        Args:
            query_embedding: Query vector
            k: Number of results to return

        Returns:
            List of (chunk/knowledge, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []

        # Reshape for FAISS
        query = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query, min(k, self.index.ntotal))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue

            score = float(distances[0][i])

            if idx < len(self.chunks):
                results.append((self.chunks[idx], score))
            else:
                knowledge_idx = idx - len(self.chunks)
                if knowledge_idx < len(self.medical_knowledge):
                    results.append((self.medical_knowledge[knowledge_idx], score))

        return results

    def save(self, path: str):
        """Saves the index to disk."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.faiss.write_index(self.index, f"{path}.faiss")

        metadata = {
            'chunks': [
                {
                    'chunk_id': c.chunk_id,
                    'text': c.text,
                    'document_type': c.document_type.value,
                    'section': c.section,
                    'metadata': c.metadata
                }
                for c in self.chunks
            ],
            'medical_knowledge': self.medical_knowledge
        }

        with open(f"{path}.json", 'w') as f:
            json.dump(metadata, f)

    def load(self, path: str):
        """Loads the index from disk."""
        self.index = self.faiss.read_index(f"{path}.faiss")

        with open(f"{path}.json", 'r') as f:
            metadata = json.load(f)

        self.chunks = [
            DocumentChunk(
                chunk_id=c['chunk_id'],
                text=c['text'],
                document_type=DocumentType(c['document_type']),
                section=c['section'],
                start_pos=0,
                end_pos=len(c['text']),
                metadata=c['metadata']
            )
            for c in metadata['chunks']
        ]

        self.medical_knowledge = metadata['medical_knowledge']


class ClaudeSummarizer:
    """
    Generates role-based summaries using Claude 3.5 Sonnet via Bedrock.
    """

    # Role-specific guidance inserted into prompt templates.
    ROLE_GUIDANCE = {
        SummaryRole.CLINICIAN: """You are a medical summarization assistant creating a clinical summary for a healthcare professional.

Focus on:
- Primary and secondary diagnoses with ICD codes if mentioned
- Critical findings and abnormal results
- Current medications with dosages
- Treatment plan and clinical reasoning
- Risk factors and prognosis
- Follow-up requirements

Use medical terminology appropriately. Be precise and comprehensive.""",

        SummaryRole.PATIENT: """You are a medical summarization assistant creating a patient-friendly summary.

Focus on:
- Explaining the diagnosis in simple terms
- What medications to take and when
- Warning signs to watch for
- Lifestyle recommendations
- When to seek medical attention
- Next appointment details

Avoid medical jargon. Use clear, reassuring language. Explain any medical terms used.""",

        SummaryRole.PHARMACIST: """You are a medical summarization assistant creating a summary for a pharmacist.

Focus on:
- Complete medication list with dosages and frequencies
- Drug interactions to monitor
- Contraindications based on patient history
- Allergies and adverse reactions
- Duration of therapy
- Special administration instructions
- Monitoring parameters

Be thorough with medication details. Flag any potential issues."""
    }

    # Output JSON schema
    OUTPUT_SCHEMA = {
        "type": "object",
        "required": ["summary", "key_points", "medications", "diagnoses", "follow_up_actions", "confidence_score"],
        "properties": {
            "summary": {"type": "string", "description": "Main summary text"},
            "key_points": {"type": "array", "items": {"type": "string"}, "description": "Key takeaways"},
            "medications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "dosage": {"type": "string"},
                        "frequency": {"type": "string"},
                        "instructions": {"type": "string"}
                    }
                }
            },
            "diagnoses": {"type": "array", "items": {"type": "string"}},
            "follow_up_actions": {"type": "array", "items": {"type": "string"}},
            "confidence_score": {"type": "number", "minimum": 0, "maximum": 1}
        }
    }

    def __init__(self):
        self.bedrock = create_secure_client(
            'bedrock-runtime',
            region_name=AWS_REGION
        )
        self.model_id = BEDROCK_CLAUDE_MODEL
        self.prompt_manager = BedrockPromptManager(
            model_id=self.model_id,
            region_name=AWS_REGION,
        )

    def generate_summary(self,
                         document_text: str,
                         document_id: str,
                         role: SummaryRole,
                         retrieved_context: List[str],
                         document_type: DocumentType,
                         forced_prompt_versions: Optional[Dict[str, str]] = None) -> Dict:
        """
        Generates a role-based summary using Claude 3.5 Sonnet.

        Args:
            document_text: Full document text
            document_id: Document identifier used for deterministic A/B prompt assignment
            role: Target audience role
            retrieved_context: Retrieved relevant context from FAISS
            document_type: Type of document
            forced_prompt_versions: Optional explicit template versions for debugging

        Returns:
            dict: Structured summary output
        """
        role_guidance = self.ROLE_GUIDANCE[role]
        prompt, prompt_tracking = self.prompt_manager.compose_track_b_prompt(
            document_id=document_id,
            role_key=role.value,
            role_guidance=role_guidance,
            document_type=document_type.value,
            clinical_document=document_text,
            retrieved_context=retrieved_context,
            output_schema=self.OUTPUT_SCHEMA,
            forced_versions=forced_prompt_versions,
        )

        # Call Claude via Bedrock
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "temperature": 0.1,  # Low temperature for consistency
            "messages": [
                {"role": "user", "content": prompt}
            ]
        })

        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            body=body
        )

        result = json.loads(response['body'].read())
        response_text = result['content'][0]['text']

        # Parse JSON from response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            summary_data = json.loads(response_text)
            summary_data["prompt_tracking"] = prompt_tracking
            return summary_data
        except json.JSONDecodeError:
            # Return minimal valid structure on parse error
            return {
                "summary": response_text,
                "key_points": [],
                "medications": [],
                "diagnoses": [],
                "follow_up_actions": [],
                "confidence_score": 0.5,
                "prompt_tracking": prompt_tracking,
            }


class SummaryValidator:
    """
    Validates generated summaries with rule-based checks and schema validation.
    """
    def __init__(self):
        self.engine = MedicalValidationEngine()
        self.last_report: Dict[str, Any] = {}

    def validate(self,
                 summary_output: Dict,
                 source_text: str,
                 document_type: DocumentType) -> Tuple[bool, List[str]]:
        """
        Validates a summary against rules and schema.

        Args:
            summary_output: Generated summary dict
            source_text: Original source document
            document_type: Type of document

        Returns:
            Tuple of (is_valid, list of errors)
        """
        _ = document_type  # Reserved for future document-type specific hard rules.
        report = self.engine.validate(summary_output, source_text)
        self.last_report = report
        return report['validation_passed'], report['errors']

    def calculate_hallucination_score(self, output: Dict, source: str) -> float:
        _ = output
        _ = source
        if self.last_report:
            return float(self.last_report.get("hallucination_score", 0.0))
        return 0.0

    def get_last_report(self) -> Dict[str, Any]:
        return self.last_report

    def compute_ocr_deviation_guard(
        self,
        summary_output: Dict[str, Any],
        textract_source_text: str,
        layoutlm_source_text: str,
        deviation_threshold: float = 0.30,
    ) -> Dict[str, Any]:
        return self.engine.compute_ocr_deviation_guard(
            summary_output=summary_output,
            textract_source_text=textract_source_text,
            layoutlm_source_text=layoutlm_source_text,
            deviation_threshold=deviation_threshold,
        )


class TrackBPipeline:
    """
    Main Track B RAG-Enabled Summarization Pipeline.
    """

    def __init__(self):
        self.chunker = DocumentChunker()
        self.embeddings = TitanEmbeddings()
        self.index = FAISSIndex()
        self.summarizer = ClaudeSummarizer()
        self.validator = SummaryValidator()
        self.last_metrics: Dict[str, Any] = {}

        # Create output directory
        os.makedirs(SUMMARY_OUTPUT_DIR, exist_ok=True)

    def process_document(self,
                         document_text: str,
                         document_id: str,
                         roles: List[SummaryRole] = None,
                         phi_entities: Optional[List[Dict[str, Any]]] = None,
                         textract_source_text: Optional[str] = None,
                         layoutlm_source_text: Optional[str] = None) -> Dict[str, SummaryOutput]:
        """
        Processes a document through the full Track B pipeline.

        Args:
            document_text: Full document text
            document_id: Unique document identifier
            roles: List of roles to generate summaries for (default: all)

        Returns:
            Dict mapping role to SummaryOutput
        """
        if roles is None:
            roles = list(SummaryRole)

        start_time = time.time()
        results = {}
        timings: Dict[str, float] = {}

        print(f"\nProcessing document: {document_id}")

        chunk_start = time.perf_counter()
        # 1. Chunk the document
        chunks = self.chunker.chunk_document(document_text, document_id)
        doc_type = chunks[0].document_type if chunks else DocumentType.UNKNOWN
        print(f"  Document type: {doc_type.value}")
        print(f"  Chunks created: {len(chunks)}")
        timings["chunking_seconds"] = round(time.perf_counter() - chunk_start, 4)

        embed_start = time.perf_counter()
        # 2. Generate embeddings for chunks
        chunk_texts = [c.text for c in chunks]
        chunk_embeddings = self.embeddings.embed_batch(chunk_texts)
        timings["chunk_embedding_seconds"] = round(time.perf_counter() - embed_start, 4)

        index_start = time.perf_counter()
        # 3. Add to index (for this document)
        self.index.add_chunks(chunks, chunk_embeddings)
        timings["index_add_seconds"] = round(time.perf_counter() - index_start, 4)

        query_start = time.perf_counter()
        # 4. Generate query embedding (use first chunk as summary query)
        query_text = f"Summarize this {doc_type.value}: {chunk_texts[0][:200]}"
        query_embedding = self.embeddings.embed_text(query_text)
        timings["query_embedding_seconds"] = round(time.perf_counter() - query_start, 4)

        retrieve_start = time.perf_counter()
        # 5. Retrieve relevant context
        search_results = self.index.search(query_embedding, k=5)
        retrieved_context = [r[0].text if isinstance(r[0], DocumentChunk) else r[0].get('text', '')
                            for r in search_results]
        timings["retrieval_seconds"] = round(time.perf_counter() - retrieve_start, 4)

        def _build_summary_for_role(role: SummaryRole):
            role_start = time.time()
            print(f"  Generating {role.value} summary...")
            role_validator = SummaryValidator()

            try:
                # Generate summary
                summary_data = self.summarizer.generate_summary(
                    document_text=document_text,
                    document_id=document_id,
                    role=role,
                    retrieved_context=retrieved_context,
                    document_type=doc_type
                )
                prompt_tracking = summary_data.get("prompt_tracking", {})

                # Validate
                is_valid, validation_errors = role_validator.validate(
                    summary_data, document_text, doc_type
                )
                validation_report = role_validator.get_last_report()
                corrected_summary = validation_report.get("corrected_output", summary_data)
                corrected_summary["prompt_tracking"] = prompt_tracking

                # Calculate hallucination score
                hallucination_score = role_validator.calculate_hallucination_score(
                    corrected_summary, document_text
                )
                ocr_deviation = role_validator.compute_ocr_deviation_guard(
                    summary_output=corrected_summary,
                    textract_source_text=textract_source_text or document_text,
                    layoutlm_source_text=layoutlm_source_text or textract_source_text or document_text,
                    deviation_threshold=0.30,
                )
                combined_audit_log = list(validation_report.get('audit_log', [])) + list(ocr_deviation.get('audit_checks', []))

                generation_time = int((time.time() - role_start) * 1000)

                output = SummaryOutput(
                    document_id=document_id,
                    role=role,
                    summary=corrected_summary.get('summary', ''),
                    key_points=corrected_summary.get('key_points', []),
                    medications=corrected_summary.get('medications', []),
                    diagnoses=corrected_summary.get('diagnoses', []),
                    follow_up_actions=corrected_summary.get('follow_up_actions', []),
                    confidence_score=corrected_summary.get('confidence_score', 0.5),
                    validation_passed=is_valid,
                    validation_errors=validation_errors,
                    validation_confidence_score=validation_report.get('validation_confidence_score', 0.0),
                    hallucination_score=hallucination_score,
                    ocr_deviation_flag=ocr_deviation.get('flagged_for_review', False),
                    ocr_deviation_score=ocr_deviation.get('deviation_score', 0.0),
                    ocr_deviation_details=ocr_deviation.get('critical_term_results', []),
                    validation_audit_log=combined_audit_log,
                    auto_corrections=validation_report.get('auto_corrections', []),
                    generation_time_ms=generation_time,
                    prompt_tracking=prompt_tracking,
                )

                print(f"    Validation: {'PASSED' if is_valid else 'FAILED'}")
                print(f"    Hallucination score: {hallucination_score:.3f}")
                print(f"    OCR deviation score: {ocr_deviation.get('deviation_score', 0.0):.3f} | "
                      f"Flagged: {ocr_deviation.get('flagged_for_review', False)}")
                print(f"    Time: {generation_time}ms")
                return role.value, output, generation_time

            except Exception as e:
                print(f"    ERROR: {e}")
                output = SummaryOutput(
                    document_id=document_id,
                    role=role,
                    summary=f"Error generating summary: {str(e)}",
                    key_points=[],
                    medications=[],
                    diagnoses=[],
                    follow_up_actions=[],
                    confidence_score=0,
                    validation_passed=False,
                    validation_errors=[str(e)],
                    validation_confidence_score=0.0,
                    hallucination_score=1.0,
                    ocr_deviation_flag=True,
                    ocr_deviation_score=1.0,
                    ocr_deviation_details=[],
                    validation_audit_log=[],
                    auto_corrections=[],
                    generation_time_ms=0,
                    prompt_tracking={},
                )
                return role.value, output, 0

        role_workers = max(1, min(len(roles), int(os.getenv("TRACK_B_ROLE_WORKERS", "3"))))
        with ThreadPoolExecutor(max_workers=role_workers) as executor:
            futures = [executor.submit(_build_summary_for_role, role) for role in roles]
            for future in as_completed(futures):
                role_name, output, generation_time = future.result()
                results[role_name] = output
                timings[f"role_{role_name}_seconds"] = round(generation_time / 1000.0, 4)

        total_time = time.time() - start_time
        print(f"  Total processing time: {total_time:.2f}s")
        timings["total_seconds"] = round(total_time, 4)
        timings["target_seconds"] = TRACK_B_TARGET_SECONDS
        timings["target_met"] = total_time <= TRACK_B_TARGET_SECONDS
        self.last_metrics = timings

        # 7. Save results
        self._save_results(document_id, results, phi_entities=phi_entities or [])

        return results

    def _save_results(self, document_id: str, results: Dict[str, SummaryOutput], phi_entities: List[Dict[str, Any]]):
        """Saves summary results to files."""
        phi_summary = build_phi_detection_summary(phi_entities)
        for role, output in results.items():
            role_lower = role.lower()
            apply_masking = role_lower != SummaryRole.CLINICIAN.value
            summary_text = (
                scrub_text_for_logs(output.summary, phi_entities)
                if apply_masking else output.summary
            )
            key_points = (
                [scrub_text_for_logs(point, phi_entities) for point in output.key_points]
                if apply_masking else output.key_points
            )
            medications = (
                scrub_json_value(output.medications, phi_entities)
                if apply_masking else output.medications
            )
            diagnoses = (
                [scrub_text_for_logs(dx, phi_entities) for dx in output.diagnoses]
                if apply_masking else output.diagnoses
            )
            follow_up_actions = (
                [scrub_text_for_logs(action, phi_entities) for action in output.follow_up_actions]
                if apply_masking else output.follow_up_actions
            )

            # Save as text file (for backward compatibility)
            txt_path = os.path.join(SUMMARY_OUTPUT_DIR, f"{document_id}_{role}_summary.txt")
            with open(txt_path, 'w') as f:
                f.write(f"=== {role.upper()} SUMMARY ===\n\n")
                f.write(summary_text)
                f.write("\n\n=== KEY POINTS ===\n")
                for point in key_points:
                    f.write(f"- {point}\n")
                f.write("\n=== MEDICATIONS ===\n")
                for med in medications:
                    f.write(f"- {med.get('name', 'Unknown')}: {med.get('dosage', '')} {med.get('frequency', '')}\n")
                f.write("\n=== DIAGNOSES ===\n")
                for dx in diagnoses:
                    f.write(f"- {dx}\n")
                f.write("\n=== FOLLOW-UP ACTIONS ===\n")
                for action in follow_up_actions:
                    f.write(f"- {action}\n")

            # Save as JSON (full structured output)
            json_path = os.path.join(SUMMARY_OUTPUT_DIR, f"{document_id}_{role}_summary.json")
            with open(json_path, 'w') as f:
                json.dump({
                    'document_id': output.document_id,
                    'role': output.role.value,
                    'summary': summary_text,
                    'key_points': key_points,
                    'medications': medications,
                    'diagnoses': diagnoses,
                    'follow_up_actions': follow_up_actions,
                    'confidence_score': output.confidence_score,
                    'validation_passed': output.validation_passed,
                    'validation_errors': output.validation_errors,
                    'validation_confidence_score': output.validation_confidence_score,
                    'hallucination_score': output.hallucination_score,
                    'ocr_deviation_flag': output.ocr_deviation_flag,
                    'ocr_deviation_score': output.ocr_deviation_score,
                    'ocr_deviation_details': output.ocr_deviation_details,
                    'validation_audit_log': output.validation_audit_log,
                    'auto_corrections': output.auto_corrections,
                    'generation_time_ms': output.generation_time_ms,
                    'prompt_tracking': output.prompt_tracking,
                    'prompt_versions': output.prompt_tracking.get("selected_versions", {}),
                    'generated_at': datetime.utcnow().isoformat() + 'Z',
                    'phi_detection': phi_summary,
                    'phi_masking_applied': apply_masking,
                }, f, indent=2)

        print(f"  Results saved to {SUMMARY_OUTPUT_DIR}/")


def process_track_b_queue(queue_name: str = 'TrackB_Summary_Queue'):
    """
    Processes documents from the Track B SQS queue.

    Args:
        queue_name: Name of the SQS queue to consume from
    """
    from sqs_messaging import receive_from_sqs, delete_from_sqs
    from sqs_setup import get_queue_url
    from audit_dynamodb import get_audit_logger

    print("=" * 60)
    print("TRACK B: RAG-Enabled Summarization Pipeline")
    print("=" * 60)

    queue_url = get_queue_url(queue_name)
    if not queue_url:
        print(f"ERROR: Could not find queue {queue_name}")
        return

    pipeline = TrackBPipeline()
    audit_logger = get_audit_logger()
    sqs_client = create_secure_client('sqs', region_name=AWS_REGION)
    review_queue_name = "RVCE14_Review_Interface_Queue"
    try:
        review_queue_url = sqs_client.get_queue_url(QueueName=review_queue_name)["QueueUrl"]
    except Exception:
        review_queue_url = sqs_client.create_queue(QueueName=review_queue_name)["QueueUrl"]

    print(f"\nListening on {queue_name}...")
    print("Press Ctrl+C to stop\n")

    while True:
        messages = receive_from_sqs(queue_url, max_messages=1)

        if not messages:
            print("Queue empty. Waiting...")
            time.sleep(5)
            continue

        for message in messages:
            try:
                payload = json.loads(message['Body'])
                document_id = payload.get('document_id', 'unknown')
                source_file = payload.get('source_file')
                layout_refined_file = payload.get('layout_refined_file')

                # Load document text
                if source_file and os.path.exists(source_file):
                    with open(source_file, 'r') as f:
                        textract_data = json.load(f)

                    # Extract text from Textract blocks
                    text_lines = []
                    for block in textract_data.get('Blocks', []):
                        if block.get('BlockType') == 'LINE':
                            text_lines.append(block.get('Text', ''))

                    document_text = '\n'.join(text_lines)
                    textract_source_text = document_text
                else:
                    document_text = payload.get('text', '')
                    textract_source_text = document_text

                # LayoutLMv3 refined source text if available
                if not layout_refined_file:
                    default_layout_path = os.path.join("tier2_refined", f"{document_id}_refined.json")
                    layout_refined_file = default_layout_path if os.path.exists(default_layout_path) else None

                layoutlm_source_text = ""
                if layout_refined_file and os.path.exists(layout_refined_file):
                    with open(layout_refined_file, 'r', encoding='utf-8') as f:
                        layout_data = json.load(f)
                    refined_elements = layout_data.get("refined_elements", [])
                    layoutlm_source_text = "\n".join([e.get("text", "") for e in refined_elements if e.get("text")])

                if not document_text:
                    print(f"No text found for {document_id}")
                    delete_from_sqs(queue_url, message['ReceiptHandle'])
                    continue

                phi_entities = detect_phi_entities(document_text)
                phi_summary = build_phi_detection_summary(phi_entities)
                print(f"PHI entities flagged for {document_id}: {phi_summary['entity_count']}")

                # Process document
                results = pipeline.process_document(
                    document_text,
                    document_id,
                    phi_entities=phi_entities,
                    textract_source_text=textract_source_text,
                    layoutlm_source_text=layoutlm_source_text,
                )

                # Route OCR-deviation flagged outputs to Review Interface and audit log.
                for role_name, output in results.items():
                    if not output.ocr_deviation_flag:
                        continue
                    review_payload = {
                        "document_id": document_id,
                        "role": role_name,
                        "ocr_deviation_flag": True,
                        "ocr_deviation_score": output.ocr_deviation_score,
                        "ocr_deviation_details": output.ocr_deviation_details,
                        "source_file": source_file,
                        "layout_refined_file": layout_refined_file,
                        "queued_at": datetime.utcnow().isoformat() + "Z",
                    }
                    try:
                        sqs_client.send_message(QueueUrl=review_queue_url, MessageBody=json.dumps(review_payload))
                        print(f"Queued {document_id}/{role_name} to {review_queue_name} due to OCR deviation.")
                    except Exception as queue_error:
                        print(f"Failed to queue review payload for {document_id}/{role_name}: {queue_error}")

                    try:
                        if audit_logger:
                            audit_logger.log_change(
                                document_id=document_id,
                                user_id="SYSTEM",
                                change_type="OCR_DEVIATION_GUARD",
                                before_state={"role": role_name, "status": "auto_accept_candidate"},
                                after_state={"role": role_name, "status": "manual_review_required"},
                                metadata={
                                    "ocr_deviation_score": output.ocr_deviation_score,
                                    "ocr_deviation_details": output.ocr_deviation_details,
                                },
                            )
                    except Exception as audit_error:
                        print(f"Audit logging failed for OCR deviation guard on {document_id}/{role_name}: {audit_error}")

                # Delete processed message
                delete_from_sqs(queue_url, message['ReceiptHandle'])
                print(f"Document {document_id} processed successfully")

            except Exception as e:
                print(f"ERROR processing message: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--queue":
        # Queue mode
        process_track_b_queue()
    else:
        # Test mode with sample document
        print("=" * 60)
        print("TRACK B: RAG-Enabled Summarization Pipeline (Test Mode)")
        print("=" * 60)

        sample_document = """
        DISCHARGE SUMMARY

        Patient: [REDACTED]
        Date of Admission: 2024-01-10
        Date of Discharge: 2024-01-15

        CHIEF COMPLAINT:
        Chest pain and shortness of breath for 2 days.

        HISTORY OF PRESENT ILLNESS:
        65-year-old male with history of hypertension and type 2 diabetes
        presented with substernal chest pain radiating to left arm, associated
        with diaphoresis and dyspnea. Pain started 2 days ago and has been
        intermittent. Patient has history of smoking (30 pack-years, quit 5 years ago).

        PAST MEDICAL HISTORY:
        1. Hypertension - diagnosed 2015
        2. Type 2 Diabetes Mellitus - diagnosed 2018
        3. Hyperlipidemia

        CURRENT MEDICATIONS:
        1. Metformin 1000mg twice daily
        2. Lisinopril 20mg daily
        3. Atorvastatin 40mg at bedtime
        4. Aspirin 81mg daily

        PHYSICAL EXAMINATION:
        BP: 145/90 mmHg, HR: 88 bpm, RR: 18/min, SpO2: 96% on room air
        Heart: Regular rhythm, no murmurs
        Lungs: Clear bilaterally

        INVESTIGATIONS:
        - ECG: ST depression in leads V4-V6
        - Troponin I: 0.8 ng/mL (elevated)
        - Echo: EF 45%, mild LV hypokinesis

        ASSESSMENT:
        1. Non-ST elevation myocardial infarction (NSTEMI)
        2. Hypertension, uncontrolled
        3. Type 2 Diabetes Mellitus

        PLAN:
        1. Admitted to CCU for monitoring
        2. Started on Heparin drip
        3. Cardiology consultation for possible catheterization
        4. Continue home medications
        5. Add Clopidogrel 75mg daily
        6. Add Metoprolol 25mg twice daily

        DISCHARGE MEDICATIONS:
        1. Metformin 1000mg twice daily
        2. Lisinopril 20mg daily
        3. Atorvastatin 40mg at bedtime
        4. Aspirin 81mg daily
        5. Clopidogrel 75mg daily
        6. Metoprolol 25mg twice daily

        DISCHARGE INSTRUCTIONS:
        - Follow low-sodium, diabetic diet
        - Avoid strenuous activity for 2 weeks
        - Monitor blood pressure daily
        - Return if chest pain, shortness of breath, or dizziness occurs

        FOLLOW-UP:
        - Cardiology clinic in 1 week
        - Primary care in 2 weeks
        """

        pipeline = TrackBPipeline()
        results = pipeline.process_document(
            document_text=sample_document,
            document_id="test_discharge_001",
            textract_source_text=sample_document,
            layoutlm_source_text=sample_document,
        )

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        for role, output in results.items():
            print(f"\n{role.upper()}:")
            print(f"  Validation: {'PASSED' if output.validation_passed else 'FAILED'}")
            print(f"  Confidence: {output.confidence_score:.2f}")
            print(f"  Time: {output.generation_time_ms}ms")
            print(f"  Key Points: {len(output.key_points)}")
            print(f"  Medications: {len(output.medications)}")
            print(f"  Diagnoses: {len(output.diagnoses)}")
