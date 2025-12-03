"""
LlamaIndex Experiments - Fault Extraction from PDF Manuals

This script extracts machine faults, root causes, and fixing steps from PDF manuals.
Uses Azure OpenAI for LLM-based extraction and LlamaIndex for document processing.
"""

import os
import json
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
from pypdf import PdfReader
from pydantic import BaseModel, Field
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

# --- Configuration Constants ---

# Document chunking parameters
CHUNK_SIZE = 8192  # Characters per node
CHUNK_OVERLAP = 400  # Overlap between consecutive chunks

# Root cause extraction parameters
CONTEXT_WINDOW = 1  # Number of nodes before/after each fault mention to include
MAX_NODES_FOR_ROOT_CAUSE = 8  # Maximum nodes to send to LLM for root cause extraction

# Metadata extraction parameters
METADATA_TEXT_LENGTH = 2000  # Characters from start of document to use for metadata

# Pricing: dollars per 1M tokens (Standard tier)
MODEL_PRICING_INPUT = {
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "gpt-5-mini": 0.25,
    "gpt-5-nano": 0.05,
}
MODEL_PRICING_OUTPUT = {
    "gpt-4o": 10.00,
    "gpt-4o-mini": 0.60,
    "gpt-5-mini": 2.00,
    "gpt-5-nano": 0.40,
}

# Token estimation: approximate 4 characters per token
def estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    # Rough heuristic; replace with model-specific tokenizer if needed
    return max(1, int(len(text) / 4))

class TokenTracker:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.events = []  # (label, in_tokens, out_tokens)

    def record(self, label: str, prompt_text: str, output_text: str):
        in_tok = estimate_tokens_from_text(prompt_text)
        out_tok = estimate_tokens_from_text(output_text)
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.events.append((label, in_tok, out_tok))

    def summary(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        }

def current_cost_estimate() -> dict:
    """Compute current estimated costs using selected model."""
    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    in_price = MODEL_PRICING_INPUT.get(model_name)
    out_price = MODEL_PRICING_OUTPUT.get(model_name)
    usage_input = token_tracker.input_tokens
    usage_output = token_tracker.output_tokens
    if in_price is None or out_price is None:
        return {"model": model_name, "in_cost": None, "out_cost": None, "total_cost": None}
    in_cost = (usage_input / 1_000_000) * in_price
    out_cost = (usage_output / 1_000_000) * out_price
    return {"model": model_name, "in_cost": in_cost, "out_cost": out_cost, "total_cost": in_cost + out_cost}

# --- Data Models ---

class MachineMetadata(BaseModel):
    """Metadata about the machine from the manual."""
    manufacturer: str
    model: str
    series: str
    description: str

class FixingStep(BaseModel):
    """A single step in a fixing sequence."""
    order: int
    description: str

class FixingSequence(BaseModel):
    """A sequence of steps to fix a root cause."""
    source: str  # "manual", "online", or "hybrid"
    steps: List[FixingStep]
    confidence_score: float

class RootCause(BaseModel):
    """A root cause for a machine fault."""
    id: str
    description: str
    fixing_sequences: List[FixingSequence]

class MachineFault(BaseModel):
    """A machine fault with optional root causes."""
    fault_code: str
    fault_message: str
    source_chunk: str
    source_node_index: Optional[int] = None
    root_causes: Optional[List[RootCause]] = Field(default_factory=list)

class FaultList(BaseModel):
    """List of faults for structured extraction."""
    faults: List[MachineFault]

class RootCauseList(BaseModel):
    """List of root causes for structured extraction."""
    root_causes: List[RootCause]

# --- Setup ---

llm = AzureOpenAI(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

token_tracker = TokenTracker()

# --- Extraction Functions ---

def extract_metadata(text: str) -> MachineMetadata:
    """Extract machine metadata from the manual using structured output."""
    print("Extracting Metadata...")
    
    prompt_template_str = """Extract the machine metadata from the following text.

Text:
---------------------
{text}
---------------------

Provide the manufacturer, model, series, and a brief description."""

    program = LLMTextCompletionProgram.from_defaults(
        output_cls=MachineMetadata,
        llm=llm,
        prompt_template_str=prompt_template_str,
        verbose=True
    )
    
    used_text = text[:METADATA_TEXT_LENGTH]
    result = program(text=used_text)
    try:
        # LLMTextCompletionProgram doesn't expose raw text; reconstruct prompt minimalistically
        prompt_used = prompt_template_str.format(text=used_text)
        token_tracker.record("metadata", prompt_used, json.dumps(result.model_dump()))
    except Exception:
        pass
    return result

def extract_faults_from_nodes(nodes) -> List[MachineFault]:
    """Extract fault codes from document nodes using LlamaIndex."""
    print(f"Extracting fault codes from {len(nodes)} nodes...")
    
    prompt_template_str = """Extract all machine fault codes from the following text.

For each fault, identify:
- The fault code (e.g., "4085", "SR-1234")
- The fault message/description
- Include a relevant excerpt in source_chunk

DO NOT extract root causes or fixing steps yet.

Text:
---------------------
{text}
---------------------

Extract all fault codes."""

    program = LLMTextCompletionProgram.from_defaults(
        output_cls=FaultList,
        llm=llm,
        prompt_template_str=prompt_template_str,
        verbose=False
    )
    
    all_faults = []
    seen_codes = set()
    
    node_bar = tqdm(nodes, desc="Nodes", unit="node")
    for node_idx, node in enumerate(node_bar):
        try:
            result = program(text=node.text)
            try:
                prompt_used = prompt_template_str.format(text=node.text)
                # Combine faults into one JSON string for output estimation
                out_text = json.dumps([f.model_dump() for f in result.faults])
                token_tracker.record("faults_node", prompt_used, out_text)
                # Update tqdm postfix with live cost estimate
                cost = current_cost_estimate()
                if cost["total_cost"] is not None:
                    node_bar.set_postfix({
                        "tokens": f"{token_tracker.input_tokens + token_tracker.output_tokens}",
                        "cost": f"${cost['total_cost']:.4f}"
                    })
            except Exception:
                pass
            for fault in result.faults:
                if fault.fault_code not in seen_codes:
                    fault.source_node_index = node_idx  # Store where we found this fault
                    all_faults.append(fault)
                    seen_codes.add(fault.fault_code)
        except Exception as e:
            print(f"Error processing node: {e}")
            continue
    
    return all_faults

def extract_root_causes_for_fault(
    fault_code: str, 
    fault_message: str, 
    nodes, 
    source_node_index: Optional[int] = None,
    context_window: int = CONTEXT_WINDOW
) -> List[RootCause]:
    """Extract root causes for a fault by finding ALL nodes mentioning it plus surrounding context.
    
    Args:
        fault_code: The fault code
        fault_message: The fault message
        nodes: All document nodes (each ~1024 chars with 200 char overlap)
        source_node_index: Index of the node where fault was first found (used for fallback)
        context_window: Number of nodes before/after each mention to include (default: CONTEXT_WINDOW)
    """
    
    prompt_template_str = """For fault {fault_code} ({fault_message}), extract root causes and fixing sequences.

For each root cause:
- Provide a unique id (e.g., "RC-{fault_code}-001")
- Describe the root cause
- Provide fixing sequences with ordered steps
- Mark source as "manual"
- Provide a confidence score (0.0-1.0)

Text:
---------------------
{text}
---------------------

Extract root causes for this fault."""

    program = LLMTextCompletionProgram.from_defaults(
        output_cls=RootCauseList,
        llm=llm,
        prompt_template_str=prompt_template_str,
        verbose=False
    )
    
    # Find ALL nodes mentioning this fault (by code or message)
    matching_indices = set()
    for idx, node in enumerate(nodes):
        if fault_code.lower() in node.text.lower() or fault_message[:30].lower() in node.text.lower():
            # Add this node and surrounding context
            for offset in range(-context_window, context_window + 1):
                context_idx = idx + offset
                if 0 <= context_idx < len(nodes):
                    matching_indices.add(context_idx)
    
    # Convert to sorted list and get the nodes
    if matching_indices:
        relevant_indices = sorted(matching_indices)
        relevant_nodes = [nodes[i] for i in relevant_indices]
    else:
        # Fallback: if no matches found, use nodes around first occurrence
        if source_node_index is not None:
            start_idx = max(0, source_node_index - context_window)
            end_idx = min(len(nodes), source_node_index + context_window + 1)
            relevant_nodes = nodes[start_idx:end_idx]
        else:
            relevant_nodes = nodes[:5]  # Last resort: first few nodes
    
    # Combine relevant nodes (limit to prevent token overflow)
    combined_text = "\n\n---\n\n".join([node.text for node in relevant_nodes[:MAX_NODES_FOR_ROOT_CAUSE]])
    
    try:
        result = program(fault_code=fault_code, fault_message=fault_message, text=combined_text)
        try:
            prompt_used = prompt_template_str.format(fault_code=fault_code, fault_message=fault_message, text=combined_text)
            out_text = json.dumps([rc.model_dump() for rc in result.root_causes])
            token_tracker.record(f"root_causes_{fault_code}", prompt_used, out_text)
        except Exception:
            pass
        return result.root_causes
    except Exception as e:
        print(f"  Error extracting root causes for {fault_code}: {e}")
        return []

def main():
    # CLI args
    parser = argparse.ArgumentParser(description="Extract faults and metadata from a PDF manual using LlamaIndex (Azure OpenAI)")
    parser.add_argument("pdf", nargs="?", default=None, help="Path to the input PDF manual (optional, defaults to sample laser manual)")
    parser.add_argument("--out", dest="out_dir", default=None, help="Optional output directory. Defaults to manual_processing/data/llamaindex_outputs/<pdf_stem>/<timestamp>/")
    parser.add_argument("--show-costs", action="store_true", help="Print token usage and estimated costs at the end")
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.pdf is None:
        default_pdf = os.path.join(script_dir, "sample_manuals", "B-70254EN_01_101028_fanuc_laser_digital.pdf")
        pdf_path = os.path.abspath(default_pdf)
        print(f"No PDF argument supplied. Using default: {pdf_path}")
    else:
        pdf_path = os.path.abspath(args.pdf)
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Step 0: Parse PDF using pypdf
    print(f"Parsing {pdf_path} with pypdf...")
    parse_start = time.perf_counter()
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in tqdm(reader.pages, desc="Pages", unit="page"):
        page_text = page.extract_text() or ""
        full_text += page_text + "\n\n"
    parse_elapsed = time.perf_counter() - parse_start
    print(f"Parsed {len(reader.pages)} pages, {len(full_text)} characters in {parse_elapsed:.2f}s")
    
    # Create LlamaIndex document and split into nodes
    print("Creating document nodes with SentenceSplitter...")
    doc = Document(text=full_text)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents([doc])
    print(f"Created {len(nodes)} nodes")
    
    # Step 0: Extract Metadata
    meta_start = time.perf_counter()
    metadata = extract_metadata(full_text)
    meta_elapsed = time.perf_counter() - meta_start
    print("\n--- Metadata ---")
    print(metadata.model_dump_json(indent=2))
    print(f"Metadata extraction time: {meta_elapsed:.2f}s")
    
    # Step 1: Extract Fault Codes
    faults_start = time.perf_counter()
    all_faults = extract_faults_from_nodes(nodes)
    faults_elapsed = time.perf_counter() - faults_start
    print(f"\n--- Extracted {len(all_faults)} unique fault codes ---")
    print(f"Fault code extraction time: {faults_elapsed:.2f}s")
    
    # Step 2: Extract Root Causes for each fault
    print("\n--- Step 2: Extracting root causes and fixing sequences ---")
    root_causes_start = time.perf_counter()
    
    fault_db = {}
    root_cause_db = {}
    
    rc_bar = tqdm(all_faults, desc="Root Causes", unit="fault")
    for fault in rc_bar:
        # Extract root causes for this fault (searching only nearby nodes)
        root_causes = extract_root_causes_for_fault(
            fault.fault_code,
            fault.fault_message,
            nodes,
            source_node_index=fault.source_node_index
        )
        
        # Update fault with root causes
        fault.root_causes = root_causes
        
        # Store in databases (exclude internal source_node_index)
        fault_db[fault.fault_code] = {
            "fault_code": fault.fault_code,
            "fault_message": fault.fault_message,
            "source_chunk": fault.source_chunk,
            "root_cause_ids": [rc.id for rc in root_causes]
        }
        
        for rc in root_causes:
            root_cause_db[rc.id] = rc.model_dump()
    
    root_causes_elapsed = time.perf_counter() - root_causes_start
    # Update tqdm postfix with live cost estimate after processing all faults (final state)
    cost = current_cost_estimate()
    if cost["total_cost"] is not None:
        rc_bar.set_postfix({
            "tokens": f"{token_tracker.input_tokens + token_tracker.output_tokens}",
            "cost": f"${cost['total_cost']:.4f}"
        })
    
    # Save to files
    if args.out_dir:
        output_dir = os.path.abspath(args.out_dir)
    else:
        pdf_stem = Path(pdf_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(script_dir, "data", "llamaindex_outputs", pdf_stem, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save raw manual text as markdown
    markdown_path = os.path.join(output_dir, f"{Path(pdf_path).stem}.md")
    with open(markdown_path, "w", encoding="utf-8") as f_md:
        f_md.write(f"# Manual: {Path(pdf_path).name}\n\n")
        f_md.write(full_text)
    print(f"Saved markdown to {markdown_path}")
    
    save_start = time.perf_counter()
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)
    
    with open(os.path.join(output_dir, "fault_db.json"), "w") as f:
        json.dump(fault_db, f, indent=2)
    
    with open(os.path.join(output_dir, "root_cause_db.json"), "w") as f:
        json.dump(root_cause_db, f, indent=2)
    save_elapsed = time.perf_counter() - save_start
    
    print(f"\n--- Saved databases to {output_dir} ---")
    print(f"Markdown file: {markdown_path}")
    print(f"Metadata: {len(metadata.model_dump())} fields ( {meta_elapsed:.2f}s )")
    print(f"Fault DB: {len(fault_db)} faults (extract {faults_elapsed:.2f}s)")
    print(f"Root Cause DB: {len(root_cause_db)} root causes (extract {root_causes_elapsed:.2f}s)")
    print(f"File save time: {save_elapsed:.2f}s")
    total_elapsed = parse_elapsed + meta_elapsed + faults_elapsed + root_causes_elapsed + save_elapsed
    print(f"Total processing time: {total_elapsed:.2f}s")

    # Token usage & cost estimation
    usage = token_tracker.summary()
    if args.show_costs:
        model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        in_price = MODEL_PRICING_INPUT.get(model_name)
        out_price = MODEL_PRICING_OUTPUT.get(model_name)
        if in_price is None or out_price is None:
            print(f"\n--- Cost Estimation ---")
            print(f"Model '{model_name}' not in pricing table. Supported: {list(MODEL_PRICING_INPUT.keys())}")
        else:
            in_cost = (usage["input_tokens"] / 1_000_000) * in_price
            out_cost = (usage["output_tokens"] / 1_000_000) * out_price
            total_cost = in_cost + out_cost
            print(f"\n--- Token Usage & Estimated Costs ---")
            print(f"Model: {model_name}")
            print(f"Input tokens: {usage['input_tokens']:,} (${in_cost:.4f})")
            print(f"Output tokens: {usage['output_tokens']:,} (${out_cost:.4f})")
            print(f"Estimated total cost: ${total_cost:.4f}")
    
    # Print samples
    if fault_db:
        print("\n--- Fault DB Sample ---")
        print(json.dumps(list(fault_db.values())[:2], indent=2))
    
    if root_cause_db:
        print("\n--- Root Cause DB Sample ---")
        print(json.dumps(list(root_cause_db.values())[:2], indent=2))

if __name__ == "__main__":
    main()
