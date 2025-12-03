"""
PDF Sorting Script - Sort PDFs into Digital and Scanned Categories

This script:
1. Scans PDFs in the manuals folder and classifies them as:
   - Digital (text-extractable, ready for processing)
   - Scanned (image-based, requires OCR)

2. [Future] Uses Azure OpenAI to filter digital PDFs by usefulness/relevance
   for the knowledge graph project.

The script uses text extraction to determine if a PDF is digital or scanned.
If text can be extracted from at least 30% of pages, it's considered digital.
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Tuple, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from pypdf import PdfReader
from tqdm import tqdm
import json

# Azure OpenAI imports (for future usefulness filtering)
from llama_index.llms.azure_openai import AzureOpenAI
from pydantic import BaseModel, Field

load_dotenv()

# --- Configuration Constants ---

# PDF classification thresholds
MIN_TEXT_RATIO = 0.3  # At least 30% of pages must have extractable text
MIN_CHARS_PER_PAGE = 100  # Minimum characters per page to count as "has text"
MAX_PAGES_TO_SAMPLE = 10  # Sample first N pages for classification (performance)

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
    return max(1, int(len(text) / 4))

class TokenTracker:
    """Track token usage and costs for Azure OpenAI API calls."""
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

def current_cost_estimate(tracker: TokenTracker) -> dict:
    """Compute current estimated costs using selected model."""
    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    in_price = MODEL_PRICING_INPUT.get(model_name)
    out_price = MODEL_PRICING_OUTPUT.get(model_name)
    usage_input = tracker.input_tokens
    usage_output = tracker.output_tokens
    if in_price is None or out_price is None:
        return {"model": model_name, "in_cost": None, "out_cost": None, "total_cost": None}
    in_cost = (usage_input / 1_000_000) * in_price
    out_cost = (usage_output / 1_000_000) * out_price
    return {"model": model_name, "in_cost": in_cost, "out_cost": out_cost, "total_cost": in_cost + out_cost}


# --- Data Models for Future AI Filtering ---

class PDFUsefulnessScore(BaseModel):
    """Assessment of PDF usefulness for knowledge graph extraction."""
    is_useful: bool = Field(description="Whether the PDF is useful for technical knowledge extraction")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation of the decision")
    document_type: str = Field(description="Type of document (e.g., manual, catalog, form, advertisement)")


# --- PDF Classification Functions ---

def classify_pdf(pdf_path: Path) -> Tuple[str, dict]:
    """
    Classify a PDF as 'digital' or 'scanned' based on text extractability.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Tuple of (classification, metadata_dict)
        - classification: 'digital' or 'scanned'
        - metadata_dict: Information about the PDF
    """
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        # Sample pages for classification (avoid processing huge PDFs entirely)
        pages_to_check = min(total_pages, MAX_PAGES_TO_SAMPLE)
        pages_with_text = 0
        total_chars = 0
        
        for i in range(pages_to_check):
            try:
                text = reader.pages[i].extract_text()
                char_count = len(text.strip())
                total_chars += char_count
                
                if char_count >= MIN_CHARS_PER_PAGE:
                    pages_with_text += 1
            except Exception as e:
                # Page extraction failed, likely scanned/corrupted
                continue
        
        text_ratio = pages_with_text / pages_to_check if pages_to_check > 0 else 0
        avg_chars_per_page = total_chars / pages_to_check if pages_to_check > 0 else 0
        
        # Classify based on text ratio
        classification = "digital" if text_ratio >= MIN_TEXT_RATIO else "scanned"
        
        metadata = {
            "total_pages": total_pages,
            "pages_checked": pages_to_check,
            "pages_with_text": pages_with_text,
            "text_ratio": round(text_ratio, 2),
            "avg_chars_per_page": round(avg_chars_per_page, 1),
            "classification": classification,
            "file_size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2)
        }
        
        return classification, metadata
        
    except Exception as e:
        # If we can't read the PDF at all, consider it problematic/scanned
        metadata = {
            "error": str(e),
            "classification": "scanned",
            "total_pages": 0
        }
        return "scanned", metadata


def find_all_pdfs(root_dir: Path) -> List[Path]:
    """Recursively find all PDF files in the directory."""
    return list(root_dir.rglob("*.pdf")) + list(root_dir.rglob("*.PDF"))


def sort_pdfs(manuals_dir: Path, output_dir: Path, dry_run: bool = False) -> dict:
    """
    Sort PDFs from manuals_dir into digital/scanned subdirectories.
    
    Args:
        manuals_dir: Source directory containing PDFs
        output_dir: Destination directory for sorted PDFs
        dry_run: If True, only report what would be done without moving files
        
    Returns:
        Dictionary with sorting statistics
    """
    # Find all PDFs
    print(f"Scanning for PDFs in: {manuals_dir}")
    pdf_files = find_all_pdfs(manuals_dir)
    print(f"Found {len(pdf_files)} PDF files\n")
    
    if len(pdf_files) == 0:
        print("No PDFs found to process.")
        return {"total": 0, "digital": 0, "scanned": 0, "errors": 0}
    
    # Create output directories
    digital_dir = output_dir / "digital"
    scanned_dir = output_dir / "scanned"
    
    if not dry_run:
        digital_dir.mkdir(parents=True, exist_ok=True)
        scanned_dir.mkdir(parents=True, exist_ok=True)
    
    # Process PDFs
    stats = {
        "total": len(pdf_files),
        "digital": 0,
        "scanned": 0,
        "errors": 0,
        "details": []
    }
    
    print("Classifying and sorting PDFs...")
    for pdf_path in tqdm(pdf_files):
        try:
            classification, metadata = classify_pdf(pdf_path)
            
            # Determine destination
            dest_dir = digital_dir if classification == "digital" else scanned_dir
            
            # Create relative path structure
            try:
                rel_path = pdf_path.relative_to(manuals_dir)
            except ValueError:
                # If file is not under manuals_dir, use just the filename
                rel_path = Path(pdf_path.name)
            
            dest_path = dest_dir / rel_path
            
            # Record details
            file_info = {
                "source": str(pdf_path),
                "destination": str(dest_path),
                "classification": classification,
                "metadata": metadata
            }
            stats["details"].append(file_info)
            
            # Update counters
            if classification == "digital":
                stats["digital"] += 1
            else:
                stats["scanned"] += 1
            
            # Move or copy file
            if not dry_run:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pdf_path, dest_path)
            
        except Exception as e:
            stats["errors"] += 1
            error_info = {
                "source": str(pdf_path),
                "error": str(e),
                "classification": "error"
            }
            stats["details"].append(error_info)
            print(f"\nError processing {pdf_path}: {e}")
    
    return stats


# --- Future: AI-based Usefulness Filtering ---

def setup_azure_openai() -> Optional[AzureOpenAI]:
    """
    Initialize Azure OpenAI LLM for usefulness filtering.
    Returns None if credentials are not configured.
    """
    try:
        llm = AzureOpenAI(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0.3
        )
        return llm
    except Exception as e:
        print(f"Warning: Could not initialize Azure OpenAI: {e}")
        return None


def assess_pdf_usefulness(pdf_path: Path, llm: AzureOpenAI, tracker: TokenTracker) -> PDFUsefulnessScore:
    """
    [FUTURE IMPLEMENTATION]
    Use Azure OpenAI to assess whether a PDF is useful for knowledge extraction.
    
    This would analyze:
    - Document type (manual vs. advertisement vs. form)
    - Technical content quality
    - Relevance to machinery/equipment knowledge
    
    Args:
        pdf_path: Path to the PDF file
        llm: Azure OpenAI instance
        tracker: Token usage tracker
        
    Returns:
        PDFUsefulnessScore with assessment
    """
    # TODO: Implementation placeholder for future development
    # This would:
    # 1. Extract first few pages of text
    # 2. Send to LLM with structured prompt
    # 3. Get back usefulness assessment
    # 4. Track token usage
    
    raise NotImplementedError("AI-based usefulness filtering not yet implemented")


# --- Main Function ---

def main():
    parser = argparse.ArgumentParser(
        description="Sort PDFs into digital (readable) and scanned (needs OCR) categories"
    )
    parser.add_argument(
        "--manuals-dir",
        type=Path,
        default=Path(__file__).parent.parent / "manuals",
        help="Directory containing PDF files to sort (default: ../manuals)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "sorted_pdfs",
        help="Directory to output sorted PDFs (default: ../sorted_pdfs)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done without actually moving files"
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Save detailed JSON report to this file"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.manuals_dir.exists():
        print(f"Error: Manuals directory not found: {args.manuals_dir}")
        return 1
    
    # Run sorting
    print(f"{'=' * 60}")
    print(f"PDF Sorting Script")
    print(f"{'=' * 60}")
    print(f"Source: {args.manuals_dir}")
    print(f"Destination: {args.output_dir}")
    print(f"Mode: {'DRY RUN (no files moved)' if args.dry_run else 'LIVE (files will be copied)'}")
    print(f"{'=' * 60}\n")
    
    start_time = datetime.now()
    stats = sort_pdfs(args.manuals_dir, args.output_dir, args.dry_run)
    end_time = datetime.now()
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Sorting Complete")
    print(f"{'=' * 60}")
    print(f"Total PDFs processed: {stats['total']}")
    print(f"  - Digital (readable): {stats['digital']}")
    print(f"  - Scanned (needs OCR): {stats['scanned']}")
    print(f"  - Errors: {stats['errors']}")
    print(f"Time elapsed: {end_time - start_time}")
    print(f"{'=' * 60}")
    
    # Save report if requested
    if args.report:
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "manuals_dir": str(args.manuals_dir),
            "output_dir": str(args.output_dir),
            "dry_run": args.dry_run,
            "duration_seconds": (end_time - start_time).total_seconds(),
            "statistics": stats
        }
        
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed report saved to: {args.report}")
    
    return 0


if __name__ == "__main__":
    exit(main())
