import argparse
import asyncio
import logging
import os
import sys
import faulthandler

faulthandler.enable()

# FIX: Set these BEFORE importing any Paddle/PyTorch/NumPy modules
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ldrs.ldrs_pipeline import LDRSConfig, LDRSPipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_indexer(pdf_path: str, output_dir: str, use_ocr: bool):
    try:
        config = LDRSConfig(
            results_dir=output_dir,
            pdf_dir=os.path.dirname(pdf_path),
            md_dir=output_dir,
            model="qwen3-vl",
        )
        pipeline = LDRSPipeline(config)
        md_path = await pipeline.index_document_from_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            md_filename=None,
            use_ocr=use_ocr,
        )
        logger.info(f"Successfully indexed document. Markdown saved to {md_path}")
    except Exception as e:
        logger.error(f"Failed to index document: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone LDRS Indexer")
    parser.add_argument(
        "--pdf_path", type=str, required=True, help="Path to the PDF file"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--use_ocr", action="store_true", help="Use OCR for text extraction"
    )

    args = parser.parse_args()

    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    asyncio.run(run_indexer(args.pdf_path, args.output_dir, args.use_ocr))
