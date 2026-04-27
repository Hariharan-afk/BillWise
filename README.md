@'
# BillWise

BillWise is a hybrid document understanding pipeline for automated restaurant expense management.

## Project Structure

- `main_workflow/`  
  Clean production-oriented runtime for the BillWise application.
- `experiments/`  
  Archived reference implementations kept for reproducibility:
  - `extraction_hybrid_reference/`
  - `categorization_reference/`
  - `app_shell_reference/`
- `checkpoints/`  
  Trained categorization model weights.
- `data/`  
  Labeled categorization datasets and other project data.

## Main Workflow Features

- Automated bill ingestion through Twilio webhook flow
- Duplicate detection using image hash and content-level checks
- Hybrid bill extraction using OCR/LayoutLM and VLM fallback
- Item abbreviation normalization
- Grocery / expense categorization
- Cloud CSV storage for extracted bill data
- Chat-based analytics interface
- Local CSV fallback for development without cloud access
- Gemini-powered advanced analytics for natural-language questions

## Environment Setup

Create and activate a virtual environment, then install dependencies.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt