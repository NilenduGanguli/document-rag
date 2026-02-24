FROM python:3.11-slim

WORKDIR /app

# System libraries:
#   libgl1 libglib2.0-0       — OpenCV runtime
#   poppler-utils             — pdf2image needs pdftoppm
#   tesseract-ocr             — pytesseract OCR fallback
#   libmagic1                 — python-magic MIME detection
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Main dependency install.
# unstructured[pdf] is excluded from requirements.txt because its transitive
# dep unstructured-inference==0.7.23 pins onnxruntime<1.16, which conflicts
# with fastembed (>=1.17) and gliner.
RUN pip install --no-cache-dir -r requirements.txt

# Install unstructured-inference bypassing its outdated onnxruntime pin.
# The layout inference code works correctly with onnxruntime>=1.17 at runtime.
# layoutparser and opencv-python-headless are its key runtime deps.
# pdfplumber is pinned <0.7 so it does not upgrade pdfminer.six past 20220319;
# unstructured==0.12.5 imports PSSyntaxError from pdfminer.pdfparser, which was
# only re-exported there through pdfminer.six ~20220319 — later versions dropped it.
# After layoutparser, force numpy<2 and pandas back: layoutparser pulls numpy 2.x
# which breaks thinc (spaCy) and fastembed at runtime.
RUN pip install --no-cache-dir --no-deps "unstructured-inference==0.7.23" \
    && pip install --no-cache-dir \
        "layoutparser==0.3.4" \
        "pdfplumber>=0.6.0,<0.7" \
        "opencv-python-headless!=4.7.0.68" \
        "timm" \
    && pip install --no-cache-dir --force-reinstall \
        "numpy>=1.21,<2.0.0" \
        "pandas==2.2.1"

# Download spaCy model so it's available without internet access at runtime.
# Using direct pip install to avoid a URL-construction bug in `spacy download`
# when urllib3/chardet versions mismatch.
RUN pip install --no-cache-dir \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"

# Download NLTK corpora required by unstructured at document-parsing time.
# punkt_tab / punkt        — sentence tokenizer
# averaged_perceptron_tagger / averaged_perceptron_tagger_eng — POS tagger
RUN python -c "\
import nltk; \
nltk.download('punkt_tab', quiet=True); \
nltk.download('punkt', quiet=True); \
nltk.download('averaged_perceptron_tagger', quiet=True); \
nltk.download('averaged_perceptron_tagger_eng', quiet=True)"

COPY . .
RUN chmod +x /app/set-env.sh

ENTRYPOINT ["/app/set-env.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
