# LlamaParse + Qwen Vision Streamlit Application

## Overview

This Streamlit application demonstrates a complete end-to-end workflow that combines **LlamaParse** for intelligent PDF parsing with **Qwen Vision** models for image analysis. The workflow transforms complex PDF documents into enriched markdown by automatically extracting images and generating detailed descriptions for them.

## Workflow Architecture

The application implements the following four-stage pipeline:

### Stage 1: PDF Parsing with LlamaParse

**Input**: PDF document  
**Process**: LlamaParse uses agentic OCR to parse the entire document into structured markdown  
**Output**: Markdown content + image metadata (page numbers, bounding boxes, image IDs)

LlamaParse is particularly effective at:
- Preserving document layout and structure
- Handling complex tables and charts
- Extracting text from scanned documents
- Identifying and cataloging embedded images

### Stage 2: Image Extraction from PDF

**Input**: PDF file + image metadata from Stage 1  
**Process**: PyMuPDF (fitz) extracts images using page numbers and bounding box coordinates  
**Output**: PNG image files with preserved quality

The extraction process:
1. Reads page number and bounding box from LlamaParse metadata
2. Opens the PDF and navigates to the specified page
3. Crops the region defined by the bounding box
4. Renders at 2x resolution for enhanced quality
5. Saves as PNG to temporary storage

### Stage 3: Image Analysis with Qwen Vision

**Input**: Extracted PNG images  
**Process**: OpenRouter Qwen vision model analyzes each image  
**Output**: Detailed text descriptions of visual content

The Qwen model provides:
- Comprehensive visual element descriptions
- Color and composition analysis
- Text recognition within images
- Contextual interpretation

### Stage 4: Description Splicing

**Input**: Original markdown + image descriptions  
**Process**: Regex-based pattern matching to locate image placeholders and insert descriptions  
**Output**: Enhanced markdown with descriptions integrated

The splicing logic handles multiple placeholder formats:
- `![Image 1]`
- `![Figure 2]`
- `![image_id]`

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip or conda package manager
- API keys for LlamaParse and OpenRouter

### Step 1: Clone or Navigate to Project

```bash
cd /home/ubuntu/llamaparse_qwen_app
```

### Step 2: Install Dependencies

```bash
pip install streamlit llama-parse PyMuPDF openai python-dotenv
```

### Step 3: Configure API Keys

Create a `.env` file in the project root:

```env
LLAMAPARSE_API_KEY=your_llamaparse_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Alternatively, provide API keys through the Streamlit sidebar when running the application.

### Step 4: Obtain API Keys

**LlamaParse API Key:**
1. Visit [LlamaParse](https://www.llamaindex.ai/llamaparse)
2. Sign up for a free account (includes 10,000 free credits)
3. Navigate to the API dashboard
4. Copy your API key

**OpenRouter API Key:**
1. Visit [OpenRouter](https://openrouter.ai)
2. Create an account
3. Navigate to API keys section
4. Generate and copy your API key
5. Add credits to your account (supports multiple payment methods)

## Running the Application

### Launch Streamlit App

```bash
streamlit run streamlit_app.py
```

The application will start and open in your default browser at `http://localhost:8501`.

### Using the Application

1. **Provide API Keys** (if not in `.env`):
   - Enter LlamaParse API Key in the sidebar
   - Enter OpenRouter API Key in the sidebar

2. **Upload PDF**:
   - Click "Choose a PDF file" button
   - Select your PDF document

3. **Process**:
   - Click the "🚀 Process PDF" button
   - Monitor progress through the four stages

4. **Review Results**:
   - View parsed markdown in expandable sections
   - Examine extracted images with descriptions
   - Review final enhanced markdown

5. **Download**:
   - Click "📥 Download Enhanced Markdown" button
   - Save the enriched markdown file

## Code Structure

### Main Functions

#### `parse_pdf_with_llamaparse(pdf_file_path: str) -> tuple[str, list]`

Parses a PDF using LlamaParse and extracts markdown and image metadata.

**Parameters:**
- `pdf_file_path`: Path to the PDF file

**Returns:**
- `markdown_content`: Parsed markdown text
- `image_metadata`: List of image metadata dictionaries

**Example:**
```python
markdown, images = parse_pdf_with_llamaparse("document.pdf")
```

#### `extract_image_from_pdf(pdf_path: str, image_info: dict, output_dir: str = None) -> str`

Extracts a single image from a PDF based on metadata.

**Parameters:**
- `pdf_path`: Path to the PDF file
- `image_info`: Dictionary with `page_number`, `bbox`, and `id`
- `output_dir`: Optional output directory (defaults to temp)

**Returns:**
- Path to extracted image file

**Example:**
```python
image_path = extract_image_from_pdf(
    "document.pdf",
    {"page_number": 1, "bbox": [0, 0, 100, 100], "id": "img_1"}
)
```

#### `describe_image_with_qwen(image_path: str) -> str`

Sends an image to Qwen vision model for analysis.

**Parameters:**
- `image_path`: Path to the image file

**Returns:**
- Detailed description of the image

**Example:**
```python
description = describe_image_with_qwen("extracted_image.png")
```

#### `splice_descriptions_into_markdown(markdown_content: str, image_descriptions: dict) -> str`

Integrates image descriptions into the original markdown.

**Parameters:**
- `markdown_content`: Original markdown from LlamaParse
- `image_descriptions`: Dictionary mapping image IDs to descriptions

**Returns:**
- Enhanced markdown with descriptions

**Example:**
```python
enhanced = splice_descriptions_into_markdown(
    markdown,
    {"img_1": "A blue rectangle with white text"}
)
```

## Configuration Options

### LlamaParse Settings

The LlamaParse parser can be customized with additional options:

```python
parser = LlamaParse(
    api_key=LLAMAPARSE_API_KEY,
    result_type="markdown",           # Output format
    parse_image_metadata=True,        # Extract image info
    # Additional options:
    # language="en",                  # Language
    # skip_diagonal_text=True,        # Skip angled text
    # use_ocr=True,                   # Force OCR mode
)
```

### Qwen Vision Model Selection

Different Qwen models are available through OpenRouter:

- `qwen/qwen-vl-chat` - Standard vision model
- `qwen/qwen2.5-vl-32b-instruct` - Larger, more capable model
- `qwen/qwen3.6-plus` - Latest generation

Update the `QWEN_VISION_MODEL` variable to switch models.

## Advanced Usage

### Batch Processing Multiple PDFs

```python
import glob

pdf_files = glob.glob("documents/*.pdf")

for pdf_file in pdf_files:
    markdown, images = parse_pdf_with_llamaparse(pdf_file)
    # Process images and splice descriptions
    # Save results
```

### Custom Image Description Prompts

Modify the prompt in `describe_image_with_qwen()`:

```python
"type": "text",
"text": "Describe this image for a technical document. Focus on: 1) Main elements, 2) Data/information shown, 3) Any labels or annotations."
```

### Filtering Images by Type

```python
# Only process images larger than a certain size
large_images = [img for img in image_metadata if (img['bbox'][2] - img['bbox'][0]) > 100]
```

## Error Handling & Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "LLAMAPARSE_API_KEY not set" | Missing API key | Add key to `.env` or sidebar |
| "Cannot extract image" | Invalid bounding box | Verify image metadata from LlamaParse |
| "Qwen API error" | Invalid API key or insufficient credits | Check OpenRouter account and balance |
| "PDF parsing fails" | Corrupted or unsupported PDF | Try a different PDF or convert format |
| "Memory error with large PDFs" | Large file size | Process PDFs in smaller chunks |

### Debugging

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check Streamlit logs:

```bash
streamlit run streamlit_app.py --logger.level=debug
```

## Performance Considerations

### Optimization Tips

1. **Batch API Calls**: Process multiple images concurrently (requires async implementation)
2. **Image Compression**: Reduce image size before sending to Qwen
3. **Caching**: Cache LlamaParse results for repeated documents
4. **Model Selection**: Use faster models for large-scale processing

### Cost Estimation

**LlamaParse**: ~1-2 credits per page (10,000 free credits)  
**OpenRouter Qwen**: Variable pricing based on model and image size

Example for 10-page PDF with 5 images:
- LlamaParse: ~10-20 credits
- Qwen Vision: ~$0.01-0.05 (depending on model)

## API Reference

### LlamaParse

**Documentation**: [LlamaParse Docs](https://developers.llamaindex.ai/llamaparse/parse/)

Key features:
- Markdown output with layout preservation
- Image metadata extraction
- Table-to-spreadsheet conversion
- OCR for scanned documents
- Custom parsing instructions

### OpenRouter Qwen

**Documentation**: [OpenRouter API](https://openrouter.ai/docs)

Supported models:
- Qwen VL Chat
- Qwen 2.5 VL 32B
- Qwen 3.6 Plus

### PyMuPDF

**Documentation**: [PyMuPDF Docs](https://pymupdf.readthedocs.io/)

Key functions:
- `fitz.open()` - Open PDF
- `page.get_pixmap()` - Render page to image
- `fitz.Rect()` - Define bounding box

## Limitations & Future Enhancements

### Current Limitations

- Single-threaded image processing (processes images sequentially)
- No support for video or audio content
- Limited to PNG output for extracted images
- No built-in image quality validation

### Planned Enhancements

- [ ] Parallel image processing for faster throughput
- [ ] Support for multiple output formats (JPEG, WebP)
- [ ] Image quality assessment and filtering
- [ ] Custom description templates
- [ ] Batch processing UI
- [ ] Result caching and history
- [ ] Integration with vector databases for RAG

## Contributing

To extend or modify the application:

1. **Add new features** in separate functions
2. **Update documentation** with new capabilities
3. **Test thoroughly** with various PDF types
4. **Optimize performance** before deployment

## License

This application is provided as-is for educational and commercial use.

## Support & Resources

- **LlamaParse Support**: [LlamaParse Discord](https://discord.gg/llamaindex)
- **OpenRouter Support**: [OpenRouter Docs](https://openrouter.ai/docs)
- **Streamlit Community**: [Streamlit Forum](https://discuss.streamlit.io/)

## Example Workflow

### Input PDF

A technical document with:
- 15 pages of text
- 8 embedded diagrams
- 3 data charts
- 2 screenshots

### Processing

1. **Parse**: LlamaParse converts to markdown (~20 credits)
2. **Extract**: 8 images extracted at 2x resolution
3. **Analyze**: Qwen describes each image (~$0.02)
4. **Enhance**: Descriptions spliced into markdown

### Output

Enhanced markdown file with:
- Original text content
- Structured tables from charts
- Detailed image descriptions
- Preserved layout and hierarchy

Total processing time: ~2-3 minutes  
Total cost: ~$0.03 + LlamaParse credits

---

**Created by**: Manus AI  
**Last Updated**: May 2026  
**Version**: 1.0.0
