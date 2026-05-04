#!/usr/bin/env python3
"""
LlamaParse + Qwen Vision Streamlit Application

This application demonstrates a complete workflow:
1. Parse PDF documents using LlamaParse (returns markdown)
2. Scan markdown for image references and extract images from PDF
3. Send extracted images to OpenRouter Qwen vision model for descriptions
4. Splice descriptions back into the markdown content

Author: Manus AI
"""

import os
import base64
import re
import tempfile
from pathlib import Path
import fitz  # PyMuPDF
import streamlit as st
from llama_parse import LlamaParse
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== Configuration ====================
LLAMAPARSE_API_KEY = os.environ.get("LLAMA_PARSE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPEN_ROUTER_API_KEY")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
QWEN_VISION_MODEL = "qwen/qwen2.5-vl-72b-instruct"

# ==================== Streamlit Page Config ====================
st.set_page_config(
    page_title="LlamaParse + Qwen Vision",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Helper Functions ====================

def parse_pdf_with_llamaparse(pdf_file_path: str) -> tuple[str, list]:
    """
    Parses a PDF document using LlamaParse and returns markdown and image metadata.
    Falls back to direct image extraction if LlamaParse metadata is empty.
    
    Args:
        pdf_file_path: Path to the PDF file
        
    Returns:
        Tuple of (markdown_content, image_metadata)
    """
    if not LLAMAPARSE_API_KEY:
        raise ValueError("LLAMAPARSE_API_KEY environment variable not set.")

    with st.spinner("🔄 Parsing PDF with LlamaParse..."):
        try:
            parser = LlamaParse(
                api_key=LLAMAPARSE_API_KEY,
                result_type="markdown",
                parse_image_metadata=True
            )
            documents = parser.load_data(pdf_file_path)

            markdown_content = ""
            image_metadata = []

            for doc in documents:
                markdown_content += doc.text
                if doc.metadata and "image_metadata" in doc.metadata:
                    image_metadata.extend(doc.metadata["image_metadata"])

            # Fallback: If LlamaParse didn't detect images, extract directly from PDF
            if not image_metadata:
                st.info("⚠️ LlamaParse metadata empty. Using fallback image extraction...")
                image_metadata = extract_all_images_from_pdf(pdf_file_path)
            
            st.success(f"✅ Successfully parsed PDF. Found {len(image_metadata)} images.")
            return markdown_content, image_metadata
        except Exception as e:
            st.error(f"❌ Error parsing PDF: {str(e)}")
            raise


def extract_all_images_from_pdf(pdf_path: str) -> list:
    """
    Extracts ALL images from a PDF directly using PyMuPDF.
    Used as fallback when LlamaParse metadata is empty.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of image metadata dictionaries
    """
    try:
        doc = fitz.open(pdf_path)
        image_metadata = []
        image_counter = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get all images on this page
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                # Get image bounding box
                rects = page.get_image_rects(xref)
                
                if rects:
                    rect = rects[0]
                    bbox = [rect.x0, rect.y0, rect.x1, rect.y1]
                else:
                    # Fallback bbox if not found
                    bbox = [0, 0, pix.width, pix.height]
                
                image_metadata.append({
                    "id": image_counter,
                    "page_number": page_num + 1,
                    "bbox": bbox,
                    "xref": xref,
                    "width": pix.width,
                    "height": pix.height
                })
                image_counter += 1
        
        doc.close()
        return image_metadata
        
    except Exception as e:
        st.error(f"❌ Error extracting images: {str(e)}")
        return []


def extract_image_from_pdf(pdf_path: str, image_info: dict, output_dir: str = None) -> str:
    """
    Extracts an image from a PDF based on LlamaParse image metadata.
    
    Args:
        pdf_path: Path to the PDF file
        image_info: Image metadata dictionary with page_number, bbox, and id
        output_dir: Directory to save extracted images (defaults to temp directory)
        
    Returns:
        Path to the extracted image file
    """
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        doc = fitz.open(pdf_path)
        page_number = image_info.get("page_number", 0)
        bbox = image_info.get("bbox", [0, 0, 100, 100])
        image_id = image_info.get("id", "unknown")
        xref = image_info.get("xref", None)

        # If xref is available (from fallback extraction), use direct image extraction
        if xref is not None:
            try:
                pix = fitz.Pixmap(doc, xref)
                # Convert CMYK to RGB if needed
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                output_image_path = os.path.join(output_dir, f"image_{image_id}_page_{page_number}.png")
                pix.save(output_image_path)
                doc.close()
                return output_image_path
            except Exception as e:
                pass  # Fall through to bbox method
        
        # Fallback: Extract using bounding box
        if page_number > 0:
            page = doc[page_number - 1]
        else:
            page = doc[0]
            
        rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
        pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=rect)

        output_image_path = os.path.join(output_dir, f"image_{image_id}_page_{page_number}.png")
        pix.save(output_image_path)
        doc.close()

        return output_image_path
    except Exception as e:
        st.error(f"❌ Error extracting image: {str(e)}")
        raise
def get_context(markdown, pos, window=500):
    start = max(0, pos - window)
    end = min(len(markdown), pos + window)
    return markdown[start:end]

def describe_image_with_qwen(image_path, context_text="", rules_text=""):
    """
    Sends an image + optional context to a vision-capable model via OpenRouter
    and returns a protocol-aware structured description.
    """

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    try:
        client = OpenAI(
            base_url=OPENROUTER_API_BASE,
            api_key=OPENROUTER_API_KEY,
        )

        # Convert image → base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # 🔥 PROTOCOL-AWARE PROMPT

        prompt = f"""
You are a digital protocol analyst.

You are given:
1. A diagram (timing / dependency / block)
2. Context from the document
3. Extracted protocol rules

Your job is to INTERPRET the diagram using the rules.

-------------------------------------

Protocol Rules:
{rules_text}

Context:
{context_text}

-------------------------------------

Step 1: Identify diagram type:
- Timing Diagram
- Dependency Diagram
- Block Diagram

Step 2: Extract:
- Signals
- Relationships (timing or dependencies)

Step 3: Apply rules:
- Validate relationships using rules
- Correct any contradictions


Output a structured 2-3 line description that explains the key protocol interactions shown in the image, referencing specific signals and timing relationships but do not include any additional commentary such as protocol rules information.
"""

        response = client.chat.completions.create(
            model=QWEN_VISION_MODEL,  # e.g. "qwen/qwen2.5-vl-72b-instruct"
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        },
                    ],
                }
            ],
        )

        description = response.choices[0].message.content
        return description

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise

def splice_descriptions_into_markdown(markdown_content: str, image_descriptions: list) -> str:
    
    figure_pattern = r"(Figure\s+[A-Za-z0-9\.\-: ]+)"

    matches = list(re.finditer(figure_pattern, markdown_content))

    for i, match in enumerate(matches):
        if i >= len(image_descriptions):
            break

        description = image_descriptions[i]

        insert_text = f"{match.group(0)}\n\n** ✅ Qwen Description:**\n{description}\n"

        markdown_content = (
            markdown_content[:match.start()] +
            insert_text +
            markdown_content[match.end():]
        )

    return markdown_content
def extract_protocol_rules(markdown: str) -> str:
    """
    Extracts generic protocol rules from document text
    """
    keywords = ["must", "must not", "only after", "before", "after", "wait", "assert"]

    lines = markdown.split("\n")
    rules = []

    for line in lines:
        if any(k in line.lower() for k in keywords):
            rules.append(line.strip())

    return "\n".join(rules[:50])  # limit size
# ==================== Streamlit UI ====================

def main():
    # Header
    st.title("📄 LlamaParse + Qwen Vision Workflow")
    st.markdown(
        "Transform PDFs into enriched markdown by combining LlamaParse parsing with Qwen vision model image descriptions."
    )

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key_llamaparse = st.text_input(
            "LlamaParse API Key",
            value=LLAMAPARSE_API_KEY or "",
            type="password",
            help="Your LlamaParse API key from https://www.llamaindex.ai/llamaparse"
        )
        
        api_key_openrouter = st.text_input(
            "OpenRouter API Key",
            value=OPENROUTER_API_KEY or "",
            type="password",
            help="Your OpenRouter API key from https://openrouter.ai"
        )
        
        st.divider()
        st.markdown("### 📚 How it works:")
        st.markdown("""
        1. **Upload PDF**: Select a PDF document to process
        2. **Parse**: LlamaParse converts the PDF to markdown
        3. **Extract**: Images are extracted from the PDF
        4. **Describe**: Qwen vision model analyzes each image
        5. **Enhance**: Descriptions are spliced into the markdown
        6. **Download**: Get the enriched markdown file
        """)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📤 Upload & Process")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Select a PDF document to parse and enhance"
        )

        if uploaded_file is not None:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                pdf_path = tmp_file.name

            st.success(f"✅ File uploaded: {uploaded_file.name}")

            # Process button
            if st.button("🚀 Process PDF", type="primary", use_container_width=True):
                # Validate API keys
                if not api_key_llamaparse:
                    st.error("❌ Please provide LlamaParse API Key")
                elif not api_key_openrouter:
                    st.error("❌ Please provide OpenRouter API Key")
                else:
                    # Set API keys from sidebar input
                    os.environ["LLAMAPARSE_API_KEY"] = api_key_llamaparse
                    os.environ["OPENROUTER_API_KEY"] = api_key_openrouter

                    try:
                        # Step 1: Parse PDF
                        st.info("📖 Step 1: Parsing PDF with LlamaParse...")
                        markdown_content, image_metadata = parse_pdf_with_llamaparse(pdf_path)

                        # Display markdown preview
                        with st.expander("📝 View Parsed Markdown", expanded=False):
                            st.markdown(markdown_content)

                        # Step 2: Extract and describe images
                        if image_metadata:
                            st.info(f"🖼️ Step 2: Processing {len(image_metadata)} images...")
                            
                            image_descriptions = []
                            progress_bar = st.progress(0)
                            
                            rules_text = extract_protocol_rules(markdown_content)

                            for idx, img_info in enumerate(image_metadata):

                                image_path = extract_image_from_pdf(pdf_path, img_info)
                                approx_pos = int(len(markdown_content) * (idx / max(1, len(image_metadata))))
                                context_text = get_context(markdown_content, approx_pos)
                                description = describe_image_with_qwen(
                                    image_path,
                                    context_text,
                                    rules_text
                                )
                                image_descriptions.append(description)
                                progress_bar.progress((idx + 1) / len(image_metadata))
                            # Step 3: Splice descriptions
                            st.info("🔗 Step 3: Splicing descriptions into markdown...")
                            final_markdown = splice_descriptions_into_markdown(markdown_content, image_descriptions)
                            
                            # Display final result
                            st.success("✅ Processing complete!")
                            
                            with st.expander("📄 View Enhanced Markdown", expanded=True):
                                st.markdown(final_markdown)
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Enhanced Markdown",
                                data=final_markdown,
                                file_name=f"{Path(uploaded_file.name).stem}_enhanced.md",
                                mime="text/markdown",
                                use_container_width=True
                            )
                        else:
                            st.info("ℹ️ No images found in the PDF. Here's the parsed markdown:")
                            with st.expander("📝 View Parsed Markdown", expanded=True):
                                st.markdown(markdown_content)
                            
                            # Download button for markdown without images
                            st.download_button(
                                label="📥 Download Markdown",
                                data=markdown_content,
                                file_name=f"{Path(uploaded_file.name).stem}.md",
                                mime="text/markdown",
                                use_container_width=True
                            )

                    except Exception as e:
                        st.error(f"❌ An error occurred: {str(e)}")
                    finally:
                        # Clean up temporary file
                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)

    with col2:
        st.header("ℹ️ Information")
        
        st.markdown("### 🔧 Technologies")
        st.markdown("""
        - **LlamaParse**: AI-powered PDF parsing
        - **Qwen Vision**: Multimodal image analysis
        - **PyMuPDF**: PDF image extraction
        - **Streamlit**: Web interface
        """)
        
        st.markdown("### 📊 Workflow Steps")
        st.markdown("""
        1. **Parse** → Markdown
        2. **Extract** → Images
        3. **Analyze** → Descriptions
        4. **Enhance** → Final Markdown
        """)
        
        st.markdown("### 🔗 Links")
        col_links1, col_links2 = st.columns(2)
        with col_links1:
            st.markdown("[LlamaParse](https://www.llamaindex.ai/llamaparse)")
            st.markdown("[OpenRouter](https://openrouter.ai)")
        with col_links2:
            st.markdown("[Qwen Models](https://huggingface.co/Qwen)")
            st.markdown("[Streamlit](https://streamlit.io)")


if __name__ == "__main__":
    main()
