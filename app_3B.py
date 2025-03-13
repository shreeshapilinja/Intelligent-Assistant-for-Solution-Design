import gradio as gr
import whisper
import ollama
import os
import torch
import re
import subprocess
from pathlib import Path
import shutil

# Load Whisper model with GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base.en", device=device)

# Function to transcribe video/audio files or read text files
def transcribe_file(file, progress=gr.Progress()):
    progress(0, desc="Starting transcription...")
    try:
        if file.name.endswith('.txt'):
            with open(file.name, 'r', encoding='utf-8') as f:
                transcript = f.read()
        elif file.name.endswith(('.mp4', '.webm', '.mp3')):
            progress(0.2, desc="Transcribing audio/video...")
            result = whisper_model.transcribe(file.name, fp16=(device == "cuda"))
            transcript = result["text"]
        else:
            raise ValueError("Unsupported file type. Please upload mp4, webm, mp3, or txt files.")
        
        if not transcript.strip():
            raise ValueError("Transcription resulted in empty text. Please check the file content.")
        
        # Basic cleaning
        transcript = re.sub(r'\s+', ' ', transcript).strip()
        filler_words = r'\b(um|uh|like|you know|so|basically)\b'
        transcript = re.sub(filler_words, '', transcript, flags=re.IGNORECASE)
        transcript = re.sub(r'\s+', ' ', transcript).strip()
        progress(1.0, desc="Transcription complete.")
        return transcript
    except Exception as e:
        raise Exception(f"Transcription error: {str(e)}")

# Function to chunk the transcript
def chunk_transcript(transcript, chunk_size=4000, overlap=1000, progress=gr.Progress()):
    progress(0, desc="Chunking transcript...")
    words = transcript.split()
    chunks = []
    start = 0
    total_chunks = max(1, (len(words) // (chunk_size - overlap)) + 1)
    for i in range(total_chunks):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap if end < len(words) else len(words)
        progress((i + 1) / total_chunks, desc=f"Chunk {i + 1} of {total_chunks} created.")
    if not chunks:
        raise ValueError("No valid chunks created.")
    return chunks

# Classification prompt to determine meeting type
classification_prompt = """
Determine if the following meeting transcript chunk is from a 'Technical' or 'Non-technical' meeting.

- **Technical meetings** discuss specific business or technical problems requiring solutions (e.g., software development, system design, technical implementation).
- **Non-technical meetings** focus on general business strategies, planning, or management without technical depth.

Transcript chunk:
{chunk}

Return only: 'Technical' or 'Non-technical'
"""

# Updated technical extraction prompt
technical_extract_prompt = """
From the following meeting transcript chunk, extract all relevant information in a detailed and structured manner:

- **Summary**: Provide a concise yet detailed overview of the main topics discussed, decisions made, and key takeaways from this chunk. Aim for 3-5 sentences.

- **Key Points**: List all significant information or highlights as bullet points. Be thorough and include all notable details.

- **Action Items**: List specific tasks or actions assigned or mentioned. If none, state "None".

- **Problem Statements**: Identify the main business problems or challenges discussed in the transcript chunk. For each problem, provide a concise paragraph (2-3 sentences) that:
  - Describes the issue in the context of the business or process (e.g., QA testing).
  - Explains the specific challenge or inefficiency, including any examples or details mentioned.
  - States what the solution must achieve to address the problem.
  Ensure that each problem statement is a single, flowing paragraph without internal subheadings. If multiple issues are closely related (e.g., share a common theme or process like content comparison in QA), consolidate them into a single problem statement rather than splitting them into separate entries. Only create separate problem statements for distinctly unrelated issues.

  **Example**: "Manual comparison of web portal designs and PDFs against a golden standard in QA processes is time-consuming and error-prone, with the team spending significant effort verifying elements like button positions and colors while often missing minor deviations due to text positioning shifts in PDFs. These inefficiencies delay the verification process and compromise quality. A solution is needed to automate these comparisons, improve accuracy, and handle text layout variations effectively."

- **Solution Components**: Identify any parts of the solution discussed in this chunk, such as methodologies, techniques, or approaches for addressing the problems. Include specific details where possible.

- **Technology Suggestions**: Suggest specific tools, libraries, or frameworks that could be used in the solution, based on the discussion or your knowledge. For each, include a brief justification. Examples: "OpenCV - For robust image processing capabilities," "PyPDF2 - For reliable text extraction from PDFs."

- **Challenges**: Identify potential issues or limitations mentioned or that you can infer for the proposed solution.

Return the output in plain text with this format:

Summary:
[Summary text]

Key Points:
- [Key point 1]
- [Key point 2]
- ...

Action Items:
[Action items or "None"]

Problem Statements:
- [Concise problem statement paragraph 1]
- [Concise problem statement paragraph 2, if unrelated to 1]
- ...

Solution Components:
- [Solution component 1]
- [Solution component 2]
- ...

Technology Suggestions:
- [Tool/Library 1 - Justification]
- [Tool/Library 2 - Justification]
- ...

Challenges:
- [Challenge 1]
- [Challenge 2]
- ...

---

Transcript chunk:
{chunk}
"""

# Non-technical extraction prompt
non_technical_extract_prompt = """
From the following non-technical meeting transcript chunk, extract the following information:

- **Summary**: Provide a concise overview of the main topics discussed, decisions made, and key takeaways. Aim for 3-5 sentences.
- **Key Points**: List significant information or highlights as bullet points.
- **Action Items**: List specific tasks or actions assigned or mentioned. If none, state "None".

Return the output in plain text with this format:

Summary:
[Summary text]

Key Points:
- [Key point 1]
- [Key point 2]
- ...

Action Items:
[Action items or "None"]

---

Transcript chunk:
{chunk}
"""

# Updated technical report prompt
technical_report_prompt = """
Generate a comprehensive Markdown report for a technical meeting based on the following extracted information from transcript chunk(s):

{combined_chunks}

Use this exact structure and ensure the report is detailed enough to span 3-4 pages when rendered:

# Intelligent Assistant for Solution Design Report - Technical Meeting

## Summary
[Combine all summaries into a cohesive paragraph of 5-7 sentences. Provide a detailed overview of the meeting, covering main topics, decisions, and takeaways. Ensure it reads smoothly and reflects all chunks.]

## Key Points
- [Unique key point 1]
- [Unique key point 2]
- ...
[List all unique key points from all chunks as bullet points. Remove duplicates and ensure completeness.]

## Action Items
- [Unique action item 1]
- [Unique action item 2]
- ...
[List all unique action items from all chunks. If none, write "None".]

## Problem Statement - 1 
During the meeting, the following problem statements were identified related to the main topic or use case:
1. **[Problem Statement 1]**: [paragraph from extraction having all similar things in one]
...

[Compile all unique problem statements from the extracted chunks as a numbered list. Each entry should be a paragraph (2-3 sentences) that integrates the business context, specific challenge, and requirements without internal subheadings. Remove duplicates and ensure related issues are consolidated within a single statement where appropriate.] i.e if other problem statements are similar or related to main then include the sub in main no need as separate problem statement let the problem statement be long no problem.

## Solution Design for Problem Statement 1
[Provide a comprehensive solution design that addresses all identified problem statements. Structure the solution with subsections tailored to the problems]

For each subsection, elaborate on specific methodologies, techniques, or processes, ensuring the explanation is thorough and practical. Aim for a total section length that significantly enhances detail.]

## Technology Stack for Problem Statement 1
[Compile a list of language, specific tools, libraries, or frameworks from the technology suggestions across all chunks. Remove duplicates and ensure each entry includes a justification for its selection]  e.g., Python language, "OpenCV - Chosen for its robust image manipulation capabilities suitable for pixel-wise comparison." Ensure relevance to the solution design.]

## Challenges for Problem Statement 1
[Identify potential challenges or limitations of the proposed solution, based on the extracted information. Provide explanations for each challenge, focusing on implementation or deployment issues.]

'' try not to give this section i.e as most of problems sub problem only are discussed so put those in problem statement 1 only if very much unrelated then only create this section 
if other Problem Statement is there then only continue else stop here, other than 1 is discussed then same way as for 1st continue by starting from problem statement i.e 
2. **[Problem Statement 2]**: [paragraph from extraction having completely different from Problem Statement 1 i.e completely unrelated to 1] then again ## Problem Statement 2 , Solution Design , ....  ''

- Ensure the report is detailed, professional, and avoids leading spaces/tabs before headings or subheadings in Markdown.
"""

# Non-technical report prompt
non_technical_report_prompt = """
Generate a Markdown report for a non-technical meeting based on the following extracted information from transcript chunk(s):

{combined_chunks}

Use this structure:

# Intelligent Assistant Report - Non-Technical Meeting

## Summary
[Combine all summaries into a cohesive paragraph of 5-7 sentences.]

## Key Points
- [Unique key point 1]
- [Unique key point 2]
- ...

## Action Items
- [Unique action item 1]
- [Unique action item 2]
- ...
[If none, write "None".]

- Ensure the report is concise and professional.
"""

# Function to process with Ollama
def process_with_ollama(prompt, num_predict=8192, num_ctx=8192, progress=gr.Progress()):
    try:
        response = ollama.generate(
            model="qwen2.5:3b",
            prompt=prompt,
            options={"num_predict": num_predict, "num_ctx": num_ctx}
        )
        progress(1.0, desc="Processing complete.")
        return response.get('response', '').strip()
    except Exception as e:
        raise Exception(f"Ollama processing error: {str(e)}")

# Function to combine chunks
def combine_chunks(chunks):
    if len(chunks) == 1:
        return chunks[0]
    combined = ""
    for i, chunk in enumerate(chunks, 1):
        combined += f"Chunk {i}:\n{chunk}\n\n"
    return combined

# Function to save Markdown report
def save_markdown_report(report_md, progress=gr.Progress()):
    try:
        md_path = "report.md"
        with open(md_path, "w", encoding='utf-8') as f:
            f.write(report_md)
        progress(1.0, desc="Markdown report saved.")
        return md_path
    except Exception as e:
        raise Exception(f"Markdown file saving error: {str(e)}")

# Function to convert Markdown to PDF using mdpdf
def convert_md_to_pdf(md_path, progress=gr.Progress()):
    if shutil.which("mdpdf") is None:
        progress(1.0, desc="PDF conversion skipped (mdpdf not installed).")
        return None
    try:
        pdf_path = "report.pdf"
        subprocess.run(["mdpdf", md_path, "-o", pdf_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        progress(1.0, desc="PDF report generated.")
        return pdf_path
    except subprocess.CalledProcessError as e:
        raise Exception(f"PDF conversion error: {e.stderr}")
    except Exception as e:
        raise Exception(f"PDF conversion error: {str(e)}")

# Main function to generate report
def generate_report(file, progress=gr.Progress()):
    if file is None:
        return "Please upload a file (mp4, webm, mp3, or txt).", None, None
    
    try:
        # Step 1: Transcribe the file
        progress(0, desc="Transcribing file...")
        transcript = transcribe_file(file, progress)
        if not transcript:
            return "Transcript is empty. Please upload a valid file.", None, None
        
        # Step 2: Chunk the transcript
        progress(0, desc="Chunking transcript...")
        chunks = chunk_transcript(transcript, progress=progress)
        
        # Step 3: Classify meeting type based on first chunk
        progress(0, desc="Classifying meeting type...")
        classification = process_with_ollama(
            classification_prompt.format(chunk=chunks[0]),
            num_predict=20,
            num_ctx=1000,
            progress=progress
        )
        meeting_type = classification.strip().lower()
        
        # Step 4: Extract information from each chunk based on meeting type
        extracted_chunks = []
        for i, chunk in enumerate(chunks):
            progress((i / len(chunks)), desc=f"Processing chunk {i + 1} of {len(chunks)}...")
            if meeting_type == "technical":
                extracted_chunk = process_with_ollama(
                    technical_extract_prompt.format(chunk=chunk),
                    num_predict=4096,
                    num_ctx=8192,
                    progress=progress
                )
            else:
                extracted_chunk = process_with_ollama(
                    non_technical_extract_prompt.format(chunk=chunk),
                    num_predict=2048,
                    num_ctx=4096,
                    progress=progress
                )
            extracted_chunks.append(extracted_chunk)
            print(extracted_chunk)
        
        # Step 5: Combine extracted chunks
        combined_chunks = combine_chunks(extracted_chunks)
        
        # Step 6: Generate final report based on meeting type
        progress(0, desc="Generating final report...")
        if meeting_type == "technical":
            report_md = process_with_ollama(
                technical_report_prompt.format(combined_chunks=combined_chunks),
                num_predict=8192,
                num_ctx=8192,
                progress=progress
            )
        else:
            report_md = process_with_ollama(
                non_technical_report_prompt.format(combined_chunks=combined_chunks),
                num_predict=4096,
                num_ctx=4096,
                progress=progress
            )
        
        # Step 7: Save Markdown report
        progress(0, desc="Saving Markdown report...")
        md_path = save_markdown_report(report_md, progress)
        
        # Step 8: Convert to PDF (if mdpdf is available)
        progress(0, desc="Converting to PDF...")
        pdf_path = convert_md_to_pdf(md_path, progress)
        
        return report_md, md_path, pdf_path if pdf_path else None
    
    except Exception as e:
        return f"Error: {str(e)}", None, None

# Gradio Interface
with gr.Blocks(title="Intelligent Assistant for Solution Design") as app:
    gr.Markdown("# Intelligent Assistant for Solution Design")
    gr.Markdown("Upload a video (mp4, webm), audio (mp3), or transcript (txt) file to generate a detailed report.")
    gr.Markdown("*Note: Processing may take a few moments depending on file size. Please wait after clicking 'Generate Report'.*")
    if shutil.which("mdpdf") is None:
        gr.Markdown("*Warning: mdpdf is not installed. PDF generation will be skipped. Install mdpdf for PDF reports.*")
    
    # Input components
    file_input = gr.File(label="Upload Video, Audio, or Transcript", file_types=[".mp4", ".webm", ".mp3", ".txt"])
    generate_btn = gr.Button("Generate Report", variant="primary")
    
    # Output components
    report_output = gr.Markdown(label="Generated Report")
    md_output = gr.File(label="Download Report as Markdown (.md)")
    pdf_output = gr.File(label="Download Report as PDF (.pdf)")
    
    # Connect button click to generate_report function
    generate_btn.click(
        fn=generate_report,
        inputs=file_input,
        outputs=[report_output, md_output, pdf_output]
    )

# Launch the application
app.launch(share=True)