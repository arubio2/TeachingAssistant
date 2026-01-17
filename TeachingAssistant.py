import os
import re
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import PyPDF2
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz, process
import json

# Configuration
INPUT_PDF = "exams.pdf"
STUDENT_LIST_FILE = "students.txt"  # One name per line: "FirstName LastName"
OUTPUT_DIR = "split_exams"
TEMP_DIR = "temp_images"
EXAM_START_PATTERN = "Universidad de Navarra"
USE_GPU = True
MATCHING_THRESHOLD = 60  # Minimum fuzzy match score (0-100)

# Initialize device
device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load TrOCR model for handwritten text
print("Loading TrOCR model...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
model.to(device)
print("Model loaded!")

# Global variables for student list
STUDENT_LIST = []
USED_STUDENTS = set()


def setup_directories():
    """Create necessary directories"""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(TEMP_DIR).mkdir(exist_ok=True)


def load_student_list(filepath):
    """Load student list from file"""
    global STUDENT_LIST
    
    if not os.path.exists(filepath):
        print(f"Warning: Student list file '{filepath}' not found!")
        print("Create a file with one student name per line: 'FirstName LastName'")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        students = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(students)} students from list")
    STUDENT_LIST = students
    return students


def normalize_name(name):
    """Normalize name for comparison (remove accents, lowercase, etc.)"""
    import unicodedata
    # Remove accents
    name = ''.join(c for c in unicodedata.normalize('NFD', name)
                   if unicodedata.category(c) != 'Mn')
    # Lowercase and remove extra spaces
    name = ' '.join(name.lower().split())
    return name


def find_best_matching_student(ocr_text, student_list, used_students):
    """
    Find the best matching student from the list using fuzzy matching.
    Returns: (matched_name, confidence_score)
    """
    if not student_list:
        return None, 0
    
    # Normalize OCR text
    ocr_normalized = normalize_name(ocr_text)
    
    # Create list of available students (not yet used)
    available_students = [s for s in student_list if s not in used_students]
    
    if not available_students:
        print("  Warning: All students from list have been assigned!")
        available_students = student_list
    
    # Normalize student names
    normalized_students = {normalize_name(s): s for s in available_students}
    
    # Try different matching strategies
    best_match = None
    best_score = 0
    
    # Strategy 1: Direct fuzzy matching on full names
    for norm_name, orig_name in normalized_students.items():
        score = fuzz.ratio(ocr_normalized, norm_name)
        if score > best_score:
            best_score = score
            best_match = orig_name
    
    # Strategy 2: Token set ratio (handles word order differences)
    for norm_name, orig_name in normalized_students.items():
        score = fuzz.token_set_ratio(ocr_normalized, norm_name)
        if score > best_score:
            best_score = score
            best_match = orig_name
    
    # Strategy 3: Partial matching (if OCR text contains the name)
    for norm_name, orig_name in normalized_students.items():
        score = fuzz.partial_ratio(ocr_normalized, norm_name)
        if score > best_score:
            best_score = score
            best_match = orig_name
    
    # Strategy 4: Check if any student name is contained in OCR text
    for norm_name, orig_name in normalized_students.items():
        if norm_name in ocr_normalized or ocr_normalized in norm_name:
            score = max(best_score, 85)  # High confidence for substring match
            best_score = score
            best_match = orig_name
            break
    
    return best_match, best_score


def extract_text_from_image_region(image, region="top"):
    """
    Extract text from specific regions of the image.
    Splits image into horizontal strips and processes each.
    """
    width, height = image.size
    texts = []
    
    # Define regions to scan (top portion of the page)
    if region == "top":
        # Split top 40% into 4 horizontal strips
        num_strips = 4
        strip_height = int(height * 0.4) // num_strips
        
        for i in range(num_strips):
            y1 = i * strip_height
            y2 = (i + 1) * strip_height
            strip = image.crop((0, y1, width, y2))
            
            # Process each strip
            pixel_values = processor(strip, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            
            generated_ids = model.generate(pixel_values, max_length=64)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if text.strip():
                texts.append(text.strip())
    
    return " ".join(texts)


def extract_text_from_page_for_pattern(pdf_path, page_num):
    """Extract text from page to detect pattern (Universidad de Navarra)"""
    images = convert_from_path(
        pdf_path, 
        first_page=page_num + 1,
        last_page=page_num + 1,
        dpi=200
    )
    
    if not images:
        return ""
    
    text = extract_text_from_image_region(images[0], region="top")
    return text


def detect_exam_boundaries(pdf_path, pattern):
    """Detect which pages start a new exam based on pattern"""
    print(f"Analyzing PDF to detect exam boundaries...")
    print(f"Looking for pattern: '{pattern}'")
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        total_pages = len(reader.pages)
        print(f"Total pages: {total_pages}")
        
        exam_start_pages = []
        
        for page_num in range(total_pages):
            if page_num % 10 == 0:
                print(f"Scanning page {page_num + 1}/{total_pages}...")
            
            # Extract text from page
            text = extract_text_from_page_for_pattern(pdf_path, page_num)
            
            # Check if pattern exists in the text (flexible matching)
            pattern_normalized = pattern.lower().replace(" ", "")
            text_normalized = text.lower().replace(" ", "")
            
            if pattern_normalized in text_normalized:
                exam_start_pages.append(page_num)
                print(f"  ✓ Exam start found on page {page_num + 1}")
        
        print(f"\nFound {len(exam_start_pages)} exams")
        return exam_start_pages, total_pages


def split_pdf_by_boundaries(pdf_path, start_pages, total_pages):
    """Split PDF based on detected exam boundaries"""
    print(f"\nSplitting PDF into {len(start_pages)} exam files...")
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        exam_files = []
        
        for i, start_page in enumerate(start_pages):
            # Determine end page
            if i < len(start_pages) - 1:
                end_page = start_pages[i + 1] - 1
            else:
                end_page = total_pages - 1
            
            # Create PDF for this exam
            writer = PyPDF2.PdfWriter()
            for page_num in range(start_page, end_page + 1):
                writer.add_page(reader.pages[page_num])
            
            # Save with temporary name
            temp_filename = f"{OUTPUT_DIR}/exam_{i + 1:03d}_temp.pdf"
            with open(temp_filename, 'wb') as output_file:
                writer.write(output_file)
            
            exam_files.append({
                'path': temp_filename,
                'exam_num': i + 1,
                'start_page': start_page + 1,
                'end_page': end_page + 1,
                'num_pages': end_page - start_page + 1
            })
            
            print(f"  Exam {i + 1}: pages {start_page + 1}-{end_page + 1} ({end_page - start_page + 1} pages)")
        
        return exam_files


def extract_text_from_first_page_handwriting(pdf_path):
    """Extract text from first page focusing on handwritten content"""
    images = convert_from_path(
        pdf_path, 
        first_page=1,
        last_page=1,
        dpi=300  # Higher DPI for better recognition
    )
    
    if not images:
        return ""
    
    # Extract text from top portion where names typically are
    text = extract_text_from_image_region(images[0], region="top")
    
    return text


def process_single_exam(exam_info, student_list, used_students):
    """Process a single exam"""
    exam_num = exam_info['exam_num']
    print(f"\n[Exam {exam_num}] Processing...")
    
    # Extract text using TrOCR for handwriting
    ocr_text = extract_text_from_first_page_handwriting(exam_info['path'])
    
    print(f"[Exam {exam_num}] OCR text: {ocr_text[:150]}...")
    
    # Match against student list
    matched_name, confidence = find_best_matching_student(ocr_text, student_list, used_students)
    
    if matched_name and confidence >= MATCHING_THRESHOLD:
        print(f"[Exam {exam_num}] ✓ Matched: {matched_name} (confidence: {confidence}%)")
        student_name = matched_name.replace(" ", "_")
        used_students.add(matched_name)
    else:
        print(f"[Exam {exam_num}] ⚠ No good match (best: {confidence}%)")
        student_name = f"Unknown_Student_{exam_num}"
    
    return {
        'exam_info': exam_info,
        'student_name': student_name,
        'ocr_text': ocr_text,
        'matched_original': matched_name,
        'confidence': confidence
    }


def sanitize_filename(name):
    """Ensure filename is valid"""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    if len(name) > 100:
        name = name[:100]
    return name


def rename_exams(results):
    """Rename exam files based on extracted names"""
    print("\n\nRenaming files...")
    
    for result in results:
        exam_info = result['exam_info']
        student_name = sanitize_filename(result['student_name'])
        
        # Create new filename
        new_filename = f"{OUTPUT_DIR}/{student_name}_exam.pdf"
        
        # Handle duplicate names
        counter = 1
        while os.path.exists(new_filename):
            new_filename = f"{OUTPUT_DIR}/{student_name}_{counter}_exam.pdf"
            counter += 1
        
        # Rename file
        os.rename(exam_info['path'], new_filename)
        print(f"  ✓ {os.path.basename(new_filename)}")


def save_matching_report(results, output_file="matching_report.json"):
    """Save detailed report of matching results"""
    report = []
    for result in results:
        report.append({
            'exam_num': result['exam_info']['exam_num'],
            'filename': result['student_name'] + '_exam.pdf',
            'matched_student': result.get('matched_original', 'Unknown'),
            'confidence': result['confidence'],
            'ocr_text': result['ocr_text'][:200]
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nMatching report saved to: {output_file}")


def process_exams_sequential(exam_files, student_list):
    """Process exams sequentially with student list matching"""
    print("\n" + "="*60)
    print("EXTRACTING STUDENT NAMES (Using GPU + Student List)")
    print("="*60)
    
    results = []
    used_students = set()
    
    for exam_info in exam_files:
        try:
            result = process_single_exam(exam_info, student_list, used_students)
            results.append(result)
        except Exception as e:
            print(f"[Exam {exam_info['exam_num']}] Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'exam_info': exam_info,
                'student_name': f"Error_Student_{exam_info['exam_num']}",
                'ocr_text': '',
                'matched_original': None,
                'confidence': 0
            })
    
    return results


def print_summary(results, student_list):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total_exams = len(results)
    high_confidence = sum(1 for r in results if r['confidence'] >= 80)
    medium_confidence = sum(1 for r in results if 60 <= r['confidence'] < 80)
    low_confidence = sum(1 for r in results if r['confidence'] < 60)
    
    print(f"Total exams processed: {total_exams}")
    print(f"High confidence matches (≥80%): {high_confidence}")
    print(f"Medium confidence matches (60-79%): {medium_confidence}")
    print(f"Low confidence / Unknown (< 60%): {low_confidence}")
    
    matched_students = {r['matched_original'] for r in results if r['matched_original']}
    print(f"\nUnique students identified: {len(matched_students)}")
    print(f"Students in list: {len(student_list)}")
    print(f"Students not present: {len(student_list) - len(matched_students)}")
    
    if low_confidence > 0:
        print(f"\n⚠ Warning: {low_confidence} exams need manual review")


def main():
    """Main execution function"""
    print("="*60)
    print("PDF EXAM SPLITTER WITH STUDENT LIST MATCHING")
    print("="*60)
    print(f"Device: {device}")
    print()
    
    # Check if input file exists
    if not os.path.exists(INPUT_PDF):
        print(f"Error: {INPUT_PDF} not found!")
        return
    
    # Setup
    setup_directories()
    
    # Load student list
    print("="*60)
    print("LOADING STUDENT LIST")
    print("-"*60)
    student_list = load_student_list(STUDENT_LIST_FILE)
    print()
    
    # Step 1: Detect exam boundaries
    print("="*60)
    print("STEP 1: DETECTING EXAM BOUNDARIES")
    print("-"*60)
    start_pages, total_pages = detect_exam_boundaries(INPUT_PDF, EXAM_START_PATTERN)
    
    if not start_pages:
        print(f"\nError: No exams found with pattern '{EXAM_START_PATTERN}'")
        return
    
    # Step 2: Split PDF
    print("\n" + "="*60)
    print("STEP 2: SPLITTING PDF")
    print("-"*60)
    exam_files = split_pdf_by_boundaries(INPUT_PDF, start_pages, total_pages)
    
    # Step 3: Extract names and match with student list
    results = process_exams_sequential(exam_files, student_list)
    
    # Step 4: Rename files
    rename_exams(results)
    
    # Step 5: Save report and summary
    save_matching_report(results)
    print_summary(results, student_list)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}/")
    print()


if __name__ == "__main__":
    main()