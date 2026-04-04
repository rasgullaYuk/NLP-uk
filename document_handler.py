import fitz  # PyMuPDF
import os
import shutil
import logging

logger = logging.getLogger("tier0.document_handler")

SUPPORTED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif'}


def _safe_doc_name(file_path):
    """
    Convert a filename to a safe prefix for output images.
    e.g. 'Discharge Summary 1.pdf' → 'discharge_summary_1'
    """
    stem = os.path.splitext(os.path.basename(file_path))[0]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return safe.lower()


def prepare_document(file_path, output_dir="temp_pages"):
    """
    The 'Receptionist': Checks if the file is a PDF, JPEG, or TIFF.
    Returns a list of image paths ready for OpenCV preprocessing.

    Naming convention: {doc_name}page{n}_original.jpg
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_extension = os.path.splitext(file_path)[1].lower()
    doc_name = _safe_doc_name(file_path)
    image_paths = []
    failed_pages = []

    print(f"\n[Receptionist] Received file: {file_path}")
    print(f"[Receptionist] Detected format: {file_extension}")

    # SCENARIO A: It's a PDF (Needs to be split into images)
    if file_extension == '.pdf':
        print("[Receptionist] This is a PDF. Sending to Page Ripper...")
        try:
            pdf_document = fitz.open(file_path)
            total_pages = len(pdf_document)
            for page_num in range(total_pages):
                try:
                    page = pdf_document.load_page(page_num)
                    mat = fitz.Matrix(2.0, 2.0)  # ~300 DPI
                    pix = page.get_pixmap(matrix=mat)

                    # Naming convention: {doc_name}page{n}_original.jpg
                    img_path = os.path.join(output_dir, f"{doc_name}page{page_num + 1}_original.jpg")
                    pix.save(img_path)
                    image_paths.append(img_path)
                    print(f"  ✓ Page {page_num + 1}/{total_pages} extracted.")
                except Exception as e:
                    # Log and continue — don't let one bad page kill the batch
                    print(f"  ✗ Page {page_num + 1} failed: {e}")
                    failed_pages.append({"page": page_num + 1, "error": str(e)})

            print(f"[Receptionist] Extracted {len(image_paths)}/{total_pages} pages.")
            if failed_pages:
                print(f"[Receptionist] WARNING: {len(failed_pages)} page(s) failed and were skipped.")
        except Exception as e:
            print(f"[Receptionist] ERROR opening PDF: {e}")

    # SCENARIO B: It's already an Image (JPEG, JPG, PNG, TIFF)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
        print("[Receptionist] This is already an image. Sending directly to preprocessing...")
        try:
            # Copy into working folder with correct naming convention
            new_path = os.path.join(output_dir, f"{doc_name}page1_original.jpg")
            shutil.copy(file_path, new_path)
            image_paths.append(new_path)
            print(f"  ✓ Image copied as: {os.path.basename(new_path)}")
        except Exception as e:
            print(f"  ✗ Failed to copy image: {e}")
            failed_pages.append({"page": 1, "error": str(e)})

    # SCENARIO C: Unsupported file
    else:
        print(f"[Receptionist] ERROR: I don't know how to handle {file_extension} files!")

    return image_paths, failed_pages


def prepare_batch(input_path, output_dir="temp_pages", recursive=True):
    """
    Batch version of prepare_document.
    Accepts either a single file path or a directory.
    Processes all supported documents found.

    Returns:
        all_image_paths : flat list of all extracted image paths
        batch_results   : per-document result dicts for the JSON report
    """
    all_image_paths = []
    batch_results = []

    input_path = str(input_path)

    if os.path.isfile(input_path):
        # Single file mode
        image_paths, failed_pages = prepare_document(input_path, output_dir)
        all_image_paths.extend(image_paths)
        batch_results.append({
            "document": os.path.basename(input_path),
            "doc_name": _safe_doc_name(input_path),
            "total_pages": len(image_paths) + len(failed_pages),
            "extracted": len(image_paths),
            "failed_extraction": failed_pages,
            "image_paths": image_paths,
        })

    elif os.path.isdir(input_path):
        # Directory mode — find all supported files
        found_files = []
        if recursive:
            for root, dirs, files in os.walk(input_path):
                for f in files:
                    if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                        found_files.append(os.path.join(root, f))
        else:
            for f in os.listdir(input_path):
                full = os.path.join(input_path, f)
                if os.path.isfile(full) and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                    found_files.append(full)

        if not found_files:
            print(f"[Receptionist] WARNING: No supported documents found in {input_path}")
            return all_image_paths, batch_results

        print(f"[Receptionist] Found {len(found_files)} document(s) to process.")
        for doc_path in sorted(found_files):
            image_paths, failed_pages = prepare_document(doc_path, output_dir)
            all_image_paths.extend(image_paths)
            batch_results.append({
                "document": os.path.basename(doc_path),
                "doc_name": _safe_doc_name(doc_path),
                "total_pages": len(image_paths) + len(failed_pages),
                "extracted": len(image_paths),
                "failed_extraction": failed_pages,
                "image_paths": image_paths,
            })
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    return all_image_paths, batch_results