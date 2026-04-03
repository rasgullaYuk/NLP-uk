import fitz  # PyMuPDF
import os
import shutil

def prepare_document(file_path, output_dir="temp_pages"):
    """
    The 'Receptionist': Checks if the file is a PDF, JPEG, or TIFF.
    Returns a list of image paths ready for OpenCV preprocessing.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the file extension (e.g., '.pdf', '.jpg', '.tiff')
    file_extension = os.path.splitext(file_path)[1].lower()
    image_paths = []

    print(f"\n[Receptionist] Received file: {file_path}")
    print(f"[Receptionist] Detected format: {file_extension}")

    # SCENARIO A: It's a PDF (Needs to be split into images)
    if file_extension == '.pdf':
        print("[Receptionist] This is a PDF. Sending to Page Ripper...")
        try:
            pdf_document = fitz.open(file_path)
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                mat = fitz.Matrix(2.0, 2.0) # High quality 300 DPI
                pix = page.get_pixmap(matrix=mat)
                
                img_path = os.path.join(output_dir, f"page_{page_num + 1}.jpg")
                pix.save(img_path)
                image_paths.append(img_path)
            print(f"[Receptionist] Extracted {len(image_paths)} pages.")
        except Exception as e:
            print(f"Error opening PDF: {e}")

    # SCENARIO B: It's already an Image (JPEG, JPG, PNG, TIFF)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
        print("[Receptionist] This is already an image. Sending directly to preprocessing...")
        # We just copy the image into our working folder so we don't mess up the original
        file_name = os.path.basename(file_path)
        new_path = os.path.join(output_dir, f"raw_{file_name}")
        shutil.copy(file_path, new_path)
        image_paths.append(new_path)

    # SCENARIO C: Unsupported file
    else:
        print(f"[Receptionist] ERROR: I don't know how to handle {file_extension} files!")

    return image_paths