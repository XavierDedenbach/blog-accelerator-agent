"""
File operations utilities for handling uploads, markdown parsing, and asset management.
"""

import os
import re
import zipfile
import shutil
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
import logging


class FileOpsError(Exception):
    """Exception raised for file operation errors."""
    pass


def extract_zip(zip_path: str, extract_to: str) -> str:
    """
    Extract a ZIP file containing blog assets.
    
    Args:
        zip_path: Path to the ZIP file
        extract_to: Directory to extract to
        
    Returns:
        Path to the extracted folder
        
    Raises:
        FileOpsError: If extraction fails
    """
    try:
        # Create extraction directory if it doesn't exist
        os.makedirs(extract_to, exist_ok=True)
        
        # Generate a unique folder name based on the zip filename
        zip_name = os.path.basename(zip_path)
        folder_name = os.path.splitext(zip_name)[0]
        target_dir = os.path.join(extract_to, folder_name)
        
        # Create a fresh directory
        if os.path.exists(target_dir):
            # Add timestamp to make unique
            import time
            timestamp = int(time.time())
            target_dir = f"{target_dir}_{timestamp}"
        os.makedirs(target_dir, exist_ok=True)
        
        # Extract the zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        return target_dir
    except zipfile.BadZipFile:
        raise FileOpsError(f"Invalid ZIP file: {zip_path}")
    except (OSError, IOError) as e:
        raise FileOpsError(f"Error extracting ZIP: {e}")


def find_markdown_file(directory: str) -> str:
    """
    Find a markdown file in a directory.
    
    Args:
        directory: Directory to search in
        
    Returns:
        Path to the markdown file
        
    Raises:
        FileOpsError: If no markdown file is found or multiple files are found
    """
    markdown_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.md'):
                markdown_files.append(os.path.join(root, file))
    
    if not markdown_files:
        raise FileOpsError(f"No markdown file found in {directory}")
    if len(markdown_files) > 1:
        # Find the one that looks most like a blog post (not a README, etc.)
        non_readme = [f for f in markdown_files if 'readme' not in f.lower()]
        if len(non_readme) == 1:
            return non_readme[0]
        raise FileOpsError(f"Multiple markdown files found in {directory}. Please upload a folder with a single .md file.")
    
    return markdown_files[0]


def read_markdown_file(file_path: str) -> str:
    """
    Read the content of a markdown file.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        Content of the markdown file
        
    Raises:
        FileOpsError: If the file cannot be read
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except (OSError, IOError) as e:
        raise FileOpsError(f"Error reading markdown file: {e}")


def parse_image_references(markdown_content: str) -> List[str]:
    """
    Extract image references from markdown content.
    
    Args:
        markdown_content: Markdown content string
        
    Returns:
        List of image references
        
    Example:
        >>> parse_image_references("![Alt text](images/image.png)")
        ['images/image.png']
    """
    # Match Markdown image syntax: ![alt text](image_path)
    image_references = re.findall(r'!\[.*?\]\((.*?)\)', markdown_content)
    
    # Also match HTML img tags: <img src="image_path" />
    html_images = re.findall(r'<img[^>]*src=["\'](.*?)["\'][^>]*>', markdown_content)
    
    # Combine results and remove any URLs (we only want local files)
    all_images = image_references + html_images
    local_images = [img for img in all_images 
                    if not img.startswith(('http://', 'https://'))]
    
    return local_images


def detect_version_from_filename(filename: str) -> Tuple[str, int]:
    """
    Detect blog title and version from filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        Tuple of (blog_title, version)
        
    Example:
        >>> detect_version_from_filename("why-microgrids_v3.md")
        ('why-microgrids', 3)
    """
    # Remove path and extension
    base_name = os.path.basename(filename)
    name_only = os.path.splitext(base_name)[0]
    
    # Check for explicit version suffix (e.g., _v3, -v2)
    version_match = re.search(r'[_-]v(\d+)$', name_only)
    if version_match:
        version = int(version_match.group(1))
        blog_title = re.sub(r'[_-]v\d+$', '', name_only)
        return blog_title, version
    
    # No explicit version, assume version 1
    return name_only, 1


def load_image_as_base64(image_path: str) -> str:
    """
    Load an image file and convert it to base64 encoding.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded image string
        
    Raises:
        FileOpsError: If the image cannot be read
    """
    try:
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except (OSError, IOError) as e:
        raise FileOpsError(f"Error reading image file: {e}")


def get_image_format(image_path: str) -> str:
    """
    Get the format/extension of an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image format (e.g., 'png', 'jpg')
    """
    return os.path.splitext(image_path)[1].lstrip('.').lower()


def resolve_image_path(base_dir: str, image_reference: str) -> str:
    """
    Resolve a potentially relative image path to an absolute path.
    
    Args:
        base_dir: Base directory (location of the markdown file)
        image_reference: Image reference from markdown
        
    Returns:
        Absolute path to the image
        
    Raises:
        FileOpsError: If the image cannot be found
    """
    # Check if already absolute
    if os.path.isabs(image_reference):
        if os.path.exists(image_reference):
            return image_reference
        raise FileOpsError(f"Image not found: {image_reference}")
    
    # Try relative to the markdown file
    abs_path = os.path.join(base_dir, image_reference)
    if os.path.exists(abs_path):
        return abs_path
    
    # Try looking in common subfolders
    for subfolder in ['images', 'assets', 'img', 'media']:
        test_path = os.path.join(base_dir, subfolder, os.path.basename(image_reference))
        if os.path.exists(test_path):
            return test_path
    
    # Try a deeper search if still not found
    for root, _, files in os.walk(base_dir):
        if os.path.basename(image_reference) in files:
            return os.path.join(root, os.path.basename(image_reference))
    
    raise FileOpsError(f"Image not found: {image_reference}")


def collect_blog_assets(
    markdown_path: str
) -> Dict[str, Any]:
    """
    Collect all assets (markdown, images) from a blog post.
    
    Args:
        markdown_path: Path to the markdown file
        
    Returns:
        Dict containing:
            - 'content': The markdown content
            - 'images': Dict of image references and their base64 encoding
            - 'blog_title': Extracted blog title
            - 'version': Detected version
            - 'asset_folder': Path to the asset folder
    """
    try:
        # Get directory containing the markdown file
        base_dir = os.path.dirname(markdown_path)
        
        # Get blog title and version from filename
        blog_title, version = detect_version_from_filename(markdown_path)
        
        # Read markdown content
        content = read_markdown_file(markdown_path)
        
        # Find image references
        image_refs = parse_image_references(content)
        
        # Load all images as base64
        images = {}
        for img_ref in image_refs:
            try:
                # Resolve the image path (could be relative)
                img_path = resolve_image_path(base_dir, img_ref)
                
                # Get the format and load as base64
                img_format = get_image_format(img_path)
                img_base64 = load_image_as_base64(img_path)
                
                # Store in the images dictionary
                images[img_ref] = {
                    'format': img_format,
                    'base64': img_base64,
                    'path': img_path
                }
            except FileOpsError as e:
                # Log the error but continue with other images
                logging.warning(f"Error loading image {img_ref}: {e}")
        
        return {
            'content': content,
            'images': images,
            'blog_title': blog_title,
            'version': version,
            'asset_folder': base_dir
        }
    except Exception as e:
        raise FileOpsError(f"Error collecting blog assets: {e}")


def process_blog_upload(zip_path: str, extract_dir: str = "data/uploads") -> Dict[str, Any]:
    """
    Process a blog upload ZIP file, extracting contents and collecting assets.
    
    Args:
        zip_path: Path to the ZIP file
        extract_dir: Directory to extract to
        
    Returns:
        Dict containing blog information and assets
    """
    try:
        # Extract ZIP file
        folder_path = extract_zip(zip_path, extract_dir)
        
        # Find the markdown file
        markdown_path = find_markdown_file(folder_path)
        
        # Collect blog assets
        return collect_blog_assets(markdown_path)
    except Exception as e:
        raise FileOpsError(f"Error processing blog upload: {e}") 