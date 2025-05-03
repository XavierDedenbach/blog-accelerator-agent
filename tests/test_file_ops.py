"""
Tests for file operations utilities in file_ops.py.
"""

import os
import re
import base64
import tempfile
import zipfile
import pytest
from unittest.mock import patch, MagicMock

from agents.utilities.file_ops import (
    extract_zip, find_markdown_file, read_markdown_file, parse_image_references,
    detect_version_from_filename, load_image_as_base64, get_image_format,
    resolve_image_path, collect_blog_assets, process_blog_upload,
    FileOpsError
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_zip_file(temp_dir):
    """Create a temporary ZIP file with sample blog content for testing."""
    # Create content files
    md_content = """# Test Blog

This is a test blog post with an image:

![Test Image](images/test.png)

And another image using HTML:

<img src="images/another.jpg" alt="Another Test Image" />
"""
    
    # Create directory structure
    os.makedirs(os.path.join(temp_dir, "content", "images"), exist_ok=True)
    
    # Create markdown file
    md_path = os.path.join(temp_dir, "content", "test-blog_v2.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    # Create image files with minimal content
    img1_path = os.path.join(temp_dir, "content", "images", "test.png")
    with open(img1_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDAT\x08\xd7c````\x00\x00\x00\x05\x00\x01\xa5\xf6E\\\x00\x00\x00\x00IEND\xaeB`\x82")
    
    img2_path = os.path.join(temp_dir, "content", "images", "another.jpg")
    with open(img2_path, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x03\x02\x02\x03\x02\x02\x03\x03\x03\x03\x04\x03\x03\x04\x05\x08\x05\x05\x04\x04\x05\n\x07\x07\x06\x08\x0c\n\x0c\x0c\x0b\n\x0b\x0b\r\x0e\x12\x10\r\x0e\x11\x0e\x0b\x0b\x10\x16\x10\x11\x13\x14\x15\x15\x15\x0c\x0f\x17\x18\x16\x14\x18\x12\x14\x15\x14\xff\xdb\x00C\x01\x03\x04\x04\x05\x04\x05\t\x05\x05\t\x14\r\x0b\r\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\x14\xff\xc2\x00\x11\x08\x00\x01\x00\x01\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\t\xff\xc4\x00\x14\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06\xff\xda\x00\x0c\x03\x01\x00\x02\x10\x03\x10\x00\x00\x01\t\xff\xda\x00\x08\x01\x01\x00\x01\x05\x02\xbf\xff\xda\x00\x08\x01\x03\x01\x01?\x01\x7f\xff\xda\x00\x08\x01\x02\x01\x01?\x01\x7f\xff\xda\x00\x08\x01\x01\x00\x06?\x02\xff\x00\xff\xda\x00\x08\x01\x01\x00\x01?!\xff\x00\xff\xd9")
    
    # Create ZIP file
    zip_path = os.path.join(temp_dir, "test-blog.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        # Add files to the ZIP
        for root, _, files in os.walk(os.path.join(temp_dir, "content")):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.join(temp_dir, "content"))
                zf.write(file_path, arcname)
    
    return zip_path


def test_extract_zip(temp_zip_file, temp_dir):
    """Test extracting a ZIP file."""
    extract_dir = os.path.join(temp_dir, "extracted")
    result = extract_zip(temp_zip_file, extract_dir)
    
    assert os.path.exists(result)
    assert os.path.exists(os.path.join(result, "test-blog_v2.md"))
    assert os.path.exists(os.path.join(result, "images", "test.png"))
    assert os.path.exists(os.path.join(result, "images", "another.jpg"))


def test_extract_zip_invalid_file(temp_dir):
    """Test extracting an invalid ZIP file."""
    # Create an invalid ZIP file
    invalid_zip = os.path.join(temp_dir, "invalid.zip")
    with open(invalid_zip, "wb") as f:
        f.write(b"This is not a ZIP file")
    
    with pytest.raises(FileOpsError) as excinfo:
        extract_zip(invalid_zip, temp_dir)
    assert "Invalid ZIP file" in str(excinfo.value)


def test_find_markdown_file(temp_dir):
    """Test finding a markdown file in a directory."""
    # Create a markdown file
    os.makedirs(os.path.join(temp_dir, "content"), exist_ok=True)
    md_path = os.path.join(temp_dir, "content", "test.md")
    with open(md_path, "w") as f:
        f.write("# Test")
    
    result = find_markdown_file(os.path.join(temp_dir, "content"))
    assert result == md_path


def test_find_markdown_file_multiple(temp_dir):
    """Test finding a markdown file when multiple are present."""
    # Create markdown files
    os.makedirs(os.path.join(temp_dir, "content"), exist_ok=True)
    md_path1 = os.path.join(temp_dir, "content", "test.md")
    md_path2 = os.path.join(temp_dir, "content", "README.md")
    
    with open(md_path1, "w") as f:
        f.write("# Test")
    with open(md_path2, "w") as f:
        f.write("# README")
    
    # Should find test.md, not README.md
    result = find_markdown_file(os.path.join(temp_dir, "content"))
    assert result == md_path1


def test_find_markdown_file_not_found(temp_dir):
    """Test finding a markdown file when none is present."""
    os.makedirs(os.path.join(temp_dir, "content"), exist_ok=True)
    
    with pytest.raises(FileOpsError) as excinfo:
        find_markdown_file(os.path.join(temp_dir, "content"))
    assert "No markdown file found" in str(excinfo.value)


def test_read_markdown_file(temp_dir):
    """Test reading a markdown file."""
    # Create a markdown file
    md_path = os.path.join(temp_dir, "test.md")
    content = "# Test\n\nThis is a test."
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    result = read_markdown_file(md_path)
    assert result == content


def test_parse_image_references():
    """Test parsing image references from markdown content."""
    content = """# Test

![Image 1](images/image1.png)

Some text.

![Image 2](../assets/image2.jpg)

<img src="images/image3.gif" alt="Image 3" />

Also works with remote images, but these should be filtered out:
![Remote Image](https://example.com/image.png)
<img src="http://example.com/image.jpg" />
"""
    
    result = parse_image_references(content)
    assert len(result) == 3
    assert "images/image1.png" in result
    assert "../assets/image2.jpg" in result
    assert "images/image3.gif" in result
    assert "https://example.com/image.png" not in result
    assert "http://example.com/image.jpg" not in result


def test_detect_version_from_filename():
    """Test detecting version from filename."""
    # Test with version suffix
    assert detect_version_from_filename("test-blog_v3.md") == ("test-blog", 3)
    assert detect_version_from_filename("test-blog-v2.md") == ("test-blog", 2)
    
    # Test without version suffix
    assert detect_version_from_filename("test-blog.md") == ("test-blog", 1)
    
    # Test with path
    assert detect_version_from_filename("/path/to/test-blog_v4.md") == ("test-blog", 4)


def test_load_image_as_base64(temp_dir):
    """Test loading an image as base64."""
    # Create a simple image file
    img_path = os.path.join(temp_dir, "test.png")
    with open(img_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDAT\x08\xd7c````\x00\x00\x00\x05\x00\x01\xa5\xf6E\\\x00\x00\x00\x00IEND\xaeB`\x82")
    
    result = load_image_as_base64(img_path)
    # Decode the base64 to verify it's valid
    decoded = base64.b64decode(result)
    # Check that it starts with the PNG signature
    assert decoded.startswith(b"\x89PNG\r\n\x1a\n")


def test_get_image_format():
    """Test getting image format from path."""
    assert get_image_format("test.png") == "png"
    assert get_image_format("test.jpg") == "jpg"
    assert get_image_format("test.svg") == "svg"
    assert get_image_format("/path/to/test.jpeg") == "jpeg"
    assert get_image_format("test.PNG") == "png"  # Case-insensitive


def test_resolve_image_path(temp_dir):
    """Test resolving an image path."""
    # Create directory structure
    os.makedirs(os.path.join(temp_dir, "images"), exist_ok=True)
    
    # Create test files
    img_path = os.path.join(temp_dir, "images", "test.png")
    with open(img_path, "wb") as f:
        f.write(b"test")
    
    # Test relative path
    assert resolve_image_path(temp_dir, "images/test.png") == img_path
    
    # Test with just filename, should find in images folder
    assert resolve_image_path(temp_dir, "test.png") == img_path


def test_resolve_image_path_not_found(temp_dir):
    """Test resolving an image path that doesn't exist."""
    with pytest.raises(FileOpsError) as excinfo:
        resolve_image_path(temp_dir, "nonexistent.png")
    assert "Image not found" in str(excinfo.value)


def test_collect_blog_assets(temp_dir):
    """Test collecting blog assets."""
    # Create directory structure
    os.makedirs(os.path.join(temp_dir, "images"), exist_ok=True)
    
    # Create markdown file
    md_content = """# Test Blog

![Test Image](images/test.png)
"""
    md_path = os.path.join(temp_dir, "test-blog_v2.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    # Create image file
    img_path = os.path.join(temp_dir, "images", "test.png")
    with open(img_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDAT\x08\xd7c````\x00\x00\x00\x05\x00\x01\xa5\xf6E\\\x00\x00\x00\x00IEND\xaeB`\x82")
    
    # Collect assets
    result = collect_blog_assets(md_path)
    
    # Check results
    assert result["blog_title"] == "test-blog"
    assert result["version"] == 2
    assert result["content"] == md_content
    assert result["asset_folder"] == temp_dir
    assert "images/test.png" in result["images"]
    assert result["images"]["images/test.png"]["format"] == "png"
    assert isinstance(result["images"]["images/test.png"]["base64"], str)


def test_process_blog_upload(temp_zip_file, temp_dir):
    """Test processing a blog upload."""
    with patch('agents.utilities.file_ops.extract_zip') as mock_extract:
        with patch('agents.utilities.file_ops.find_markdown_file') as mock_find:
            with patch('agents.utilities.file_ops.collect_blog_assets') as mock_collect:
                # Setup mocks
                extract_dir = os.path.join(temp_dir, "extracted")
                mock_extract.return_value = extract_dir
                md_path = os.path.join(extract_dir, "test-blog_v2.md")
                mock_find.return_value = md_path
                mock_collect.return_value = {
                    "blog_title": "test-blog",
                    "version": 2,
                    "content": "# Test Blog",
                    "images": {"images/test.png": {"format": "png", "base64": "test"}}
                }
                
                # Call function
                result = process_blog_upload(temp_zip_file)
                
                # Verify mocks were called with expected args
                mock_extract.assert_called_once_with(temp_zip_file, "data/uploads")
                mock_find.assert_called_once_with(extract_dir)
                mock_collect.assert_called_once_with(md_path)
                
                # Verify result
                assert result["blog_title"] == "test-blog"
                assert result["version"] == 2 