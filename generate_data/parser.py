import os
from typing import List

from markitdown import MarkItDown

def get_valid_extensions() -> List[str]:
    """get valid extensions"""
    return ["docx", "pdf", "txt", "md", "xlsx", "csv", "pptx"]

def convert_to_markdown(file_path: str) -> str:
    """convert to markdown using markitdown"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    md = MarkItDown(enable_plugins=True)  # Set to True to enable plugins
    markdown_content = md.convert(file_path)
    return markdown_content.text_content


def chunk_markdown(markdown_text, chunk_size=1000, overlap_size=100):
    """
    Split a markdown string into chunks of approximately the specified size with overlap.
    Preserves the integrity of markdown elements like tables and code blocks.

    Args:
        markdown_text (str): The markdown text to split into chunks
        chunk_size (int): The target size of each chunk in characters
        overlap_size (int): The number of characters to overlap between chunks

    Returns:
        list: A list of markdown chunks
    """
    # Initialize variables
    chunks = []
    start_pos = 0
    text_length = len(markdown_text)
    if chunk_size == 0:
        return [markdown_text]

    # Edge case: if the text is shorter than chunk_size, return it as a single chunk
    if text_length <= chunk_size:
        return [markdown_text]

    # Split the text into lines to help with processing tables and code blocks
    lines = markdown_text.split("\n")
    line_positions = []

    # Calculate the start and end positions of each line
    current_pos = 0
    for line in lines:
        line_length = len(line) + 1  # +1 for the newline character
        line_positions.append((current_pos, current_pos + line_length))
        current_pos += line_length

    # Helper function to check if a position is within a table
    def is_in_table(position):
        # Find which line contains this position
        for i, (start, end) in enumerate(line_positions):
            if start <= position < end:
                # Check if this line or adjacent lines contain table markers
                line_idx = i

                # Check current line for table markers
                if line_idx < len(lines) and (
                    "|" in lines[line_idx]
                    or lines[line_idx].strip().startswith("|")
                    or lines[line_idx].strip().startswith("+-")
                    or lines[line_idx].strip().startswith("| -")
                ):
                    return True

                # Check previous line
                if line_idx > 0 and (
                    "|" in lines[line_idx - 1]
                    or lines[line_idx - 1].strip().startswith("|")
                    or lines[line_idx - 1].strip().startswith("+-")
                    or lines[line_idx - 1].strip().startswith("| -")
                ):
                    return True

                # Check next line
                if line_idx < len(lines) - 1 and (
                    "|" in lines[line_idx + 1]
                    or lines[line_idx + 1].strip().startswith("|")
                    or lines[line_idx + 1].strip().startswith("+-")
                    or lines[line_idx + 1].strip().startswith("| -")
                ):
                    return True

                break
        return False

    # Helper function to find the end of a table
    def find_table_end(position):
        # Find which line contains this position
        for i, (start, end) in enumerate(line_positions):
            if start <= position < end:
                line_idx = i

                # Find the next line that doesn't look like part of a table
                for j in range(line_idx, len(lines)):
                    if not (
                        "|" in lines[j]
                        or lines[j].strip().startswith("|")
                        or lines[j].strip().startswith("+-")
                        or lines[j].strip().startswith("| -")
                    ):
                        if j < len(line_positions):
                            return line_positions[j][0]

                # If we reach here, the table extends to the end of the text
                return text_length

        return position  # Fallback

    # Helper function to check if a position is within a code block
    def is_in_code_block(position):
        # Find which line contains this position
        for i, (start, end) in enumerate(line_positions):
            if start <= position < end:
                # Check surrounding lines for code block markers
                line_idx = i

                # Look backward for the start of a code block
                code_block_start = False
                for j in range(line_idx, -1, -1):
                    if lines[j].strip().startswith("```"):
                        code_block_start = True
                        break

                # Look forward for the end of the code block
                code_block_end = False
                if code_block_start:
                    for j in range(line_idx + 1, len(lines)):
                        if lines[j].strip().startswith("```"):
                            code_block_end = True
                            break

                # If we found a start but no end (or vice versa), we're in a code block
                return code_block_start and not code_block_end

        return False

    # Helper function to find the end of a code block
    def find_code_block_end(position):
        # Find which line contains this position
        for i, (start, end) in enumerate(line_positions):
            if start <= position < end:
                line_idx = i

                # Find the next code block end marker
                for j in range(line_idx, len(lines)):
                    if lines[j].strip().startswith("```"):
                        if j + 1 < len(line_positions):
                            return line_positions[j + 1][
                                0
                            ]  # Return position after the end marker

                # If we reach here, the code block extends to the end of the text
                return text_length

        return position  # Fallback

    while start_pos < text_length:
        # Calculate end position for this chunk
        end_pos = min(start_pos + chunk_size, text_length)

        # Check if we're in the middle of a table or code block
        if is_in_table(end_pos):
            # Find the end of the table
            table_end = find_table_end(end_pos)
            if table_end > end_pos:
                end_pos = table_end
        elif is_in_code_block(end_pos):
            # Find the end of the code block
            code_block_end = find_code_block_end(end_pos)
            if code_block_end > end_pos:
                end_pos = code_block_end

        # If we're not at the end of the text and not already at a natural break,
        # try to find a good breaking point
        if end_pos < text_length:
            # Look for a paragraph break first (double newline)
            paragraph_break = markdown_text.rfind("\n\n", start_pos, end_pos)

            # Look for a single newline
            newline = markdown_text.rfind("\n", start_pos, end_pos)

            # Look for a sentence end (period followed by space or newline)
            sentence_end = max(
                markdown_text.rfind(". ", start_pos, end_pos),
                markdown_text.rfind(".\n", start_pos, end_pos),
            )

            # Look for other potential break points
            other_breaks = [
                markdown_text.rfind(". ", start_pos, end_pos),
                markdown_text.rfind("? ", start_pos, end_pos),
                markdown_text.rfind("! ", start_pos, end_pos),
                markdown_text.rfind(": ", start_pos, end_pos),
                markdown_text.rfind("; ", start_pos, end_pos),
            ]
            other_break = max(other_breaks)

            # Choose the best break point based on priority
            # Prefer paragraph breaks, then line breaks, then sentence ends, then other punctuation
            if paragraph_break != -1 and paragraph_break > start_pos + chunk_size // 2:
                end_pos = paragraph_break + 2  # Include the double newline
            elif newline != -1 and newline > start_pos + chunk_size // 2:
                end_pos = newline + 1  # Include the newline
            elif sentence_end != -1 and sentence_end > start_pos + chunk_size // 2:
                end_pos = sentence_end + 2  # Include the period and space
            elif other_break != -1 and other_break > start_pos + chunk_size // 2:
                end_pos = other_break + 2  # Include the punctuation and space

        # Extract the chunk
        chunk = markdown_text[start_pos:end_pos]
        chunks.append(chunk)

        # Move the start position for the next chunk, considering overlap
        start_pos = end_pos - overlap_size

        # Make sure we're making progress (in case of very small chunks)
        if start_pos >= text_length:
            break
        if len(chunk) <= overlap_size:
            start_pos = end_pos  # No overlap for very small chunks

    return chunks


def chunk_text(text, chunk_size=1000, overlap_size=100, break_chars=None):
    """
    Split a text string into chunks of approximately the specified size with overlap.
    This is a simpler version that only considers text length, without special handling
    for markdown elements.

    Args:
        text (str): The text to split into chunks
        chunk_size (int): The target size of each chunk in characters
        overlap_size (int): The number of characters to overlap between chunks
        break_chars (list, optional): List of strings to use as break characters,
                                     in order of preference. Defaults to a standard set.

    Returns:
        list: A list of text chunks
    """
    if chunk_size == 0:
        return [text]
    # Use default break characters if none provided
    if break_chars is None:
        break_chars = [
            "。",
            "，",
            "！",
            "？",
            "；",
            "：",
            " ",
            "\n\n",  # Paragraph break
            "\n",  # Line break
            ". ",  # Sentence end
            "? ",  # Question mark
            "! ",  # Exclamation mark
            "; ",  # Semicolon
            ", ",  # Comma
            " ",  # Space (last resort)
        ]

    # Initialize variables
    chunks = []
    start_pos = 0
    text_length = len(text)

    # Edge case: if the text is shorter than chunk_size, return it as a single chunk
    if text_length <= chunk_size:
        return [text]

    while start_pos < text_length:
        # Calculate end position for this chunk
        end_pos = min(start_pos + chunk_size, text_length)

        # If we're not at the end of the text and not at a natural break,
        # try to find a good breaking point based on the provided break_chars
        if end_pos < text_length:
            # Build a list of candidate break points based on the break_chars
            break_candidates = []
            for char in break_chars:
                break_candidates.append(text.rfind(char, start_pos, end_pos))

            # Find the first valid break point
            for i, break_point in enumerate(break_candidates):
                if break_point != -1 and break_point > start_pos + (chunk_size // 4):
                    # Add the length of the break character to include it
                    char_len = len(break_chars[i])
                    end_pos = break_point + char_len
                    break

        # Extract the chunk
        chunk = text[start_pos:end_pos]
        chunks.append(chunk)

        # Move the start position for the next chunk, considering overlap
        start_pos = end_pos - overlap_size

        # Make sure we're making progress
        if start_pos >= text_length:
            break
        if len(chunk) <= overlap_size:
            start_pos = end_pos  # No overlap for very small chunks

    return chunks


def load_chunks(
    file_path: str, chunk_size: int = 1000, overlap_size: int = 200
) -> List[str]:
    """load chunks from file"""
    content = convert_to_markdown(file_path)
    chunks = chunk_text(content, chunk_size, overlap_size)
    return chunks


# Usage example
if __name__ == "__main__":
    # Initialize the chunker with custom chunk size and overlap
    file_path = "test_data/test.docx"
    content = convert_to_markdown(file_path)
    chunks = chunk_text(content, 1000, 200)
    for chunk in chunks:
        print("****")
        print(chunk)
        print("----")
