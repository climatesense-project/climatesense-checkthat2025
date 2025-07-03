import re
from typing import List


def replace_mentions(text: str) -> str:
    """Replace Twitter mentions with @user in a given string.

    Args:
        text (str): The input string containing Twitter mentions.

    Returns:
        str: The string with mentions replaced by @user.
    """
    return re.sub(r"@\w+", "@user", text)


def replace_urls(text: str, urls: List[str]) -> str:
    """Replace URLs in a string with the url in the urls list iteratively if the list of URLs is not empty.

    Args:
        text (str): The input string containing URLs.
        urls (List[str]): A list of URLs to replace in the text.

    Returns:
        str: The string with URLs replaced by the urls in the list.
    """
    if urls:
        for url in urls:
            # Use regex to match any URL in the text and replace it with '@url'
            text = re.sub(r"https?://\S+|www\.\S+", url, text)
    return text


def snake_to_titlecase(snake_str: str) -> str:
    """Convert a snake_case string to Title Case.

    This function takes a string in snake_case format (words separated by underscores)
    and converts it to Title Case, where each word starts with an uppercase letter.

    Args:
        snake_str (str): The input string in snake_case format.

    Returns:
        str: The converted string in Title Case format.

    Example:
        >>> snake_to_titlecase("scientific_claim")
        'Scientific Claim'
    """
    return " ".join(word.capitalize() for word in snake_str.split("_"))
