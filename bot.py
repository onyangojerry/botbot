import requests
from bs4 import BeautifulSoup
import os

# Set to avoid scraping the same pages multiple times
visited_pages = set()

# Function to get all links on a page
def get_all_links(url, base_url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        # Get all anchor tags
        links = set([link.get('href') for link in soup.find_all('a', href=True)])
        full_links = set()
        for link in links:
            # Create absolute URL
            if link.startswith('/'):
                full_links.add(base_url + link)
            elif link.startswith('http'):
                full_links.add(link)
        return full_links
    except Exception as e:
        print(f"Error fetching links from {url}: {e}")
        return set()

# Function to scrape content from a single page
def scrape_page(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract and return all text content (e.g., <p>, <h1>, etc.)
        page_content = ''
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li']):
            page_content += tag.get_text(strip=True) + '\n'

        return page_content
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

# Recursive function to scrape all linked pages
def scrape_website(start_url, base_url):
    if start_url in visited_pages:
        return
    visited_pages.add(start_url)
    
    print(f"Scraping {start_url}")
    content = scrape_page(start_url)
    
    # Save page content to a file
    with open("school_website_data.txt", "a", encoding="utf-8") as file:
        file.write(f"URL: {start_url}\n")
        file.write(content + "\n\n")

    # Find all links and recursively scrape them
    links = get_all_links(start_url, base_url)
    for link in links:
        if link not in visited_pages:
            scrape_website(link, base_url)

# Example Usage
start_url = "https://www.pomona.edu"  # Replace with your school website's URL
base_url = "https://www.pomona.edu"  # The base URL for relative links

# Start scraping
scrape_website(start_url, base_url)
