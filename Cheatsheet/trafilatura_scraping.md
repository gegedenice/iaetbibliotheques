#  Scraping with trafilatura

Pieces of code for making webscraping with [trafilatura](https://trafilatura.readthedocs.io/en/latest/), a python package, CLI tool and/or GUI for extracting high quality texts from the web

## installation

```
pip install -U trafilatura
```

## GUI

```
pip install -U trafilatura[gui]
trafilatura_gui
```

## CLI

```
trafilatura -u "https://projet2024.abes.fr/docs/2.4/projet2024"
```

## Python

### Helper for cleaning (too long) urls

```
def clean_url(url):
    """
    The function `get_pagename_from_url` takes a URL as input and returns the last part of the URL after
    the last slash, with any non-alphabetic characters removed, and truncated to a maximum length of 100
    characters.

    :param url: A string representing the URL of a webpage
    :type url: str
    :return: the cleaned pagename extracted from the given URL.
    """
    pagename = url.rsplit("/", 1)[-1]
    cleaned_pagename = re.sub("[^A-Z]", "", pagename, 0, re.IGNORECASE)
    # to avoid too long pagenames
    if len(cleaned_pagename) > 100:
        return cleaned_pagename[:100]
    else:
        return cleaned_pagename
```		

### Get urls from sitemap

#### With trafilatura

```
from trafilatura import sitemaps

def get_urls_from_sitemap(sitemap_url: str) -> str:
    """
    The function `get_urls_from_sitemap` takes a sitemap URL as input and returns a list of URLs found
    in the sitemap.

    :param sitemap_url: The sitemap_url parameter is a string that represents the URL of the sitemap. A
    sitemap is a file that lists all the URLs of a website and helps search engines crawl and index the
    website's pages
    :type sitemap_url: str
    :return: a list of URLs extracted from the provided sitemap URL.
    """
    return sitemaps.sitemap_search(sitemap_url, target_lang="en")
```

#### Without trafilatura

```
from bs4 import BeautifulSoup
import requests

def make_get_call(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        return response.text
    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
    except requests.exceptions.RequestException as err:
        print ("Something went wrong",err)

response = make_get_call('https://barcelona-declaration.org/sitemap.xml')
soup = BeautifulSoup(response, "xml")

links = [link.text for link in soup.find_all("loc") if link.parent.name == "url"]
print(links)

# returns
['https://barcelona-declaration.org/',
 'https://barcelona-declaration.org/about/',
 'https://barcelona-declaration.org/background_and_context/',
 'https://barcelona-declaration.org/commitments/',
 'https://barcelona-declaration.org/definitions/',
 'https://barcelona-declaration.org/infographic/',
 'https://barcelona-declaration.org/media/',
 'https://barcelona-declaration.org/preamble/',
 'https://barcelona-declaration.org/resources/',
 'https://barcelona-declaration.org/signatories/',
 'https://barcelona-declaration.org/signatories_backup/',
 'https://barcelona-declaration.org/temp_icons/',
 'https://barcelona-declaration.org/translations/',
 'https://barcelona-declaration.org/categories/',
 'https://barcelona-declaration.org/tags/']
```

## Scraping

```
from trafilatura import fetch_url, extract

def extract_from_url(url: str) -> str:
    """
    The function `extract_from_url` takes a URL as input, downloads the content from that URL, and then
    extracts specific information from the downloaded content.

    :param url: The `url` parameter is a string that represents the URL from which you want to extract
    data
    :type url: str
    :return: a string.
    """
    downloaded = fetch_url(url)
    if downloaded is not None:
        result = extract(
            downloaded, 
			include_comments=False, # no comments in output
			include_tables=True, #keep tables examination
			include_links=True, # output with links
			output_format='xml', # available formats are : bare text (default)|markdown|csv|json|xml|bare text+markdown...
			no_fallback=True
        )
        return result
```

## Complete from sitemap to local saving texts

```
from trafilatura import sitemaps, fetch_url, extract

def load_urls_to_dirfiles(
    dir: str,
	sitemap_url: str,
) -> None:
    """
    The function `load_sitemap_to_dirfiles` takes a directory path and a sitemap URL, retrieves the URLs
    from the sitemap, downloads the web pages, extracts the content, and saves it as text files in the
    specified directory.

    :param dir: The `dir` parameter is a string that represents the directory where the downloaded files
    will be saved
    :type dir: str
    :param sitemap_url: The `sitemap_url` parameter is a string that represents the URL of a sitemap. A
    sitemap is a file that lists all the URLs of a website, allowing search engines to crawl and index
    the website's pages more efficiently
    :type sitemap_url: str
    """
	urls = sitemaps.sitemap_search(sitemap_url, target_lang="en")
    for url in urls:
        page_name = get_pagename_from_url(url)
        downloaded = fetch_url(url)
        if downloaded is not None:
            result = extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                no_fallback=True,
            )
            with open(f"{dir}/{page_name}.txt", "w", encoding="utf-8") as f:
                f.write(result)
```
