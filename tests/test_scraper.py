import pytest
import requests

def test_website_accessibility():
    """Check if the target website is still up and not blocking us."""
    url = "https://syarah.com/en/filters"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    
    assert response.status_code == 200
    assert "filters" in response.url # التأكد أنه لم يقم بعمل Redirect لصفحة حظر