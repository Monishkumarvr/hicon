"""
API Client for HiCon - HMAC authenticated cloud sync
"""
import hmac
import hashlib
import json
import time
import logging
import requests
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class APIClient:
    """Handle API communication with HMAC authentication"""
    
    def __init__(self, base_url: str, secret: str, customer_id: str):
        """
        Initialize API client.
        
        Args:
            base_url: API base URL (e.g., "http://api.example.com/api/v1")
            secret: HMAC secret key
            customer_id: Customer ID (e.g., "451")
        """
        self.base_url = base_url
        self.secret = secret
        self.customer_id = customer_id
        self.session = requests.Session()
        self.retry_attempts = 3
        self.timeout = 30
        
        logger.info(f"API Client initialized - URL: {base_url}, Customer: {customer_id}")
    
    def generate_sync_id(self, prefix: str) -> str:
        """Generate unique sync ID"""
        import random
        return f"{prefix}-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
    
    def generate_hmac_signature(self, body: str) -> str:
        """Generate HMAC-SHA256 signature"""
        return hmac.new(
            self.secret.encode('utf-8'),
            body.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def send_melting_data(self, items: List[Dict]) -> Dict:
        """
        Send melting events (tapping, deslagging, pyrometer, spectro) to API.

        Args:
            items: List of melting event records

        Returns:
            API response dictionary
        """
        payload = {"items": items}
        return self._post_with_hmac("/melting", payload)
    
    def send_pouring_data(self, items: List[Dict]) -> Dict:
        """
        Send pouring detection data to API.

        Args:
            items: List of pouring event records with mould-wise timing data

        Returns:
            API response dictionary
        """
        payload = {"items": items}
        return self._post_with_hmac("/pouring", payload)
    
    def _post_with_hmac(self, endpoint: str, payload: Dict) -> Dict:
        """
        POST request with HMAC authentication.
        
        Args:
            endpoint: API endpoint (e.g., "/safety")
            payload: Request payload
        
        Returns:
            API response dictionary
        
        Raises:
            requests.exceptions.RequestException: If request fails after retries
        """
        body = json.dumps(payload)
        signature = self.generate_hmac_signature(body)
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            "X-HMAC-Signature": signature,
            "Content-Type": "application/json"
        }
        
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.post(
                    url,
                    data=body,
                    headers=headers,
                    timeout=self.timeout
                )

                # Log response for debugging
                if response.status_code >= 400:
                    logger.error(f"API Error {response.status_code}: {response.text}")

                response.raise_for_status()
                result = response.json()

                # Reference implementation response format:
                # {
                #     "total": 2,
                #     "successful": 1,
                #     "failed": 0,
                #     "skipped_duplicates": 1,
                #     "results": [
                #         {"sync_id": "abc123", "success": true, "error": null},
                #         {"sync_id": "def456", "success": false, "error": "Duplicate"}
                #     ]
                # }

                # Log aggregate stats
                logger.info(
                    f"✓ API {endpoint} - "
                    f"Total: {result.get('total', 0)}, "
                    f"Successful: {result.get('successful', 0)}, "
                    f"Failed: {result.get('failed', 0)}, "
                    f"Duplicates: {result.get('skipped_duplicates', 0)}"
                )

                # Log per-item results for debugging
                for item_result in result.get('results', []):
                    sync_id = item_result.get('sync_id', 'unknown')
                    success = item_result.get('success', False)
                    error = item_result.get('error')

                    if success:
                        logger.debug(f"    ✓ {sync_id}: Success")
                    else:
                        logger.warning(f"    ✗ {sync_id}: {error}")

                return result

            except requests.exceptions.Timeout:
                logger.error(f"  ✗ Request timeout after {self.timeout}s (attempt {attempt+1}/{self.retry_attempts})")

            except requests.exceptions.ConnectionError:
                logger.error(f"  ✗ Connection error - API server may be down (attempt {attempt+1}/{self.retry_attempts})")

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                if status_code == 400:
                    logger.error(f"  ✗ Bad request (400) - Invalid payload format")
                elif status_code == 401:
                    logger.error(f"  ✗ Unauthorized (401) - HMAC signature invalid")
                elif status_code == 500:
                    logger.error(f"  ✗ Server error (500) - Backend issue")
                else:
                    logger.error(f"  ✗ HTTP {status_code}: {e}")

                # Don't retry on 4xx errors (client errors)
                if 400 <= status_code < 500:
                    raise

            except requests.exceptions.RequestException as e:
                logger.error(f"  ✗ Unexpected error (attempt {attempt+1}/{self.retry_attempts}): {e}")
                if attempt == self.retry_attempts - 1:
                    raise

            # Exponential backoff before retry
            if attempt < self.retry_attempts - 1:
                sleep_time = 2 ** attempt
                logger.info(f"  Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
        
        raise requests.exceptions.RequestException(f"Failed after {self.retry_attempts} attempts")
