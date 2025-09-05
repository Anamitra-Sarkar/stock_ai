import asyncio
import time
import aiohttp

class DataAgent:
    """
    Data Agent: Upgraded for high performance.
    - Uses asynchronous requests for speed.
    - Implements caching to respect API limits and improve response time.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self._cache = {}
        self.CACHE_DURATION = 300  # Cache data for 5 minutes (300 seconds)

    def _is_cache_valid(self, ticker):
        """Checks if the cache for a ticker is still valid."""
        if ticker not in self._cache:
            return False
        return (time.time() - self._cache[ticker]['timestamp']) < self.CACHE_DURATION

    async def _fetch_json(self, session, params):
        """Helper function to perform an async GET request."""
        try:
            async with session.get(self.base_url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"AIOHTTP Error fetching data: {e}")
            return None

    async def get_current_stock_data_async(self, session, ticker):
        """
        Asynchronously fetches the latest price data for a given stock ticker.
        Checks cache first.
        """
        if self._is_cache_valid(ticker) and 'price_data' in self._cache[ticker]:
            return self._cache[ticker]['price_data']

        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker,
            "apikey": self.api_key
        }
        data = await self._fetch_json(session, params)

        if not data or 'Global Quote' not in data or not data['Global Quote']:
            print(f"Warning: No stock data received for {ticker}.")
            return None

        quote = data['Global Quote']
        price_data = {
            'name': ticker,
            'price': float(quote.get('05. price', 0)),
        }

        # Update cache
        if ticker not in self._cache: self._cache[ticker] = {}
        self._cache[ticker]['price_data'] = price_data
        self._cache[ticker]['timestamp'] = time.time()

        return price_data

    async def get_latest_news_async(self, session, ticker):
        """
        Asynchronously fetches the latest news for a given stock ticker.
        Checks cache first.
        """
        if self._is_cache_valid(ticker) and 'news_data' in self._cache[ticker]:
            return self._cache[ticker]['news_data']

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "limit": "1",
            "apikey": self.api_key
        }
        data = await self._fetch_json(session, params)

        if not data or 'feed' not in data or not data['feed']:
            news_data = "No recent news available."
        else:
            news_data = data['feed'][0].get('title', "No title available.")

        # Update cache
        if ticker not in self._cache: self._cache[ticker] = {}
        self._cache[ticker]['news_data'] = news_data
        self._cache[ticker]['timestamp'] = time.time()

        return news_data