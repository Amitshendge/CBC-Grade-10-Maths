# redis_utils.py
import redis
import logging
from typing import Optional
from config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_USERNAME

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RedisUtils:
    _client = None  # class-level shared client

    @classmethod
    def get_client(cls) -> redis.Redis:
        """Return a shared Redis client (singleton)."""
        if cls._client is None:
            try:
                cls._client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    username=REDIS_USERNAME,
                    password=REDIS_PASSWORD,
                    decode_responses=True,
                    socket_timeout=5,
                    health_check_interval=30,
                )
                cls._client.ping()
                logger.info("✅ Connected to Redis at %s:%s", REDIS_HOST, REDIS_PORT)
            except Exception as e:
                logger.error("❌ Redis connection failed: %s", e)
                raise
        return cls._client

    def __init__(self):
        self.redis_server = self.get_client()

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """
        Set a key with optional TTL (time-to-live in seconds).
        """
        try:
            result = self.redis_server.set(key, value, ex=ttl)
            return bool(result)
        except Exception as e:
            logger.error("Error setting key %s: %s", key, e)
            return False

    def get(self, key: str) -> Optional[str]:
        """
        Get the value of a key.
        """
        try:
            return self.redis_server.get(key)
        except Exception as e:
            logger.error("Error getting key %s: %s", key, e)
            return None

    def update(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """
        Update a key (only if it exists).
        """
        try:
            if not self.redis_server.exists(key):
                logger.warning("Key %s not found, cannot update", key)
                return False
            return self.set(key, value, ttl)
        except Exception as e:
            logger.error("Error updating key %s: %s", key, e)
            return False

    def delete(self, key: str) -> bool:
        """
        Delete a key.
        """
        try:
            return self.redis_server.delete(key) > 0
        except Exception as e:
            logger.error("Error deleting key %s: %s", key, e)
            return False
    
    def get_redis_info(self) -> dict:
        """
        Get Redis server information including memory usage.
        """
        try:
            info = self.redis_server.info()
            return info
        except Exception as e:
            logger.error("Error getting Redis info: %s", e)
            return {}


if __name__ == "__main__":
    redis_utils = RedisUtils()

    # Example usage
    redis_utils.set("foo", "bar", ttl=10)  # expires in 10s
    print(redis_utils.get("foo"))
    redis_utils.update("foo", "baz", ttl=20)
    print(redis_utils.get("foo"))
    redis_utils.delete("foo")
    print(redis_utils.get("foo"))  # should be None
