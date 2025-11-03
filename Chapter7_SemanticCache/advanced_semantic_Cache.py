
# advanced_semantic_cache.py - Advanced Semantic Caching Implementation

import asyncio
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import hashlib
import json

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a cache entry with metadata"""
    key: str
    prompt: str
    service_id: str
    response: Dict[str, Any]
    embedding: np.ndarray
    timestamp: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    ttl: Optional[timedelta] = None
    metadata: Optional[Dict[str, Any]] = None

class AdvancedSemanticCache:
    """
    Advanced semantic cache with TTL, LRU eviction, and persistence
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_size: int = 1000,
        default_ttl: Optional[timedelta] = None,
        persistence_file: Optional[str] = None
    ):
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.persistence_file = persistence_file

        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # For LRU eviction

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }

        # Load from persistence if available
        if persistence_file:
            self._load_from_persistence()

    def _generate_key(self, prompt: str, service_id: str) -> str:
        """Generate unique key for cache entry"""
        content = f"{service_id}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(emb1, emb2) / (norm1 * norm2)

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        if entry.ttl is None:
            return False
        return datetime.now() - entry.timestamp > entry.ttl

    def _cleanup_expired(self):
        """Remove expired entries"""
        expired_keys = [
            key for key, entry in self.cache.items()
            if self._is_expired(entry)
        ]

        for key in expired_keys:
            self._remove_entry(key)
            self.stats["evictions"] += 1
            logger.debug(f"Removed expired cache entry: {key}")

    def _evict_lru(self):
        """Evict least recently used entries if cache is full"""
        while len(self.cache) >= self.max_size and self.access_order:
            lru_key = self.access_order[0]
            self._remove_entry(lru_key)
            self.stats["evictions"] += 1
            logger.debug(f"Evicted LRU cache entry: {lru_key}")

    def _remove_entry(self, key: str):
        """Remove entry from cache and access order"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)

    def _update_access(self, key: str):
        """Update access statistics and LRU order"""
        if key in self.cache:
            self.cache[key].access_count += 1
            self.cache[key].last_accessed = datetime.now()

            # Update LRU order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

    async def get(
        self,
        prompt: str,
        service_id: str,
        embedding: np.ndarray
    ) -> Optional[Tuple[Dict[str, Any], CacheEntry]]:
        """Get cached response for semantically similar prompt"""
        self.stats["total_requests"] += 1

        # Cleanup expired entries
        self._cleanup_expired()

        # Find best match
        best_match = None
        best_similarity = 0.0
        best_key = None

        for key, entry in self.cache.items():
            if entry.service_id != service_id:
                continue

            similarity = self._calculate_similarity(embedding, entry.embedding)

            if similarity >= self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
                best_key = key

        if best_match:
            self._update_access(best_key)
            self.stats["hits"] += 1

            logger.info(
                f"Cache hit for service '{service_id}' with similarity {best_similarity:.3f}"
            )

            return best_match.response, best_match

        self.stats["misses"] += 1
        return None

    async def set(
        self,
        prompt: str,
        service_id: str,
        response: Dict[str, Any],
        embedding: np.ndarray,
        ttl: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store response in cache"""
        key = self._generate_key(prompt, service_id)

        # Use default TTL if not specified
        if ttl is None:
            ttl = self.default_ttl

        entry = CacheEntry(
            key=key,
            prompt=prompt,
            service_id=service_id,
            response=response,
            embedding=embedding,
            timestamp=datetime.now(),
            ttl=ttl,
            metadata=metadata
        )

        # Evict if necessary
        self._evict_lru()

        # Store entry
        self.cache[key] = entry
        self.access_order.append(key)

        logger.info(f"Cached response for service '{service_id}'")

        # Persist if configured
        if self.persistence_file:
            await self._save_to_persistence()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = 0.0
        if self.stats["total_requests"] > 0:
            hit_rate = self.stats["hits"] / self.stats["total_requests"]

        return {
            **self.stats,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "similarity_threshold": self.similarity_threshold
        }

    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_order.clear()
        logger.info("Cache cleared")

    async def _save_to_persistence(self):
        """Save cache to persistent storage"""
        if not self.persistence_file:
            return

        try:
            # Convert cache to serializable format
            serializable_cache = {}
            for key, entry in self.cache.items():
                serializable_cache[key] = {
                    **asdict(entry),
                    "embedding": entry.embedding.tolist(),  # Convert numpy array
                    "timestamp": entry.timestamp.isoformat(),
                    "last_accessed": entry.last_accessed.isoformat() if entry.last_accessed else None,
                    "ttl": entry.ttl.total_seconds() if entry.ttl else None
                }

            cache_data = {
                "cache": serializable_cache,
                "access_order": self.access_order,
                "stats": self.stats,
                "saved_at": datetime.now().isoformat()
            }

            with open(self.persistence_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            logger.debug(f"Cache saved to {self.persistence_file}")

        except Exception as e:
            logger.error(f"Failed to save cache to persistence: {e}")

    def _load_from_persistence(self):
        """Load cache from persistent storage"""
        if not self.persistence_file:
            return

        try:
            with open(self.persistence_file, 'r') as f:
                cache_data = json.load(f)

            # Restore cache entries
            for key, entry_data in cache_data.get("cache", {}).items():
                entry = CacheEntry(
                    key=entry_data["key"],
                    prompt=entry_data["prompt"],
                    service_id=entry_data["service_id"],
                    response=entry_data["response"],
                    embedding=np.array(entry_data["embedding"]),
                    timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                    access_count=entry_data.get("access_count", 0),
                    last_accessed=datetime.fromisoformat(entry_data["last_accessed"]) if entry_data.get("last_accessed") else None,
                    ttl=timedelta(seconds=entry_data["ttl"]) if entry_data.get("ttl") else None,
                    metadata=entry_data.get("metadata")
                )

                # Only restore non-expired entries
                if not self._is_expired(entry):
                    self.cache[key] = entry

            # Restore access order and stats
            self.access_order = cache_data.get("access_order", [])
            self.stats.update(cache_data.get("stats", {}))

            logger.info(f"Loaded {len(self.cache)} cache entries from persistence")

        except FileNotFoundError:
            logger.info("No persistence file found, starting with empty cache")
        except Exception as e:
            logger.error(f"Failed to load cache from persistence: {e}")

class CacheAnalytics:
    """Analytics and monitoring for semantic cache"""

    def __init__(self, cache: AdvancedSemanticCache):
        self.cache = cache

    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed cache analytics"""
        stats = self.cache.get_stats()

        # Additional analytics
        service_distribution = {}
        access_patterns = {}

        for entry in self.cache.cache.values():
            # Service distribution
            service_distribution[entry.service_id] = service_distribution.get(entry.service_id, 0) + 1

            # Access patterns
            if entry.last_accessed:
                hour = entry.last_accessed.hour
                access_patterns[hour] = access_patterns.get(hour, 0) + 1

        return {
            **stats,
            "service_distribution": service_distribution,
            "access_patterns": access_patterns,
            "memory_usage_estimate": self._estimate_memory_usage()
        }

    def _estimate_memory_usage(self) -> str:
        """Estimate memory usage of cache"""
        total_size = 0

        for entry in self.cache.cache.values():
            # Approximate size calculation
            total_size += len(entry.prompt.encode('utf-8'))
            total_size += len(json.dumps(entry.response).encode('utf-8'))
            total_size += entry.embedding.nbytes
            total_size += 1000  # Overhead estimate

        # Convert to human readable format
        if total_size < 1024:
            return f"{total_size} B"
        elif total_size < 1024 * 1024:
            return f"{total_size / 1024:.1f} KB"
        else:
            return f"{total_size / (1024 * 1024):.1f} MB"

    def get_performance_report(self) -> str:
        """Generate a performance report"""
        stats = self.get_detailed_stats()

        report = f"""
Semantic Cache Performance Report
================================

Cache Statistics:
- Total Requests: {stats['total_requests']}
- Cache Hits: {stats['hits']}
- Cache Misses: {stats['misses']}
- Hit Rate: {stats['hit_rate']:.2%}
- Evictions: {stats['evictions']}

Cache Configuration:
- Current Size: {stats['cache_size']} / {stats['max_size']}
- Similarity Threshold: {stats['similarity_threshold']}
- Memory Usage: {stats['memory_usage_estimate']}

Service Distribution:
"""

        for service, count in stats['service_distribution'].items():
            report += f"- {service}: {count} entries\n"

        report += "\nAccess Patterns:\n"

        for hour, count in stats['access_patterns'].items():
            report += f"- {hour}: {count} entries\n"

        return report