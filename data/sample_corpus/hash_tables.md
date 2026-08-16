# Hash Tables

A hash table is a data structure that maps keys to values using a hash
function to compute an index into an array of buckets, from which the
desired value can be found. In the average case, lookup, insertion, and
deletion all run in O(1) time, which is the main reason hash tables are the
default choice for implementing sets, maps, and caches in most languages
(Python's `dict`, Java's `HashMap`, C++'s `unordered_map`).

## Collision Resolution

Because the hash function maps a large key space onto a small array of
buckets, collisions - two different keys hashing to the same bucket - are
inevitable. There are two dominant strategies for handling them. Separate
chaining stores a linked list (or small dynamic array) of entries at each
bucket, so a collision just appends to that bucket's list; lookup degrades
to O(k) where k is the number of entries in that bucket. Open addressing
instead stores every entry directly in the bucket array and, on a collision,
probes for the next open slot using a fixed rule (linear probing, quadratic
probing, or double hashing); this avoids the memory overhead of linked
lists but requires careful handling of deletions, since simply clearing a
slot can break the probe chain for later lookups.

## Load Factor and Resizing

The load factor is the ratio of stored entries to the number of buckets. As
the load factor grows, collisions become more frequent and performance
degrades toward O(n) in the worst case. Most hash table implementations
trigger a resize - typically doubling the bucket array and rehashing every
existing entry - once the load factor crosses a threshold (commonly 0.75).
This resize is an O(n) operation, but because it happens exponentially less
often as the table grows, the amortized cost per insertion remains O(1).

## Hash Function Quality

A good hash function distributes keys uniformly across buckets to minimize
collisions and should be fast to compute, since it runs on every operation.
A poor hash function - one that clusters many keys into the same bucket -
can degrade every operation to O(n) regardless of the collision resolution
strategy used. This is also the basis of a denial-of-service attack vector
called "hash flooding," where an attacker crafts inputs that all hash to
the same bucket; many languages now seed their hash functions randomly at
startup specifically to prevent an attacker from predicting collisions.

## LRU Cache Implementation

A classic interview problem is implementing an LRU (Least Recently Used)
cache with O(1) get and put operations. The standard solution combines a
hash table (for O(1) key lookup) with a doubly linked list (for O(1)
reordering of recency): the hash table maps keys to nodes in the linked
list, and the linked list is kept ordered from most- to least-recently-used
so that evicting the least recently used entry is just removing the tail
node.
