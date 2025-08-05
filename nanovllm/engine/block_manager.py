from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        # Identify if blocks have the same content
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        Compute the hash of a block, only after a block is fully filled with tokens.
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        # Append the data to previous value, numpy will use little-endian by default
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset() # Reference count is set to 1
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        Allocate blocks for a sequence.
        """
        # Make sure the it's the first time to allocate blocks
        assert not seq.block_table
        h = -1
        cache_miss = False

        # seq.num_blocks is the number of blocks needed to store the sequence,
        # this can be calcualted statically
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # Only compute the hash if the block is full
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                # Cache miss, or the block is not the same as existing one
                cache_miss = True

            if cache_miss:
                # Allocate new block if cache miss
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # Maybe hash table has the block_id but used_block_ids is cleared
                    block = self._allocate_block(block_id)

            if h != -1:
                # Update the hash value of block
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        # Deallocate from end to start
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                # Only release the block if ref count is 0
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # len(self.free_block_ids): How many free blocks does the manager have
        # Only need one more block if the last block has only one token
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        # Get the last block in the block table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            # Has only one token in the last block
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # The last block is full, we need to update the hash and token_ids
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)

            # Calculate the hash of the last block (based on the previous block's hash)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)

            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
