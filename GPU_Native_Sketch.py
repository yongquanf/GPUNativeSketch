# lab: filtering using pytorch tensor
import cupy as cp

# global variables
hash_murmur3_seeding = cp.ElementwiseKernel(
    in_params='raw int32 key, int32 seed, int32 _capacity',
    out_params='raw int32 y',
    operation='''
    int k = seed;
    y[i] = key[i];
    y[i] ^= k;
    y[i] ^= y[i] >> 16;
    y[i] *= 0x85ebca6b;
    y[i] ^= y[i] >> 13;
    y[i] *= 0xc2b2ae35;
    y[i] ^= y[i] >> 16;
    y[i] = y[i] % _capacity;
    ''',
    name='hash_murmur3_seeding'
)


def murmur3_cp_array(key, seed, _capacity):
    y = cp.empty(key.size, dtype=cp.int32)
    # cp, output hashing
    hash_murmur3_seeding(key, seed, _capacity, y, size=y.size)
    return y

# single item key


def murmur3_cp(input_key, seed, _capacity):
    '''
    single key, output one hash value
    '''
    key = cp.array([input_key], dtype=cp.int32)
    y = murmur3_cp_array(key, seed, _capacity)
    return y[0]


# test
N = 1000000
key = cp.arange(N, dtype=cp.int32)
_capacity = cp.int32(1000)
seed = cp.int32(123)
y = murmur3_cp_array(key, seed, _capacity)
print(y)

# test single key
input_key = cp.int32(23456)
y = murmur3_cp(input_key, seed, _capacity)
print(y)


# Bloom filter on gpu, murmur3


class Basis_BloomFilter:
    def __init__(self, capacity, error_rate):
        self.capacity = capacity
        self.error_rate = error_rate
        self.bit_array = cp.zeros(capacity, dtype=cp.int32)
        self.hash_count = self.get_hash_count(capacity, error_rate)

    def add(self, item):
        for seed in range(self.hash_count):
            index = murmur3_cp(item, cp.int32(seed), cp.int32(self.capacity))
            self.bit_array[index] = 1

    def contains(self, item):
        for seed in range(self.hash_count):
            index = murmur3_cp(item, cp.int32(seed), cp.int32(self.capacity))
            if self.bit_array[index] == 0:
                return False
        return True

    def get_hash_count(self, capacity, error_rate):
        return int((capacity * abs(cp.log(error_rate))) / (cp.log(2) ** 2))

# version two
# batch key


class Batch_Bloom_Filter:
    def __init__(self, capacity, error_rate):
        self.capacity = capacity
        self.error_rate = error_rate
        self.bit_array = cp.zeros(capacity, dtype=cp.int32)
        self.hash_count = self.get_hash_count(capacity, error_rate)

    def add(self, items):
        for seed in range(self.hash_count):
            indices = murmur3_cp_array(
                items, cp.int32(seed), cp.int32(self.capacity))
            self.bit_array[indices] = 1

    def contains(self, items):
        # initialize the existence array cp.ones(self.bit_array, dtype=cp.bool)
        existence_array = cp.ones(items.size, dtype=cp.bool)
        for seed in range(self.hash_count):
            indices = murmur3_cp_array(
                items, cp.int32(seed), cp.int32(self.capacity))
            # update existence array on indices to self.bit_array[indices]
            existence_array = cp.logical_and(
                existence_array, self.bit_array[indices])
        return existence_array

    def get_hash_count(self, capacity, error_rate):
        return int((capacity * abs(cp.log(error_rate))) / (cp.log(2) ** 2))


# count sketch, signed sum
# version one
class count_sketch:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = cp.zeros((width, depth), dtype=cp.int32)

    def add(self, item):
        for i in range(self.depth):
            index = murmur3_cp(item, cp.int32(i), cp.int32(self.width))
            sign = murmur3_cp(item, cp.int32(i + self.depth),
                              cp.int32(2)) * 2 - 1
            self.table[index, i] += sign

    def insert_value(self, item, value):
        for i in range(self.depth):
            index = murmur3_cp(item, cp.int32(i), cp.int32(self.width))
            sign = murmur3_cp(item, cp.int32(i + self.depth),
                              cp.int32(2)) * 2 - 1
            self.table[index, i] += sign * value

    def query(self, item):
        result = cp.zeros(self.depth, dtype=cp.int32)
        bkt_indexes = cp.zeros(self.depth, dtype=cp.int32)
        # offset at all buckets in 1d
        for i in range(self.depth):
            index = murmur3_cp(item, cp.int32(i), cp.int32(self.width))
            sign = murmur3_cp(item, cp.int32(i + self.depth),
                              cp.int32(2)) * 2 - 1
            result[i] = sign * self.table[index, i]
            # offset
            offset = i*self.width
            bkt_indexes[i] = index + offset
        return result, bkt_indexes

# version two


class count_sketch_v2:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = cp.zeros((width, depth), dtype=cp.int32)

    def add(self, items):
        for i in range(self.depth):
            indices = murmur3_cp_array(
                items, cp.int32(i), cp.int32(self.width))
            signs = murmur3_cp_array(
                items, cp.int32(i + self.depth), cp.int32(2)) * 2 - 1
            self.table[indices, i] += signs

    def query(self, items):
        result = cp.zeros((items.size, self.depth), dtype=cp.int32)
        for i in range(self.depth):
            indices = murmur3_cp_array(
                items, cp.int32(i), cp.int32(self.width))
            signs = murmur3_cp_array(
                items, cp.int32(i + self.depth), cp.int32(2)) * 2 - 1
            result[:, i] = signs * self.table[indices, i]
        return result

# count_min
# version one


class count_min:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = cp.zeros((width, depth), dtype=cp.int32)

    def add(self, item):
        for i in range(self.depth):
            index = murmur3_cp(item, cp.int32(i), cp.int32(self.width))
            self.table[index, i] += 1

    def query(self, item):
        result = cp.zeros(self.depth, dtype=cp.int32)
        for i in range(self.depth):
            index = murmur3_cp(item, cp.int32(i), cp.int32(self.width))
            result[i] = self.table[index, i]
        return result

# version two


class count_min_v2:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = cp.zeros((width, depth), dtype=cp.int32)

    def add(self, items):
        for i in range(self.depth):
            indices = murmur3_cp_array(
                items, cp.int32(i), cp.int32(self.width))
            self.table[indices, i] += 1

    def query(self, items):
        result = cp.zeros((items.size, self.depth), dtype=cp.int32)
        for i in range(self.depth):
            indices = murmur3_cp_array(
                items, cp.int32(i), cp.int32(self.width))
            result[:, i] = self.table[indices, i]
        return result

# cuckoo hash table
# version one


class cuckoo_hash_table:
    def __init__(self, capacity):
        self.capacity = capacity
        self.table = cp.zeros(capacity, dtype=cp.int32)
        self.hash_count = 2

    def add(self, item):
        for seed in range(self.hash_count):
            index = murmur3_cp(item, cp.int32(seed), cp.int32(self.capacity))
            if self.table[index] == 0:
                self.table[index] = item
                return
        # kick out
        index = murmur3_cp(item, cp.int32(0), cp.int32(self.capacity))
        old_item = self.table[index]
        self.table[index] = item
        self.add(old_item)

    def contains(self, item):
        for seed in range(self.hash_count):
            index = murmur3_cp(item, cp.int32(seed), cp.int32(self.capacity))
            if self.table[index] == item:
                return True
        return False

# version two


class cuckoo_hash_table_v2:
    def __init__(self, capacity):
        self.capacity = capacity
        self.table = cp.zeros(capacity, dtype=cp.int32)
        self.hash_count = 2

    def add(self, items):
        for seed in range(self.hash_count):
            indices = murmur3_cp_array(
                items, cp.int32(seed), cp.int32(self.capacity))
            empty_indices = indices[self.table[indices] == 0]
            self.table[empty_indices] = items[empty_indices]
            if empty_indices.size == 0:
                old_items = self.table[indices]
                self.table[indices] = items
                self.add(old_items)
                return

    def contains(self, items):
        existence_array = cp.zeros(items.size, dtype=cp.bool)
        for seed in range(self.hash_count):
            indices = murmur3_cp_array(
                items, cp.int32(seed), cp.int32(self.capacity))
            existence_array = cp.logical_or(
                existence_array, self.table[indices] == items)
        return existence_array

#
# search on each dimension for cupy array:
# e.g., X(,,), locate X(*,1) == 24, return the index of the first dimension


def filter_positions_on_array(X, locate_dim_index, to_be_located_val):
    '''
    indexing the entries in X, where the to_be_located_val is located in the locate_dim_index for all dimensions
    '''
    # Create an indexer object with the same number of dimensions as X
    indexer = [slice(None)] * X.ndim

    # Use cp.where to find the indices where X in the locate_dim_index dimension equals to_be_located_val
    indices = cp.where(X[tuple(indexer)] == to_be_located_val)

    # Filter the indices based on the locate_dim_index dimension
    filtered_indices = indices[locate_dim_index]

    return filtered_indices
