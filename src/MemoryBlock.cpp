#include "MemoryBlock.h"
#include <algorithm>
#include <stdexcept>

MemoryBlock::MemoryBlock(size_t size)
    : size(size), refCount(0), data(size > 0 ? new int[size]() : nullptr) {}

MemoryBlock::~MemoryBlock() {
    delete[] data;
}

MemoryBlock::MemoryBlock(const MemoryBlock& other)
    : size(other.size),
      refCount(0),
      data(other.size > 0 ? new int[other.size]() : nullptr) {
    if (data != nullptr && other.data != nullptr) {
        std::copy(other.data, other.data + size, data);
    }
}

MemoryBlock& MemoryBlock::operator=(const MemoryBlock& other) {
    if (this == &other) {
        return *this;
    }

    int* newData = (other.size > 0) ? new int[other.size]() : nullptr;
    if (newData != nullptr && other.data != nullptr) {
        std::copy(other.data, other.data + other.size, newData);
    }

    delete[] data;
    size = other.size;
    refCount = 0;
    data = newData;

    return *this;
}

MemoryBlock::MemoryBlock(MemoryBlock&& other) noexcept
    : size(other.size), refCount(other.refCount), data(other.data) {
    other.size = 0;
    other.refCount = 0;
    other.data = nullptr;
}

MemoryBlock& MemoryBlock::operator=(MemoryBlock&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    delete[] data;

    size = other.size;
    refCount = other.refCount;
    data = other.data;

    other.size = 0;
    other.refCount = 0;
    other.data = nullptr;

    return *this;
}

int MemoryBlock::readFromBlock(std::size_t offset) const {
    if (offset >= size) {
        throw std::out_of_range("Block read offset out of range");
    }
    return data[offset];
}

void MemoryBlock::writeToBlock(std::size_t offset, int value) {
    if (offset >= size) {
        throw std::out_of_range("Block write offset out of range");
    }
    data[offset] = value;
}

void MemoryBlock::acquire() {
    ++refCount;
}

IdleAction MemoryBlock::release() {
    if (refCount > 0) {
        --refCount;
    }
    if (refCount == 0) {
        return onLastRelease();
    }
    return IdleAction::KeepCached;
}

int MemoryBlock::getRefCount() const {
    return refCount;
}

size_t MemoryBlock::getSize() const {
    return size;
}

KVBlock::KVBlock(size_t size) : MemoryBlock(size) {}

IdleAction KVBlock::onLastRelease() const {
    return IdleAction::KeepCached;
}

MemoryBlock* KVBlock::clone() const {
    return new KVBlock(*this);
}

EmbeddingWorkspace::EmbeddingWorkspace(size_t size) : MemoryBlock(size) {}

IdleAction EmbeddingWorkspace::onLastRelease() const {
    return IdleAction::Destroy;
}

MemoryBlock* EmbeddingWorkspace::clone() const {
    return new EmbeddingWorkspace(*this);
}