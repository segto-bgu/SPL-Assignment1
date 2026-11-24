#include "GPU.h"
#include "MemoryBlock.h"
#include "Request.h"
#include <stdexcept>

GPU::GPU(std::size_t blocksNum, std::size_t blockSize)
    : blocksNum(blocksNum),
      blockSize(blockSize),
      memory(blocksNum, nullptr),
      slotInfo(blocksNum),
      reservedByRequest(),
      totalReserved(0),
      cacheHits(0),
      evictions(0) {}

void GPU::clearMemory() {
    for (std::size_t i = 0; i < memory.size(); ++i) {
        delete memory[i];
        memory[i] = nullptr;
    }
}

GPU::~GPU() {
    clearMemory();
}

GPU::GPU(const GPU& other)
    : blocksNum(other.blocksNum),
      blockSize(other.blockSize),
      memory(other.blocksNum, nullptr),
      slotInfo(other.slotInfo),
      reservedByRequest(other.reservedByRequest),
      totalReserved(other.totalReserved),
      cacheHits(other.cacheHits),
      evictions(other.evictions) {
    for (std::size_t i = 0; i < blocksNum; ++i) {
        if (other.memory[i] != nullptr) {
            memory[i] = other.memory[i]->clone();
            int refs = other.memory[i]->getRefCount();
            for (int r = 0; r < refs; ++r) {
                memory[i]->acquire();
            }
        }
    }
}

GPU& GPU::operator=(const GPU& other) {
    if (this == &other) {
        return *this;
    }

    std::vector<MemoryBlock*> newMemory(other.blocksNum, nullptr);
    try {
        for (std::size_t i = 0; i < other.blocksNum; ++i) {
            if (other.memory[i] != nullptr) {
                newMemory[i] = other.memory[i]->clone();
                int refs = other.memory[i]->getRefCount();
                for (int r = 0; r < refs; ++r) {
                    newMemory[i]->acquire();
                }
            }
        }
    } catch (...) {
        for (std::size_t i = 0; i < newMemory.size(); ++i) {
            delete newMemory[i];
        }
        throw;
    }

    clearMemory();
    blocksNum = other.blocksNum;
    blockSize = other.blockSize;
    memory = std::move(newMemory);
    slotInfo = other.slotInfo;
    reservedByRequest = other.reservedByRequest;
    totalReserved = other.totalReserved;
    cacheHits = other.cacheHits;
    evictions = other.evictions;

    return *this;
}

GPU::GPU(GPU&& other) noexcept
    : blocksNum(other.blocksNum),
      blockSize(other.blockSize),
      memory(std::move(other.memory)),
      slotInfo(std::move(other.slotInfo)),
      reservedByRequest(std::move(other.reservedByRequest)),
      totalReserved(other.totalReserved),
      cacheHits(other.cacheHits),
      evictions(other.evictions) {
    other.blocksNum = 0;
    other.blockSize = 0;
    other.totalReserved = 0;
    other.cacheHits = 0;
    other.evictions = 0;
}

GPU& GPU::operator=(GPU&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    clearMemory();

    blocksNum = other.blocksNum;
    blockSize = other.blockSize;
    memory = std::move(other.memory);
    slotInfo = std::move(other.slotInfo);
    reservedByRequest = std::move(other.reservedByRequest);
    totalReserved = other.totalReserved;
    cacheHits = other.cacheHits;
    evictions = other.evictions;

    other.blocksNum = 0;
    other.blockSize = 0;
    other.totalReserved = 0;
    other.cacheHits = 0;
    other.evictions = 0;

    return *this;
}

std::size_t GPU::availableSlots() const {
    std::size_t activeBlocks = 0;
    for (std::size_t i = 0; i < blocksNum; ++i) {
        if (memory[i] != nullptr && memory[i]->getRefCount() > 0) {
            ++activeBlocks;
        }
    }
    if (activeBlocks + totalReserved >= blocksNum) {
        return 0;
    }
    return blocksNum - activeBlocks - totalReserved;
}

std::size_t GPU::getBlocksForMemory(std::size_t memoryAmount) const {
    if (memoryAmount == 0 || blockSize == 0) {
        return 0;
    }
    return (memoryAmount + blockSize - 1) / blockSize;
}

void GPU::reserve(std::size_t n, const Request& request) {
    if (n == 0) {
        return;
    }
    reservedByRequest[request.getId()] += n;
    totalReserved += n;
}

void GPU::unreserve(std::size_t n, const Request& request) {
    if (n == 0) {
        return;
    }
    auto it = reservedByRequest.find(request.getId());
    if (it == reservedByRequest.end()) {
        return;
    }
    std::size_t toRemove = (n <= it->second) ? n : it->second;
    it->second -= toRemove;
    totalReserved -= toRemove;
    if (it->second == 0) {
        reservedByRequest.erase(it);
    }
}

std::size_t GPU::allocateBlock(const Request& request) {
    auto it = reservedByRequest.find(request.getId());
    if (it == reservedByRequest.end() || it->second == 0) {
        throw std::runtime_error("Request has no reserved blocks");
    }

    std::size_t chosenSlot = blocksNum;

    // 1. Prefer the lowest-index empty slot
    for (std::size_t i = 0; i < blocksNum; ++i) {
        if (memory[i] == nullptr) {
            chosenSlot = i;
            break;
        }
    }

    // 2. Otherwise evict the lowest-index idle block (refCount == 0)
    if (chosenSlot == blocksNum) {
        for (std::size_t i = 0; i < blocksNum; ++i) {
            if (memory[i] != nullptr && memory[i]->getRefCount() == 0) {
                delete memory[i];
                memory[i] = nullptr;
                slotInfo[i].clear();
                ++evictions;
                chosenSlot = i;
                break;
            }
        }
    }

    if (chosenSlot == blocksNum) {
        throw std::runtime_error("No empty or idle slot available");
    }

    // Double dispatch creates KVBlock or EmbeddingWorkspace in chosenSlot
    request.allocateOneBlock(*this, chosenSlot);
    slotInfo[chosenSlot].clear();

    // Consume 1 reserved block from the request
    --it->second;
    --totalReserved;
    if (it->second == 0) {
        reservedByRequest.erase(it);
    }

    // Acquire immediately (refCount becomes 1; not counted as cache hit since hasPrefix is false)
    acquire(chosenSlot);

    return chosenSlot;
}

void GPU::addMemoryBlock(const GenerativeRequest& request, std::size_t slot) {
    (void)request;
    if (slot >= blocksNum) {
        throw std::out_of_range("Slot index out of range");
    }
    memory[slot] = new KVBlock(blockSize);
    slotInfo[slot].clear();
}

void GPU::addMemoryBlock(const EmbeddingRequest& request, std::size_t slot) {
    (void)request;
    if (slot >= blocksNum) {
        throw std::out_of_range("Slot index out of range");
    }
    memory[slot] = new EmbeddingWorkspace(blockSize);
    slotInfo[slot].clear();
}

void GPU::acquire(std::size_t slot) {
    if (slot >= blocksNum || memory[slot] == nullptr) {
        throw std::out_of_range("Invalid slot in acquire");
    }
    if (slotInfo[slot].hasPrefix) {
        ++cacheHits;
    }
    memory[slot]->acquire();
}

void GPU::release(std::size_t slot) {
    if (slot >= blocksNum || memory[slot] == nullptr) {
        throw std::out_of_range("Invalid slot in release");
    }
    IdleAction action = memory[slot]->release();
    if (memory[slot]->getRefCount() == 0) {
        if (action == IdleAction::Destroy || !slotInfo[slot].hasPrefix) {
            delete memory[slot];
            memory[slot] = nullptr;
            slotInfo[slot].clear();
        }
    }
}

bool GPU::hasPrefixKey(const std::string& modelId, const std::vector<std::string>& prefixTokens) const {
    for (std::size_t i = 0; i < blocksNum; ++i) {
        if (memory[i] != nullptr &&
            slotInfo[i].hasPrefix &&
            slotInfo[i].modelId == modelId &&
            slotInfo[i].prefixTokens == prefixTokens) {
            return true;
        }
    }
    return false;
}

void GPU::copyMemoryFromOtherGPU(const GPU& other, const std::vector<std::size_t>& blocks) {
    std::vector<bool> protectedSlot(blocksNum, false);

    // Protect any blocks already on this GPU that match the chain being copied
    for (std::size_t srcSlot : blocks) {
        if (srcSlot >= other.blocksNum || other.memory[srcSlot] == nullptr || !other.slotInfo[srcSlot].hasPrefix) {
            continue;
        }
        const SlotMetadata& srcMeta = other.slotInfo[srcSlot];
        for (std::size_t i = 0; i < blocksNum; ++i) {
            if (memory[i] != nullptr &&
                slotInfo[i].hasPrefix &&
                slotInfo[i].modelId == srcMeta.modelId &&
                slotInfo[i].prefixTokens == srcMeta.prefixTokens) {
                protectedSlot[i] = true;
            }
        }
    }

    for (std::size_t srcSlot : blocks) {
        if (srcSlot >= other.blocksNum || other.memory[srcSlot] == nullptr || !other.slotInfo[srcSlot].hasPrefix) {
            continue;
        }

        const SlotMetadata& srcMeta = other.slotInfo[srcSlot];
        if (hasPrefixKey(srcMeta.modelId, srcMeta.prefixTokens)) {
            continue;
        }

        std::size_t dstSlot = blocksNum;

        // 1. Lowest-index empty slot
        for (std::size_t i = 0; i < blocksNum; ++i) {
            if (memory[i] == nullptr) {
                dstSlot = i;
                break;
            }
        }

        // 2. Lowest-index unprotected idle block
        if (dstSlot == blocksNum) {
            for (std::size_t i = 0; i < blocksNum; ++i) {
                if (!protectedSlot[i] && memory[i] != nullptr && memory[i]->getRefCount() == 0) {
                    delete memory[i];
                    memory[i] = nullptr;
                    slotInfo[i].clear();
                    ++evictions;
                    dstSlot = i;
                    break;
                }
            }
        }

        if (dstSlot == blocksNum) {
            throw std::runtime_error("No slot available for copying cached block");
        }

        memory[dstSlot] = other.memory[srcSlot]->clone(); // clone starts with refCount == 0 (idle)
        slotInfo[dstSlot] = srcMeta;
        protectedSlot[dstSlot] = true;
    }
}

void GPU::assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId) {
    if (block >= blocksNum || memory[block] == nullptr) {
        throw std::out_of_range("Invalid slot in assignPrefixToBlock");
    }

    // Do not cache if another block already holds the exact same model and prefix key
    for (std::size_t i = 0; i < blocksNum; ++i) {
        if (i != block &&
            memory[i] != nullptr &&
            slotInfo[i].hasPrefix &&
            slotInfo[i].modelId == modelId &&
            slotInfo[i].prefixTokens == prefixTokens) {
            return;
        }
    }

    slotInfo[block].hasPrefix = true;
    slotInfo[block].modelId = modelId;
    slotInfo[block].prefixTokens = prefixTokens;
}

int GPU::readFromBlock(std::size_t blockId, std::size_t offset) const {
    if (blockId >= blocksNum || memory[blockId] == nullptr) {
        throw std::out_of_range("Invalid blockId in readFromBlock");
    }
    return memory[blockId]->readFromBlock(offset);
}

void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value) {
    if (blockId >= blocksNum || memory[blockId] == nullptr) {
        throw std::out_of_range("Invalid blockId in writeToBlock");
    }
    memory[blockId]->writeToBlock(offset, value);
}

PrefixMatch GPU::findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize,
                                         const std::vector<std::string>& promptTokens) const {
    PrefixMatch result{0, {}};
    if (embeddingSize == 0 || blockSize < embeddingSize) {
        return result;
    }

    std::size_t tokensPerBlock = blockSize / embeddingSize;
    for (std::size_t endToken = tokensPerBlock; endToken <= promptTokens.size(); endToken += tokensPerBlock) {
        std::vector<std::string> targetKey(promptTokens.begin(), promptTokens.begin() + endToken);
        bool found = false;
        for (std::size_t s = 0; s < blocksNum; ++s) {
            if (memory[s] != nullptr &&
                slotInfo[s].hasPrefix &&
                slotInfo[s].modelId == modelId &&
                slotInfo[s].prefixTokens == targetKey) {
                result.blockIds.push_back(s);
                result.matchedTokens = endToken;
                found = true;
                break;
            }
        }
        if (!found) {
            break;
        }
    }

    return result;
}

std::size_t GPU::getCacheHits() const {
    return cacheHits;
}

std::size_t GPU::getEvictions() const {
    return evictions;
}

std::size_t GPU::getBlocksNum() const {
    return blocksNum;
}

std::size_t GPU::getBlockSize() const {
    return blockSize;
}