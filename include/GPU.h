#ifndef GPU_H_
#define GPU_H_

#include <string>
#include <cstddef>
#include <vector>
#include <unordered_map>

struct PrefixMatch {
    std::size_t matchedTokens;
    std::vector<std::size_t> blockIds;
};

class Request;
class GenerativeRequest;
class EmbeddingRequest;
class MemoryBlock;

class GPU {
public:
    GPU(std::size_t blocksNum, std::size_t blockSize);
    ~GPU();
    GPU(const GPU& other);
    GPU& operator=(const GPU& other);
    GPU(GPU&& other) noexcept;
    GPU& operator=(GPU&& other) noexcept;

    std::size_t availableSlots() const;
    std::size_t getBlocksForMemory(std::size_t memory) const;
    void reserve(std::size_t n, const Request& request);
    void unreserve(std::size_t n, const Request& request);
    std::size_t allocateBlock(const Request& request);
    void addMemoryBlock(const GenerativeRequest& request, std::size_t slot);
    void addMemoryBlock(const EmbeddingRequest& request, std::size_t slot);
    void acquire(std::size_t slot);
    void release(std::size_t slot);
    void copyMemoryFromOtherGPU(const GPU& other, const std::vector<std::size_t>& blocks);
    void assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId);
    int readFromBlock(std::size_t blockId, std::size_t offset) const;
    void writeToBlock(std::size_t blockId, std::size_t offset, int value);
    PrefixMatch findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize,
                                        const std::vector<std::string>& promptTokens) const;

    std::size_t getCacheHits() const;
    std::size_t getEvictions() const;
    std::size_t getBlocksNum() const;
    std::size_t getBlockSize() const;

private:
    struct SlotMetadata {
        bool hasPrefix;
        std::string modelId;
        std::vector<std::string> prefixTokens;

        SlotMetadata() : hasPrefix(false), modelId(), prefixTokens() {}
        void clear() {
            hasPrefix = false;
            modelId.clear();
            prefixTokens.clear();
        }
    };

    void clearMemory();
    bool hasPrefixKey(const std::string& modelId, const std::vector<std::string>& prefixTokens) const;

    std::size_t blocksNum;
    std::size_t blockSize;
    std::vector<MemoryBlock*> memory;
    std::vector<SlotMetadata> slotInfo;
    std::unordered_map<int, std::size_t> reservedByRequest;
    std::size_t totalReserved;
    std::size_t cacheHits;
    std::size_t evictions;
};

#endif