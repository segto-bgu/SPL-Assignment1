#include "SchedulingPolicy.h"
#include <algorithm>
#include "GPU.h"
#include "Model.h"
std::vector<int> ArrivalOrderPolicy::choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const {
    std::vector<int> order;
    for (size_t i = 0; i < eligible.size(); ++i) {
    order.push_back(i); }
    
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (eligible[a]->getAdmissionCycle() == eligible[b]->getAdmissionCycle()) {
            return eligible[a]->getId() < eligible[b]->getId();
        }
        return eligible[a]->getAdmissionCycle() < eligible[b]->getAdmissionCycle();
    });

    return order;
}

std::vector<int> ShortestJobPolicy::choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const {
    std::vector<int> order;
    for (size_t i = 0; i < eligible.size(); ++i) {
        order.push_back(i);
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (gpus[0].getBlocksForMemory(eligible[a]->getMaxNeededMemory()) == gpus[0].getBlocksForMemory(eligible[b]->getMaxNeededMemory())) {
            return eligible[a]->getId() < eligible[b]->getId();
        }
        return gpus[0].getBlocksForMemory(eligible[a]->getMaxNeededMemory()) < gpus[0].getBlocksForMemory(eligible[b]->getMaxNeededMemory());
    });

    return order;
}   

std::vector<int> LongestPrefixPolicy::choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const {
    std::vector<int> order;
    std::vector<PrefixMatch> matches(eligible.size());
    for (size_t i = 0; i < eligible.size(); ++i) {
        order.push_back(i);
        PrefixMatch match = gpus[0].findLongestCachedPrefix(eligible[i]->getModel().getModelId(), eligible[i]->getModel().getEmbeddingSize(), eligible[i]->getCacheableTokens());
        for (size_t j = 0; j < gpus.size(); j++)
        {
            PrefixMatch temp = gpus[j].findLongestCachedPrefix(eligible[i]->getModel().getModelId(), eligible[i]->getModel().getEmbeddingSize(), eligible[i]->getCacheableTokens());
            if (temp.matchedTokens > match.matchedTokens) {
                match = temp;   
            }  
        }    
        matches[i] = match;
  
    }
    

    
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (matches[a].matchedTokens == matches[b].matchedTokens) {
            return eligible[a]->getId() < eligible[b]->getId();
        }
        return matches[a].matchedTokens > matches[b].matchedTokens;
    });

    return order;
}

