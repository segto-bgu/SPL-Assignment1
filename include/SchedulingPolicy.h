#ifndef SCHEDULING_POLICY_H_
#define SCHEDULING_POLICY_H
#include <vector>
#include "Request.h"
#include "GPU.h"
#include <vector>
#include "Request.h"
#include "GPU.h"

class SchedulingPolicy {
public:
    virtual ~SchedulingPolicy() = default;
    virtual std::vector<int> choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const = 0;
    virtual SchedulingPolicy* clone() const = 0;
    virtual ~SchedulingPolicy() = default;
    
};

class ArrivalOrderPolicy : public SchedulingPolicy {
    public:
    std::vector<int> choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const override;
    SchedulingPolicy* clone() const override { return new ArrivalOrderPolicy(*this); }
};
class ShortestJobPolicy : public SchedulingPolicy { 
    public:
    std::vector<int> choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const override;    
    SchedulingPolicy* clone() const override { return new ShortestJobPolicy(*this); }   
};
class LongestPrefixPolicy : public SchedulingPolicy {
    public:
    std::vector<int> choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const override; 
    SchedulingPolicy* clone() const override { return new LongestPrefixPolicy(*this); }   
};


#endif