#ifndef REQUEST_H_
#define REQUEST_H_

#include <string>
#include <vector>

class Model;
class GPU;

class Request{
protected:
    int id;
    const Model& model;
    int admissionCycle;
    std::vector<std::string> tokens;
    std::vector<std::size_t> memoryBlocks;
    std::size_t reservedBlocks;
    std::size_t finishedTokenCounter;

public:
    Request(int id, const Model& model, int admissionCycle, std::string prompt);
    Request(const Request& other, const Model& model);
    virtual std::size_t getNeededMemory(const GPU& gpu) const=0;
    virtual std::size_t getMaxNeededMemory() const=0;
    virtual std::vector<std::string> getCacheableTokens() const=0;
    virtual void allocateOneBlock(GPU& gpu, std::size_t slot) const = 0;
    virtual Request* clone(const Model& model) const = 0;
    void deallocateMemory(GPU& gpu);
    virtual void act(GPU& gpu)=0;
    virtual bool finished() const=0;
    virtual void reserveMemory(GPU& gpu) = 0;
    virtual std::string getOutput() const = 0;
    virtual std::size_t getNeedToUnreserve() const = 0;
    virtual ~Request() = default;
    int getId() const;
    const Model& getModel() const;
    int getAdmissionCycle() const;
};

class GenerativeRequest: public Request{
private:
    std::string scriptedOutput;
    std::size_t maxOutputTokens;
    std::size_t outputTokenCounter;
    std::vector<std::string> scriptedTokens;
    std:: vector<std::string> currentTokens; // This vector will hold the tokens that have been processed so far, including both prompt and generated tokens.
    size_t blockCounter;
    std::size_t outputTokenCounter;
    std::vector<std::string> scriptedTokens;
    std:: vector<std::string> currentTokens; // This vector will hold the tokens that have been processed so far, including both prompt and generated tokens.
    size_t blockCounter;
public:
    GenerativeRequest(int id, const Model& model, int admissionCycle, std::string prompt, std::string scriptedOutput,
    std::size_t maxOutputTokens);
    GenerativeRequest(const GenerativeRequest& other, const Model& model);
    bool finished() const override;
    void act(GPU& gpu) override;
    void reserveMemory(GPU& gpu) override;
    std::size_t getNeedToUnreserve() const override;
    std::size_t getNeededMemory(const GPU& gpu) const override;
    std::size_t getMaxNeededMemory() const override;
    std::vector<std::string> getCacheableTokens() const override;
    void allocateOneBlock(GPU& gpu, std::size_t slot) const override;
    std::string getOutput() const override;
    Request* clone(const Model& model) const override;

};

class EmbeddingRequest: public Request{
private:
    std::vector<int> outputEmbeddings; // This vector will hold the output.

public:
    EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt);
    EmbeddingRequest(const EmbeddingRequest& other, const Model& model);
    bool finished() const override;
    void act(GPU& gpu) override;
    void reserveMemory(GPU& gpu) override;
    std::size_t getNeedToUnreserve() const override;
    std::size_t getNeededMemory(const GPU& gpu) const override;
    std::size_t getMaxNeededMemory() const override;
    void allocateOneBlock(GPU& gpu, std::size_t slot) const override;
    std::string getOutput() const override;
    std::vector<std::string> getCacheableTokens() const override;
    Request* clone(const Model& model) const override;

};



#endif