#include "Request.h"
#include "Model.h"
#include <vector>
#include "GPU.h"
#include <string>

// Constructor for the Request class
Request ::Request(int id, const Model& model, int admissionCycle, std::string prompt)
    : id(id), model(model), admissionCycle(admissionCycle), tokens(model.tokenizer(prompt)), memoryBlocks(), reservedBlocks(0), finishedTokenCounter(0){}

GenerativeRequest::GenerativeRequest(int id, const Model& model, int admissionCycle, std::string prompt, std::string scriptedOutput,
    std::size_t maxOutputTokens)
    : Request(id, model, admissionCycle, prompt), scriptedOutput(scriptedOutput), maxOutputTokens(maxOutputTokens), outputTokenCounter(0), scriptedTokens(model.tokenizer(scriptedOutput)),currentTokens(),blockCounter(0){}

EmbeddingRequest::EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt)
    : Request(id, model, admissionCycle, prompt),outputEmbeddings() {}


//finished() method for GenerativeRequest and EmbeddingRequest classes
bool GenerativeRequest::finished()const{
    return (outputTokenCounter >= maxOutputTokens && finishedTokenCounter  >= tokens.size() ) || finishedTokenCounter + outputTokenCounter >= scriptedTokens.size() + tokens.size();

} 

bool EmbeddingRequest::finished() const {
    return finishedTokenCounter >= tokens.size();
}

//act() method for GenerativeRequest and EmbeddingRequest classes
void GenerativeRequest::act(GPU& gpu){
    //building the currentTokens vector with the tokens that have been processed so far, including both prompt and generated tokens.
    if(currentTokens.size() < tokens.size()){
        currentTokens.push_back(tokens[currentTokens.size()]);
    }
    else if(outputTokenCounter < maxOutputTokens && finishedTokenCounter >= tokens.size() && outputTokenCounter < scriptedTokens.size()){
        currentTokens.push_back(scriptedTokens[currentTokens.size() - tokens.size()]);
    }
    //cal the offset and the slot and check if we need to allocate a new block
    std::size_t offset = ((currentTokens.size() - 1) * model.getEmbeddingSize()) % gpu.getBlockSize();
    std::size_t numSlot = (((currentTokens.size() - 1) * model.getEmbeddingSize()) - offset) / gpu.getBlockSize(); 
    if(offset == 0){
        memoryBlocks.push_back(gpu.allocateBlock(*this));
        blockCounter++;
    }
    //finally send the token to the model
    model.generateTokenEmbedding(currentTokens, gpu, memoryBlocks[numSlot], offset);
    //check if we need to assign the prefix to the block
    if(offset + model.getEmbeddingSize() == gpu.getBlockSize()){
        gpu.assignPrefixToBlock(memoryBlocks.back(), currentTokens, model.getModelId());
    }
    //increment the finishedTokenCounter or the outputTokenCounter depending on the state of the request
    if(finishedTokenCounter < tokens.size()){
        finishedTokenCounter++;
    }
    else if(outputTokenCounter < maxOutputTokens && finishedTokenCounter >= tokens.size()){
        outputTokenCounter++;
    }

}

void EmbeddingRequest::act(GPU& gpu){
    std::vector<std::string> temp;
    temp.push_back(tokens[finishedTokenCounter]);
    //finally send the token to the model
    int placement;
    int offset;
    if(gpu.getBlockSize() < 2 * model.getEmbeddingSize()){
        placement = 1;
        offset = 0;
    }
    else{
        placement = 0;
        offset = model.getEmbeddingSize();
    }
    model.generateTokenEmbedding(temp, gpu, memoryBlocks[placement], offset);
    //copy the embedding from the second block to the first block's sum
    for(size_t i = 0; i < model.getEmbeddingSize(); i++){
        gpu.writeToBlock(memoryBlocks[0],i,gpu.readFromBlock(memoryBlocks[0],i) + gpu.readFromBlock(memoryBlocks[placement], offset + i));
    }
    //if needed preperaing the  output
    finishedTokenCounter++;
    if(finishedTokenCounter == tokens.size()){
        for(size_t i = 0; i < model.getEmbeddingSize(); i++){
            outputEmbeddings.push_back(gpu.readFromBlock(memoryBlocks[0],i));
        }
    }
}

//reserveMemory() method for GenerativeRequest class
void GenerativeRequest::reserveMemory(GPU& gpu){
    PrefixMatch match = gpu.findLongestCachedPrefix(model.getModelId(), model.getEmbeddingSize(), tokens);
    for(int i: match.blockIds){
        gpu.acquire(i);
        memoryBlocks.push_back(i);
    }
    for(size_t i = 0; i < match.matchedTokens; i++){
        finishedTokenCounter++;
        currentTokens.push_back(tokens[i]);
    }
    
    std::size_t neededBlocks = gpu.getBlocksForMemory(getNeededMemory(gpu));
    reservedBlocks = neededBlocks;
    gpu.reserve(neededBlocks, *this);
}

void EmbeddingRequest::reserveMemory(GPU& gpu){
    if(gpu.getBlockSize() < 2 * model.getEmbeddingSize()){
        gpu.reserve(2, *this);
        reservedBlocks = 2;
    }
    else{
        gpu.reserve(1, *this);
        reservedBlocks = 1;
    }
    for(size_t i = 0; i < reservedBlocks; i++){
        memoryBlocks.push_back(gpu.allocateBlock(*this));
        
    }
}

//deallocateMemory() method for GenerativeRequest and EmbeddingRequest classes
void Request::deallocateMemory(GPU& gpu){
    int toUnreserve = getNeedToUnreserve();
    gpu.unreserve(toUnreserve, *this);
    for(std::size_t i = 0; i < memoryBlocks.size(); i++){
        gpu.release(memoryBlocks[i]);
    }

    memoryBlocks.clear();

}


//getNeedToUnreserve() method for GenerativeRequest and EmbeddingRequest classes
std::size_t GenerativeRequest::getNeedToUnreserve() const{
    return reservedBlocks - blockCounter; 

}

std::size_t EmbeddingRequest::getNeedToUnreserve() const{
    return 0;
}

//getNeededMemory 
std::size_t GenerativeRequest::getNeededMemory(const GPU& gpu) const{
    return((tokens.size() +maxOutputTokens - (gpu.findLongestCachedPrefix(model.getModelId(), model.getEmbeddingSize(),tokens)).matchedTokens) * model.getEmbeddingSize());
};

std::size_t EmbeddingRequest:: getNeededMemory(const GPU& gpu) const{
    return (2*model.getEmbeddingSize());

}

//getMaxNeededMemory
std::size_t GenerativeRequest:: getMaxNeededMemory() const{
    return ((tokens.size() + maxOutputTokens) * model.getEmbeddingSize());
}

std::size_t EmbeddingRequest:: getMaxNeededMemory() const{
    return (2*model.getEmbeddingSize());
}


//getCachableToken

std::vector<std::string> GenerativeRequest:: getCacheableTokens() const{
    return tokens;
}

std::vector<std::string> EmbeddingRequest:: getCacheableTokens() const{
    return std::vector<std::string>();
}



//allocateOneBlock

void GenerativeRequest:: allocateOneBlock(GPU& gpu, std::size_t slot) const{
    gpu.addMemoryBlock(*this,slot);
};

void EmbeddingRequest:: allocateOneBlock(GPU& gpu, std::size_t slot) const{
    gpu.addMemoryBlock(*this,slot);
};

//getOutPut
std::string GenerativeRequest:: getOutput() const{
    std:: string ret = "";
    if (scriptedTokens.size() > 0 && maxOutputTokens > 0)
    {
        ret = scriptedTokens[0];
       for (size_t i = 1; i < maxOutputTokens  && i < scriptedTokens.size(); i++)
       {
        ret = ret + " " + scriptedTokens[i];
        }
        
    }
    
    
    return ret;
    
}

std::string EmbeddingRequest:: getOutput() const{
    
    std:: string ret = "";
    if(outputEmbeddings.size() == 0){
        return ret;
    }
    ret ="[" + std::to_string(outputEmbeddings[0] / static_cast<int>(tokens.size()));
    for (size_t i = 1; i < outputEmbeddings.size(); i++)
    {
        ret = ret + ", " + std::to_string(outputEmbeddings[i] / static_cast<int>(tokens.size()));
    }
    ret = ret + "]";
    return (ret); 
    
}
// Copy-with-model constructors (matching member declaration order in Request.h)
Request::Request(const Request& other, const Model& model)
    : id(other.id),
      model(model),
      admissionCycle(other.admissionCycle),
      tokens(other.tokens),
      memoryBlocks(other.memoryBlocks),
      reservedBlocks(other.reservedBlocks),
      finishedTokenCounter(other.finishedTokenCounter) {}

GenerativeRequest::GenerativeRequest(const GenerativeRequest& other, const Model& model)
    : Request(other, model),
      scriptedOutput(other.scriptedOutput),
      maxOutputTokens(other.maxOutputTokens),
      outputTokenCounter(other.outputTokenCounter),
      scriptedTokens(other.scriptedTokens),
      currentTokens(other.currentTokens),
      blockCounter(other.blockCounter) {}

EmbeddingRequest::EmbeddingRequest(const EmbeddingRequest& other, const Model& model)
    : Request(other, model),
      outputEmbeddings(other.outputEmbeddings) {}

// clone() methods
Request* GenerativeRequest::clone(const Model& model) const {
    return new GenerativeRequest(*this, model);
}

Request* EmbeddingRequest::clone(const Model& model) const {
    return new EmbeddingRequest(*this, model);
}

int Request::getId() const {
    return id;
}

const Model& Request::getModel() const {
    return model;
}

int Request::getAdmissionCycle() const {
    return admissionCycle;
}