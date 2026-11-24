#include <fstream>
#include <iostream>>
#include <stdexcept>
#include "json.hpp"

#include "Server.h"
#include "Model.h"
#include "SchedulingPolicy.h"
#include "GPU.h"


using namespace std;
using json = nlohmann::json;

Config Server::readConfig(string configPath) const{
    std::ifstream input(configPath);
    json config;
    input >> config;
    // All the GPUs have the same block size.
    size_t blockSize = config["block_size"].get<size_t>();
    vector<GPUConfig> gpuConfigs;
    for(auto& gpu: config["gpus"]){
        size_t memoryBytes = gpu["memory_bytes"].get<size_t>();
        GPUConfig gpuConfig{memoryBytes};
        gpuConfigs.push_back(gpuConfig);
    }
    vector<ModelConfig> modelConfigs;
    for(auto& model: config["models"]){
        string id = model["id"].get<string>();
        size_t embeddingSize = model["embedding_size"].get<size_t>();
        ModelConfig modelConfig{id, embeddingSize};
        modelConfigs.push_back(modelConfig);
    }
    vector<RequestConfig> requestConfigs;
    for(auto& request: config["requests"]){
        size_t id = request["id"].get<size_t>();
        string typeStr = request["type"].get<string>();
        if(typeStr != "generative" && typeStr != "text_embedder"){
            throw invalid_argument("invalid request type");
        }
        RequestType type = typeStr=="generative"?RequestType::Generative:RequestType::TextEmbedder;
        string modelId = request["model"].get<string>();
        int admissionCycle = request["cycle"].get<int>();
        string prompt = request["prompt"].get<string>();
        string scriptedOutput = "";
        if (request.count("scripted_output") > 0) {
            scriptedOutput = request["scripted_output"].get<string>();
        }
        size_t maxOutputTokens = 0;
        if (request.count("max_output_tokens") > 0) {
            maxOutputTokens = request["max_output_tokens"].get<size_t>();
        }
        RequestConfig reqConfig{id, type, modelId, admissionCycle, prompt, scriptedOutput, maxOutputTokens};
        requestConfigs.push_back(reqConfig);
    }

    if (config.count("scheduling_policy") == 0) {
        throw invalid_argument("missing scheduling policy");
    }
    string policy = config["scheduling_policy"].get<string>();

    return Config{blockSize, gpuConfigs, modelConfigs, requestConfigs, policy};
}

Server::Server(string configPath): gpus(),
      models(),
      nonSubmittedRequests(),
      waitingRequests(),
      activeRequests(),
      policy(nullptr),
      currentCycle(0),
      requestGPU()
      {
    Config config = readConfig(configPath);
    for(GPUConfig gpuConfig: config.gpus){
        gpus.push_back(GPU(gpuConfig.memoryBytes/config.blockSize, config.blockSize));
    }

    for(ModelConfig modelConfig: config.models){
        models.push_back(new Model(modelConfig.id, modelConfig.embeddingSize));
    }

    for(RequestConfig requestConfig: config.requests){
        Model* model = nullptr;
        for(Model* m: models){
            if(m->getId() == requestConfig.modelId){
                model = m;
                break;
            }
        }
        if(model == nullptr){
            throw invalid_argument("request refers to non-existent model");
        }
        Request* request;
        if(requestConfig.type == RequestType::Generative){
            request = new GenerativeRequest(requestConfig.id, *model, requestConfig.admissionCycle, requestConfig.prompt, requestConfig.scriptedOutput, requestConfig.maxOutputTokens);
        } else {
            request = new EmbeddingRequest(requestConfig.id, *model, requestConfig.admissionCycle, requestConfig.prompt);
        }
        nonSubmittedRequests.push_back(request);
    }
    if (config.policy == "arrivalOrder") {
    policy = new ArrivalOrderPolicy();
    } else if (config.policy == "shortestJob") {
        policy = new ShortestJobPolicy();
    } else if (config.policy == "longestPrefix") {
        policy = new LongestPrefixPolicy();
    }
    
}

bool Server::step(){
    //check if there are no more requests to process
    if (nonSubmittedRequests.empty() && waitingRequests.empty() && activeRequests.empty()) {
        return false;
    }

    //move relevant requests to waitingrequests then remove them
    std::vector<size_t> toKeep;
    std::vector<size_t> toRemove;
    std::vector<Request*> copy; 
    for(size_t i = 0; i < nonSubmittedRequests.size(); i++){
        Request* request = nonSubmittedRequests[i];
        if(request->getAdmissionCycle() == currentCycle){
            
            std::size_t neededMemory = request->getMaxNeededMemory();
            bool found = true;
            for(size_t j = 0; j < gpus.size() && found; j++){
                if (gpus[j].getBlocksForMemory(neededMemory) <= gpus[j].getBlocksNum())
                {
                    found = false;
                }
        
            }
            if(!found){
                waitingRequests.push_back(request);

            }
            else{
                std::cout << "rejected " << request->getId() << " cycle=" << currentCycle << "\n";
                delete request;
            }
        }
        else{
            copy.push_back(request);
        }
    }
    
    nonSubmittedRequests = copy; 
    

    

    //sorting waiting requests
    std::vector<const Request*> constWaiting(waitingRequests.begin(), waitingRequests.end());
    std::vector<int> order = policy->choose(constWaiting, gpus);
    toRemove.clear();

    copy.clear();
    //start going by order
    for(int i: order){
        Request* request = waitingRequests[i];
        std::size_t neededMemory = request->getMaxNeededMemory();
        int selectedGPUIndex = -1;

        //search for the first GPU thats fits
        for(size_t j = 0; j < gpus.size() && selectedGPUIndex == -1; j++){
            PrefixMatch matchTemp = gpus[j].findLongestCachedPrefix(request->getModel().getId(), request->getModel().getEmbeddingSize(), request->getCacheableTokens());
            if(gpus[j].availableSlots() >= gpus[j].getBlocksForMemory(request->getNeededMemory(gpus[j])) + gpus[j].countIdleBlocks(matchTemp.blockIds)){
                selectedGPUIndex = j;
            }
        }

        
        if(selectedGPUIndex != -1){
            //scanning the rest GPU and choose according to the assigment
            PrefixMatch matchSelected = gpus[selectedGPUIndex].findLongestCachedPrefix(request->getModel().getId(), request->getModel().getEmbeddingSize(), request->getCacheableTokens());
            for(size_t j = selectedGPUIndex + 1; j < gpus.size(); j++){
                PrefixMatch matchTemp = gpus[j].findLongestCachedPrefix(request->getModel().getId(), request->getModel().getEmbeddingSize(), request->getCacheableTokens());
                if(gpus[j].availableSlots() >= gpus[j].getBlocksForMemory(request->getNeededMemory(gpus[j])) + gpus[j].countIdleBlocks(matchTemp.blockIds)){
                    if(matchTemp.matchedTokens > matchSelected.matchedTokens){
                        selectedGPUIndex = j;
                        matchSelected = matchTemp;
                    }
                    else if(matchTemp.matchedTokens == matchSelected.matchedTokens){
                        if(gpus[j].getBlocksForMemory(request->getNeededMemory(gpus[j])) + gpus[j].countIdleBlocks(matchTemp.blockIds) < gpus[selectedGPUIndex].getBlocksForMemory(neededMemory) -gpus[selectedGPUIndex].countIdleBlocks(matchSelected.blockIds)){
                            selectedGPUIndex = j;
                            matchSelected = matchTemp;
                        }
                    }
                }
            }

            std::vector<int> pair;
            pair.push_back(request->getId()); pair.push_back(selectedGPUIndex);
            requestGPU.push_back(pair);

            //finding gpu with longest prefix
            
            PrefixMatch longest = gpus[0].findLongestCachedPrefix(request->getModel().getId(), request->getModel().getEmbeddingSize(),request->getCacheableTokens());
            size_t longestPrefix = 0;
            for (size_t j = 0; j < gpus.size(); j++)
            {
                PrefixMatch temp = gpus[j].findLongestCachedPrefix(request->getModel().getId(), request->getModel().getEmbeddingSize(),request->getCacheableTokens());
                if (temp.matchedTokens > longest.matchedTokens) {
                    longest = temp;   
                    longestPrefix = j;
                }  
            }    
            
            if(longestPrefix != static_cast<size_t>(selectedGPUIndex)){
                gpus[selectedGPUIndex].copyMemoryFromOtherGPU(gpus[longestPrefix], longest.blockIds);
            }

            request->reserveMemory(gpus[selectedGPUIndex]);
            activeRequests.push_back(request);

        }
        else{
            copy.push_back(request);
        }
 
        

    }

    waitingRequests = copy;

    copy.clear();

    for(size_t i = 0; i < activeRequests.size(); i++){
        int GPUIndex = 0;
        bool find = true; 
        for (size_t j = 0; j < requestGPU.size() && find; j++)
        {
            if ((requestGPU[j])[0] == activeRequests[i]->getId())
            {
                GPUIndex = (requestGPU[j])[1];
                find = false;
            }
            
        }
        
        activeRequests[i]->act(gpus[GPUIndex]);
        if(activeRequests[i]->finished()){
            std::cout << "finished " << activeRequests[i]->getId()<< " cycle=" << currentCycle << " gpu=" << GPUIndex<< ": " << activeRequests[i]->getOutput() << "\n";
            activeRequests[i]->deallocateMemory(gpus[GPUIndex]);
            delete activeRequests[i];
        }
        else{
            copy.push_back(activeRequests[i]);
        }
    }
    activeRequests = copy;
    
    
    currentCycle++;
    return true;

}

void Server::serve() {
    while (step()) {
    }
    std::size_t totalHits = 0;
    std::size_t totalEvictions = 0;
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        totalHits += gpus[i].getCacheHits();
        totalEvictions += gpus[i].getEvictions();
    }
    std::cout << "summary: cycles=" << currentCycle
              << " cache_hits=" << totalHits
              << " evictions=" << totalEvictions << "\n";
}


Server::~Server() {
    for (auto model : models) {
        delete model;
    }
    for (auto request : nonSubmittedRequests) {
        delete request;
    }
    for (auto request : waitingRequests) {
        delete request;
    }
    for (auto request : activeRequests) {
        delete request;
    }
    delete policy;
}

Server::Server(const Server& other): gpus(other.gpus),
    policy(other.policy != nullptr ? other.policy->clone() : nullptr),
      currentCycle(other.currentCycle),
      requestGPU(other.requestGPU){
        for(Model* m: other.models){
           models.push_back(new Model(*m));
        }

        for (Request* R: other.nonSubmittedRequests)
        {
            for (Model* m: models)
            {
                if (m->getId() == R->getModel().getId())
                {
                    nonSubmittedRequests.push_back(R->clone(*m));
                }
                
            }
            
            
        }
        for (Request* R: other.waitingRequests)
        {
             for (Model* m: models)
            {
                if (m->getId() == R->getModel().getId())
                {
                    waitingRequests.push_back(R->clone(*m));
                }
                
            }
        }
        for (Request* R: other.activeRequests)
        {
             for (Model* m: models)
            {
                if (m->getId() == R->getModel().getId())
                {
                    activeRequests.push_back(R->clone(*m));
                }
                
            }
        }
    
}

Server& Server::operator=(const Server& other){
    if(this != &other){
        gpus = other.gpus;
        for (Model*  m: models)
        {
            delete m;
        }
        models.clear();
        for(Model* m: other.models){
           models.push_back(new Model(*m));
        }

        for (Request* R: nonSubmittedRequests)
        {
            delete R; 
        }
        nonSubmittedRequests.clear();
        for (Request* R: other.nonSubmittedRequests)
        {
            for (Model* m: models)
            {
                if (m->getId() == R->getModel().getId())
                {
                    nonSubmittedRequests.push_back(R->clone(*m));
                }
                
            }
            
            
        }


         for (Request* R: waitingRequests)
        {
            delete R; 
        }
        waitingRequests.clear();
        for (Request* R: other.waitingRequests)
        {
            for (Model* m: models)
            {
                if (m->getId() == R->getModel().getId())
                {
                    waitingRequests.push_back(R->clone(*m));
                }
                
            }
            
            
        }

         for (Request* R: activeRequests)
        {
            delete R; 
        }
        activeRequests.clear();
        for (Request* R: other.activeRequests)
        {
            for (Model* m: models)
            {
                if (m->getId() == R->getModel().getId())
                {
                    activeRequests.push_back(R->clone(*m));
                }
                
            }
            
            
        }
        delete policy;
        policy = (other.policy != nullptr) ? other.policy->clone() : nullptr;
        currentCycle = other.currentCycle; 
        requestGPU = other.requestGPU;
    }
    return *this;

}

Server::Server(Server&& other) noexcept
    : gpus(std::move(other.gpus)),
      models(std::move(other.models)),
      nonSubmittedRequests(std::move(other.nonSubmittedRequests)),
      waitingRequests(std::move(other.waitingRequests)),
      activeRequests(std::move(other.activeRequests)),
      policy(other.policy),
      currentCycle(other.currentCycle),
      requestGPU(std::move(other.requestGPU)) {
    other.policy = nullptr;
    other.currentCycle = 0;
}

Server& Server::operator=(Server&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    for (Request* R : nonSubmittedRequests) {
        delete R;
    }
    for (Request* R : waitingRequests) {
        delete R;
    }
    for (Request* R : activeRequests) {
        delete R;
    }
    for (Model* m : models) {
        delete m;
    }
    delete policy;

    gpus = std::move(other.gpus);
    models = std::move(other.models);
    nonSubmittedRequests = std::move(other.nonSubmittedRequests);
    waitingRequests = std::move(other.waitingRequests);
    activeRequests = std::move(other.activeRequests);
    policy = other.policy;
    currentCycle = other.currentCycle;
    requestGPU = std::move(other.requestGPU);

    other.policy = nullptr;
    other.currentCycle = 0;

    return *this;
}
