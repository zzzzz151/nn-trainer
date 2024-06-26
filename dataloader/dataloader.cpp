// clang-format off

#include "dataloader.hpp"
#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <thread>

// These 5 constants are set in init(), which is called in train.py
std::string DATA_FILE_NAME;
u64 DATA_FILE_BYTES = 0;
u64 NUM_DATA_ENTRIES = 0;
u64 BATCH_SIZE = 0;
u64 NUM_THREADS = 0;

std::vector<Batch> gBatches; // NUM_THREADS batches
u64 gNextBatchIdx = 0; // 0 to NUM_THREADS-1
u64 gDataFilePos = 0;

extern "C" API void init(const char* dataFileName, u64 batchSize, u64 numThreads)
{
    DATA_FILE_NAME = (std::string)dataFileName;
    BATCH_SIZE = batchSize;
    NUM_THREADS = numThreads;

    // open file in binary mode and at the end
    std::ifstream dataFile(DATA_FILE_NAME, std::ios::binary | std::ios::ate);
    assert(dataFile.is_open());

    DATA_FILE_BYTES = dataFile.tellg();
    NUM_DATA_ENTRIES = DATA_FILE_BYTES / sizeof(DataEntry);

    assert(NUM_DATA_ENTRIES % BATCH_SIZE == 0);

    for (u64 i = 0; i < NUM_THREADS; i++)
        gBatches.push_back(Batch(BATCH_SIZE));
}

void loadBatch(u64 threadId) {

    // Open file at correct position

    u64 dataFilePos = gDataFilePos + sizeof(DataEntry) * BATCH_SIZE * threadId;

    if (dataFilePos >= DATA_FILE_BYTES) 
        dataFilePos -= DATA_FILE_BYTES;

    std::ifstream dataFile(DATA_FILE_NAME, std::ios::binary);
    assert(dataFile.is_open());
    dataFile.seekg(dataFilePos, std::ios::beg);

    // Fill the batch gBatches[threadId]

    DataEntry dataEntry;
    Batch *batch = &gBatches[threadId];
    batch->numActiveFeatures = 0;

    for (u64 entryIdx = 0; entryIdx < BATCH_SIZE; entryIdx++)
    {
        dataFile.read((char*)(&dataEntry), sizeof(DataEntry));

        batch->isWhiteStm[entryIdx] = dataEntry.whiteToMove;
        batch->stmScores[entryIdx] = dataEntry.stmScore;
        batch->stmResults[entryIdx] = (float)dataEntry.stmResult / 2.0;

        while (dataEntry.occupancy > 0)
        {
            i16 square = poplsb(dataEntry.occupancy);
            bool isWhitePiece = dataEntry.pieces & 0b1;
            i16 pieceType = (dataEntry.pieces & 0b1110) >> 1;

            batch->activeFeatures[batch->numActiveFeatures * 2] = entryIdx;

            batch->activeFeatures[batch->numActiveFeatures * 2 + 1] 
                = !isWhitePiece * 384 + pieceType * 64 + square;

            batch->numActiveFeatures++;
            dataEntry.pieces >>= 4;
        }             
    }
}

extern "C" API u64 numDataEntries() {
    return NUM_DATA_ENTRIES;
}

extern "C" API Batch* nextBatch()
{
    if (gNextBatchIdx == 0 || gNextBatchIdx >= NUM_THREADS)
    {
        std::vector<std::thread> threads;
        threads.reserve(NUM_THREADS);

        for (u64 threadId = 0; threadId < NUM_THREADS; threadId++)
            threads.push_back(std::thread(loadBatch, threadId));

        // Wait for the threads
        for (auto &thread : threads) 
            thread.join();

        gDataFilePos += sizeof(DataEntry) * BATCH_SIZE * NUM_THREADS;

        if (gDataFilePos >= DATA_FILE_BYTES) 
            gDataFilePos -= DATA_FILE_BYTES;

        gNextBatchIdx = 0;
    }

    return &gBatches[gNextBatchIdx++];
}

int main()
{
    return 0;
}