// clang-format off

#include "dataloader.hpp"
#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <thread>

// These 6 constants are set in init(), which is called in train.py
std::string DATA_FILE_NAME;
u64 DATA_FILE_BYTES = 0;
u64 NUM_DATA_ENTRIES = 0;
u64 BATCH_SIZE = 0;
u64 NUM_THREADS = 0;
u8 NUM_OUTPUT_BUCKETS = 0;

std::vector<Batch> gBatches; // NUM_THREADS batches
u64 gNextBatchIdx = 0; // 0 to NUM_THREADS-1
u64 gDataFilePos = 0;

extern "C" API void init(const char* dataFileName, u32 batchSize, u8 numThreads, u8 numOutputBuckets)
{
    DATA_FILE_NAME = (std::string)dataFileName;
    BATCH_SIZE = batchSize;
    NUM_THREADS = numThreads;
    NUM_OUTPUT_BUCKETS = numOutputBuckets;

    // open file in binary mode and at the end
    std::ifstream dataFile(DATA_FILE_NAME, std::ios::binary | std::ios::ate);
    assert(dataFile.is_open());

    DATA_FILE_BYTES = dataFile.tellg();
    NUM_DATA_ENTRIES = DATA_FILE_BYTES / (u64)sizeof(DataEntry);

    assert(NUM_DATA_ENTRIES % BATCH_SIZE == 0);

    for (u64 i = 0; i < NUM_THREADS; i++)
        gBatches.push_back(Batch(BATCH_SIZE));
}

void loadBatch(u64 threadId) {
    // Open file at correct position

    u64 dataFilePos = gDataFilePos + (u64)sizeof(DataEntry) * BATCH_SIZE * threadId;

    if (dataFilePos >= DATA_FILE_BYTES) 
        dataFilePos -= DATA_FILE_BYTES;

    std::ifstream dataFile(DATA_FILE_NAME, std::ios::binary);
    assert(dataFile && dataFile.is_open());
    dataFile.seekg(dataFilePos, std::ios::beg);

    // Fill the batch gBatches[threadId]

    DataEntry dataEntry;
    Batch* batch = &gBatches[threadId];
    batch->numActiveFeaturesWhiteStm = batch->numActiveFeaturesBlackStm = 0;

    for (u64 entryIdx = 0; entryIdx < BATCH_SIZE; entryIdx++)
    {
        dataFile.read((char*)(&dataEntry), sizeof(DataEntry));

        batch->isWhiteStm[entryIdx] = dataEntry.whiteToMove;
        batch->stmScores[entryIdx] = dataEntry.stmScore;
        batch->stmResults[entryIdx] = float(dataEntry.stmResult + 1) / 2.0;
        batch->outputBuckets[entryIdx] = (std::popcount(dataEntry.occupancy) - 1) / (32 / NUM_OUTPUT_BUCKETS);

        u32 &numActiveFeatures = dataEntry.whiteToMove 
                                 ? batch->numActiveFeaturesWhiteStm 
                                 : batch->numActiveFeaturesBlackStm;

        i16* activeFeatures = dataEntry.whiteToMove 
                              ? batch->activeFeaturesWhiteStm 
                              : batch->activeFeaturesBlackStm;

        // HM (Horizontal mirroring)
        // If our king is on right side of board, mirror all pieces along vertical axis
        const u8 squareXOR = dataEntry.ourKingSquare % 8 > 3 ? 7 : 0;

        while (dataEntry.occupancy > 0)
        {
            i16 square = poplsb(dataEntry.occupancy) ^ squareXOR;
            bool isWhitePiece = dataEntry.pieces & 0b1;
            i16 pieceType = (dataEntry.pieces & 0b1110) >> 1;

            activeFeatures[numActiveFeatures * 2] = entryIdx;

            activeFeatures[numActiveFeatures * 2 + 1] 
                = !isWhitePiece * 384 + pieceType * 64 + square;

            numActiveFeatures++;
            dataEntry.pieces >>= 4;
        }             
    }
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
            if (thread.joinable())
                thread.join();

        gDataFilePos += (u64)sizeof(DataEntry) * BATCH_SIZE * NUM_THREADS;

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