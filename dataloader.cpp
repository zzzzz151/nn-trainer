// clang-format off

#include "dataloader.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <thread>

// These 9 constants are set in init(), which is called in train.py
std::string DATA_FILE_NAME = "";
u64 DATA_FILE_BYTES = 0;
u64 NUM_DATA_ENTRIES = 0;
u64 BATCH_SIZE = 0;
u64 NUM_THREADS = 0;
std::array<int, 65> INPUT_BUCKETS_MAP = {};
int NUM_INPUT_BUCKETS = 0;
bool FACTORIZER = false;
int NUM_OUTPUT_BUCKETS = 0;

std::vector<Batch> gBatches; // NUM_THREADS batches
u64 gNextBatchIdx = 0; // 0 to NUM_THREADS-1
u64 gDataFilePos = 0;

extern "C" API void init(
    const char* dataFileName, 
    u32 batchSize, 
    u8 numThreads, 
    std::array<int, 65> &inputBucketsMap,
    bool factorizer,
    u8 numOutputBuckets)
{
    DATA_FILE_NAME = (std::string)dataFileName;
    BATCH_SIZE = batchSize;
    NUM_THREADS = numThreads;
    INPUT_BUCKETS_MAP = inputBucketsMap;
    NUM_INPUT_BUCKETS = 1 + *std::max_element(INPUT_BUCKETS_MAP.begin(), INPUT_BUCKETS_MAP.end());
    FACTORIZER = factorizer;
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
    batch->numActiveFeatures = 0;

    auto feature = [](int pieceColor, int pieceType, int square, int kingSquare, int enemyQueenSquare) -> int 
    {
        // HM (Horizontal mirroring)
        // If king on right side of board, mirror this piece horizontally
        // (along vertical axis)
        if (kingSquare % 8 > 3) {
            square ^= 7;
            
            if (enemyQueenSquare != 64) 
                enemyQueenSquare ^= 7;
        }

        return INPUT_BUCKETS_MAP[enemyQueenSquare] * 768 + pieceColor * 384 + pieceType * 64 + square;
    };

    for (u64 entryIdx = 0; entryIdx < BATCH_SIZE; entryIdx++)
    {
        dataFile.read((char*)(&dataEntry), sizeof(DataEntry));

        batch->isWhiteStm[entryIdx] = dataEntry.whiteToMove;
        batch->stmScores[entryIdx] = dataEntry.stmScore;
        batch->stmResults[entryIdx] = float(dataEntry.stmResult + 1) / 2.0;
        batch->outputBuckets[entryIdx] = (std::popcount(dataEntry.occupancy) - 1) / (32 / NUM_OUTPUT_BUCKETS);

        while (dataEntry.occupancy > 0)
        {
            int square = poplsb(dataEntry.occupancy);
            int pieceColor = dataEntry.pieces & 0b1;
            int pieceType = (dataEntry.pieces & 0b1110) >> 1;

            int idx = batch->numActiveFeatures * 2;
            
            batch->activeFeaturesWhiteStm[idx] = batch->activeFeaturesBlackStm[idx] = entryIdx;

            batch->activeFeaturesWhiteStm[idx + 1] 
                = feature(pieceColor, pieceType, square, dataEntry.whiteKingSquare, dataEntry.blackQueenSquare);

            batch->activeFeaturesBlackStm[idx + 1] 
                = feature(pieceColor, pieceType, square, dataEntry.blackKingSquare, dataEntry.whiteQueenSquare);

            if (FACTORIZER) {
                idx += 2;

                batch->activeFeaturesWhiteStm[idx] = batch->activeFeaturesBlackStm[idx] = entryIdx;

                batch->activeFeaturesWhiteStm[idx + 1] 
                    = batch->activeFeaturesWhiteStm[idx - 1] % 768 + 768 * NUM_INPUT_BUCKETS;

                batch->activeFeaturesBlackStm[idx + 1] 
                    = batch->activeFeaturesBlackStm[idx - 1] % 768 + 768 * NUM_INPUT_BUCKETS;
            }

            batch->numActiveFeatures += 1 + FACTORIZER;
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