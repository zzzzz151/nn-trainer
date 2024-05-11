#include "misc.hpp"

// Needed to export functions on Windows
#ifdef _WIN32
#   define API __declspec(dllexport)
#else
#   define API
#endif

const std::string DATA_FILE_NAME = "dataloader/starzix2B.bf";
const u64 BATCH_SIZE = 16384;
const u64 NUM_THREADS = 12;

u64 DATA_FILE_BYTES = 0;
u64 NUM_DATA_ENTRIES = 0;
u64 NUM_BATCHES = 0;

std::vector<Batch> batches; // NUM_THREADS batches
u64 nextBatchIdx = 0; // 0 to NUM_THREADS-1
u64 posInFile = 0;

extern "C" API void init()
{
    // open file in binary mode and at the end
    std::ifstream file(DATA_FILE_NAME, std::ios::binary | std::ios::ate);
    assert(file.is_open());

    DATA_FILE_BYTES = file.tellg();

    NUM_DATA_ENTRIES = DATA_FILE_BYTES / sizeof(DataEntry);

    assert(NUM_DATA_ENTRIES % BATCH_SIZE == 0);

    NUM_BATCHES = NUM_DATA_ENTRIES / BATCH_SIZE;

    for (u64 i = 0; i < NUM_THREADS; i++)
        batches.push_back(Batch(BATCH_SIZE));
}

extern "C" API u64 numDataEntries() {
    return NUM_DATA_ENTRIES;
}

extern "C" API u64 batchSize() { 
    return BATCH_SIZE; 
}

extern "C" API u64 numBatches() {
    return NUM_BATCHES;
}

extern "C" API u64 numThreads() { 
    return NUM_THREADS;
}

void loadBatch(u64 threadId) {
    std::ifstream file(DATA_FILE_NAME, std::ios::binary);
    assert(file.is_open());

    u64 myPosInFile = posInFile + sizeof(DataEntry) * BATCH_SIZE * threadId;

    if (myPosInFile >= DATA_FILE_BYTES) 
        myPosInFile -= DATA_FILE_BYTES;

    file.seekg(myPosInFile, std::ios::beg);

    DataEntry dataEntry;
    Batch *batch = &batches[threadId];
    batch->numActiveFeatures = 0;

    constexpr std::array<i16, 14> PIECE_TO_FEATURE = {
        0, 64, 128, 192, 256, 320, 0, 0, 384, 448, 512, 576, 640, 704
    };

    for (u64 entryIdx = 0; entryIdx < BATCH_SIZE; entryIdx++)
    {
        file.read((char*)(&dataEntry), sizeof(DataEntry));
        auto startNumActiveFeatures = batch->numActiveFeatures;

        while (dataEntry.occ > 0)
        {
            i16 square = poplsb(dataEntry.occ);
            i16 piece = dataEntry.pieces & 0b1111;

            // Assert no out of bounds array access
            assert(batch->numActiveFeatures * 2 + 1 < BATCH_SIZE * 32 * 2);
             
            batch->stmFeatures[batch->numActiveFeatures * 2] = batch->nstmFeatures[batch->numActiveFeatures * 2] = entryIdx;

            batch->stmFeatures[batch->numActiveFeatures * 2 + 1] = PIECE_TO_FEATURE[piece] + square;
            batch->nstmFeatures[batch->numActiveFeatures * 2 + 1] = PIECE_TO_FEATURE[piece ^ 8] + (square ^ 56);

            batch->numActiveFeatures++;
            dataEntry.pieces >>= 4;
        }

        // Sort the position's features in ascending order to ensure coalesceness
        /*
        for (auto i = startNumActiveFeatures * 2 + 1; i < batch->numActiveFeatures * 2; i += 2)
            for (auto j = i + 2; j < batch->numActiveFeatures * 2; j += 2)
            {
                if (batch->stmFeatures[j] < batch->stmFeatures[i])
                    std::swap(batch->stmFeatures[i], batch->stmFeatures[j]);

                if (batch->nstmFeatures[j] < batch->nstmFeatures[i])
                    std::swap(batch->nstmFeatures[i], batch->nstmFeatures[j]);
            }
        */

        batch->stmScores[entryIdx] = dataEntry.stmScore;
        batch->stmResults[entryIdx] = (float)dataEntry.stmResult / 2.0;
    }
}

extern "C" API Batch* nextBatch()
{
    if (nextBatchIdx == 0 || nextBatchIdx >= NUM_THREADS)
    {
        std::vector<std::thread> threads;
        threads.reserve(NUM_THREADS);

        for (u64 threadId = 0; threadId < NUM_THREADS; threadId++)
            threads.push_back(std::thread(loadBatch, threadId));

        // Wait for the threads
        for (auto &thread : threads) 
            thread.join();

        posInFile += sizeof(DataEntry) * BATCH_SIZE * NUM_THREADS;

        if (posInFile >= DATA_FILE_BYTES) 
            posInFile -= DATA_FILE_BYTES;

        nextBatchIdx = 0;
    }

    return &batches[nextBatchIdx++];
}

int main()
{
    return 0;
}