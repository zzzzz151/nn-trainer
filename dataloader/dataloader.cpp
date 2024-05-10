#include "misc.hpp"

const int BATCH_SIZE = 16384;
const std::string DATA_FILE_NAME = "dataloader/sample32M.bf";

u64 NUM_DATA_ENTRIES = 0;
u64 NUM_BATCHES = 0;

extern "C" void init()
{
    // open file in binary mode and at the end
    std::ifstream file(DATA_FILE_NAME, std::ios::binary | std::ios::ate);
    assert(file.is_open());

    u64 bytes = file.tellg();

    NUM_DATA_ENTRIES = bytes / sizeof(DataEntry);
    std::cout << "Data entries: " << NUM_DATA_ENTRIES << std::endl;

    assert(NUM_DATA_ENTRIES % BATCH_SIZE == 0);

    NUM_BATCHES = NUM_DATA_ENTRIES / BATCH_SIZE;
    std::cout << "Batches: " << NUM_BATCHES << std::endl;
}

Batch batch = Batch(BATCH_SIZE);
u64 nextBatchIdx = 0;

extern "C" Batch *batchPtr()
{
    return &batch;
}

extern "C" void loadNextBatch()
{
    std::ifstream file(DATA_FILE_NAME, std::ios::binary);
    assert(file.is_open());
    file.seekg(nextBatchIdx * sizeof(DataEntry) * BATCH_SIZE, std::ios::beg);

    batch.numActiveFeatures = 0;
    DataEntry dataEntry;

    constexpr std::array<i16, 14> PIECE_TO_FEATURE = {
        0, 64, 128, 192, 256, 320, 0, 0, 384, 448, 512, 576, 640, 704
    };

    for (int entryIdx = 0; entryIdx < BATCH_SIZE; entryIdx++)
    {
        file.read((char *)(&dataEntry), sizeof(DataEntry));
        auto startNumActiveFeatures = batch.numActiveFeatures;

        while (dataEntry.occ > 0)
        {
            i16 square = poplsb(dataEntry.occ);
            i16 piece = dataEntry.pieces & 0b1111;

            // Assert no out of bounds array access
            assert(batch.numActiveFeatures * 2 + 1 < BATCH_SIZE * 32 * 2);

            batch.stmFeatures[batch.numActiveFeatures * 2] = batch.nstmFeatures[batch.numActiveFeatures * 2] = entryIdx;

            batch.stmFeatures[batch.numActiveFeatures * 2 + 1] = PIECE_TO_FEATURE[piece] + square;
            batch.nstmFeatures[batch.numActiveFeatures * 2 + 1] = PIECE_TO_FEATURE[piece ^ 8] + (square ^ 56);

            batch.numActiveFeatures++;
            dataEntry.pieces >>= 4;
        }

        // Sort the position's features in ascending order to ensure coalesceness
        for (auto i = startNumActiveFeatures * 2 + 1; i < batch.numActiveFeatures * 2; i += 2)
            for (auto j = i + 2; j < batch.numActiveFeatures * 2; j += 2)
            {
                if (batch.stmFeatures[j] < batch.stmFeatures[i])
                    std::swap(batch.stmFeatures[i], batch.stmFeatures[j]);

                if (batch.nstmFeatures[j] < batch.nstmFeatures[i])
                    std::swap(batch.nstmFeatures[i], batch.nstmFeatures[j]);
            }

        batch.stmScores[entryIdx] = dataEntry.stmScore;
        batch.stmResults[entryIdx] = (float)dataEntry.stmResult / 2.0;
    }

    nextBatchIdx++;
    if (nextBatchIdx >= NUM_BATCHES) nextBatchIdx = 0;
}

int main()
{
    return 0;
}