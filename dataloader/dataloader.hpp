// clang-format off

#pragma once

#include <array>
#include <cstdint>
#include <cassert>
#include <string>
#include <bit>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using u128 = unsigned __int128;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

constexpr int PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5;

// Needed to export functions on Windows
#ifdef _WIN32
#   define API __declspec(dllexport)
#else
#   define API
#endif

inline auto lsb(u64 bitboard) {
    return std::countr_zero(bitboard);
}

inline u8 poplsb(u64 &bitboard)
{
    auto idx = lsb(bitboard);
    bitboard &= bitboard - 1; // compiler optimizes this to _blsr_u64
    return idx;
}

struct DataEntry {
    public:
    bool whiteToMove;
    u64 occupancy;

    // 4 bits per piece for a max of 32 pieces
    // lsb is isWhitePiece
    u128 pieces;

    i16 stmScore;
    u8 stmResult; // 0 = stm lost, 1 = draw, 2 = stm won

    std::array<u8, 4> extra; // padding to ensure 32 bytes

} __attribute__((packed));

static_assert(sizeof(DataEntry) == 32); // 32 bytes

struct Batch {
    public:

    u32 batchSize = 0;

    u32 numActiveFeatures = 0;
    i16 *activeFeatures;
    
    bool *isWhiteStm;
    
    float *stmScores, *stmResults;

    Batch(u32 batchSize)
    {
        this->batchSize = batchSize;

        // Indices of active features
        // array size is * 2 because the indices are (positionIndex, featureIndex)
        // aka a (numActiveFeatures, 2) matrix
        activeFeatures = new i16[batchSize * 32 * 2];

        isWhiteStm = new bool[batchSize];
        stmScores = new float[batchSize];
        stmResults = new float[batchSize];
    }
};
