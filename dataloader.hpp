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

constexpr int WHITE = 0, BLACK = 1;
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
    // lsb is piece color, other 3 bits is piece type
    u128 pieces;

    u8 whiteKingSquare, 
       blackKingSquare,
       whiteQueenSquare,
       blackQueenSquare;

    i16 stmScore;
    i8 stmResult; // -1, 0, 1

} __attribute__((packed));

static_assert(sizeof(DataEntry) == 32); // 32 bytes

struct Batch {
    public:

    u32 numActiveFeatures= 0;

    i16* activeFeaturesWhiteStm;    
    i16* activeFeaturesBlackStm;

    bool* isWhiteStm;
    
    float* stmScores;
    float* stmResults;

    u8* outputBuckets;

    Batch(u32 batchSize)
    {
        // Indices of active features
        // array size is * 2 because the indices are (positionIndex, featureIndex)
        // aka a (numActiveFeatures, 2) matrix
        activeFeaturesWhiteStm = new i16[batchSize * 64 * 2];
        activeFeaturesBlackStm = new i16[batchSize * 64 * 2];

        isWhiteStm = new bool[batchSize];
        stmScores = new float[batchSize];
        stmResults = new float[batchSize];
        outputBuckets = new u8[batchSize];
    }
};
