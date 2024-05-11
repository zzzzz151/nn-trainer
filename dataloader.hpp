#include <iostream>
#include <string>
#include <cstdint>
#include <array>
#include <vector>
#include <fstream>
#include <cassert>
#include <thread>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using u128 = unsigned __int128;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

// Needed to export functions on Windows
#ifdef _WIN32
#   define API __declspec(dllexport)
#else
#   define API
#endif

// bulletformat
struct DataEntry {
    public:

    u64 occ; // occupancy bitboard

    // if black stm, pieces are flipped vertically and color switched
    // 4 bits per piece for a max of 32 pieces
    // msb is color (white = 1, black = 0)
    u128 pieces;

    i16 stmScore; 
    u8 stmResult; // 0 = stm lost, 1 = draw, 2 = stm won

    u8 kingSquare;
    u8 oppKingSquare;

    std::array<u8, 3> extra; // padding to ensure 32 bytes

} __attribute__((packed));

static_assert(sizeof(DataEntry) == 32); // 32 bytes

struct Batch {
    public:

    u32 batchSize = 0, numActiveFeatures = 0;

    i16 *stmFeatures, *nstmFeatures;

    float *stmScores, *stmResults;

    Batch(u32 batchSize)
    {
        this->batchSize = batchSize;

        // Indices of active features
        // array size is * 2 because the indices are (positionIndex, featureIndex)
        // aka a (num_active_features, 2) matrix
        stmFeatures = new i16[batchSize * 32 * 2];
        nstmFeatures = new i16[batchSize * 32 * 2];

        stmScores = new float[batchSize];
        stmResults = new float[batchSize];
    }

    /*
    ~Batch()
    {
        // RAII! Or use std::unique_ptr<T[]>, but remember that only raw pointers should
        // be passed through language boundaries as std::unique_ptr doesn't have stable ABI
        delete[] stmFeatures;
        delete[] nstmFeatures;
        delete[] stmScores;
        delete[] stmResults;
    }
    */
};

#if defined(__GNUC__) // GCC, Clang, ICC

    inline u8 lsb(u64 b)
    {
        assert(b);
        return u8(__builtin_ctzll(b));
    }
    inline u8 msb(u64 b)
    {
        assert(b);
        return u8(63 ^ __builtin_clzll(b));
    }

#else // Assume MSVC Windows 64

    #include <intrin.h>
    inline u8 lsb(u64 b)
    {
        unsigned long idx;
        _BitScanForward64(&idx, b);
        return (u8)idx;
    }
    inline u8 msb(u64 b)
    {
        unsigned long idx;
        _BitScanReverse64(&idx, b);
        return (u8)idx;
    }

#endif

inline u8 poplsb(u64 &mask)
{
    u8 s = lsb(mask);
    mask &= mask - 1; // compiler optimizes this to _blsr_u64
    return u8(s);
}