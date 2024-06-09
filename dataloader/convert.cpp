// clang-format off

#include "dataloader.hpp"
#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <unordered_map>

std::unordered_map<char, u8> CHAR_TO_PIECE_TYPE = {
    {'P', 0}, {'N', 1}, {'B', 2}, {'R', 3}, {'Q', 4}, {'K', 5},
    {'p', 0}, {'n', 1}, {'b', 2}, {'r', 3}, {'q', 4}, {'k', 5},
};

void trim(std::string &str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    size_t last = str.find_last_not_of(" \t\n\r");
    
    if (first == std::string::npos) // The string is empty or contains only whitespace characters
    {
        str = "";
        return;
    }
    
    str = str.substr(first, (last - first + 1));
}

std::vector<std::string> splitString(std::string &str, char delimiter)
{
    trim(str);
    if (str == "") return std::vector<std::string>{};

    std::vector<std::string> strSplit;
    std::stringstream ss(str);
    std::string token;

    while (getline(ss, token, delimiter))
    {
        trim(token);
        strSplit.push_back(token);
    }

    return strSplit;
}

int charToInt(char myChar) { return myChar - '0'; }

int main(int argc, char* argv[]) {    
    assert(argc == 2);
    std::string fileName = (std::string)argv[1];

    std::cout << fileName << std::endl;
    assert(fileName.substr(fileName.length() - 4) == ".txt");

    std::ifstream inputFile(fileName);
    assert(inputFile && inputFile.is_open());

    fileName.replace(fileName.size() - 4, 4, ".bin");
    std::cout << fileName << std::endl;

    std::ofstream outputFile(fileName, std::ios::binary);
    assert(outputFile && outputFile.is_open());

    std::string line;
    u64 entriesWritten = 0;

    DataEntry dataEntry;
    dataEntry.extra = {};

    while (std::getline(inputFile, line))
    {
        std::stringstream iss(line);

        std::vector<std::string> tokens = splitString(line, '|');
        assert(tokens.size() == 3);

        dataEntry.occupancy = 0;
        dataEntry.pieces = 0;

        std::string fen = tokens[0];
        std::vector<std::string> fenSplit = splitString(fen, ' ');

        std::vector<std::string> fenRows = splitString(fenSplit[0], '/');
        assert(fenRows.size() == 8);

        for (int row = 0; row < 8; row++) 
        {
            int file = 0;
            for (char &myChar : fenRows[7 - row])
            {
                if (isdigit(myChar)) {
                    file += charToInt(myChar);
                    continue;
                }

                bool isWhitePiece = isupper(myChar);

                // Assert map contains the key
                assert(CHAR_TO_PIECE_TYPE.find(myChar) != CHAR_TO_PIECE_TYPE.end());

                u128 piece = (u8)isWhitePiece | (CHAR_TO_PIECE_TYPE[myChar] << 1);
                dataEntry.pieces |= piece << u128(std::popcount(dataEntry.occupancy) * 4);

                u64 square = row * 8 + file;
                dataEntry.occupancy |= 1ULL << square;
                
                file++;
            }
        }

        assert(std::popcount(dataEntry.occupancy) >= 2 && std::popcount(dataEntry.occupancy) <= 32);

        std::string stm = fenSplit[1];
        assert(stm == "w" || stm == "b");
        dataEntry.whiteToMove = stm == "w";

        dataEntry.stmScore = std::stoll(tokens[1]);
        if (stm == "b") dataEntry.stmScore *= -1;

        int whiteResult= std::stof(tokens[2]) * 2.0;
        assert(whiteResult == 0 || whiteResult == 1 || whiteResult == 2);
        dataEntry.stmResult = stm == "w" ? whiteResult : 2 - whiteResult;

        outputFile.write((char*)(&dataEntry), sizeof(DataEntry));
        entriesWritten++;

        if (entriesWritten % 100'000'000ULL == 0) 
            std::cout << "Entries written: " << entriesWritten << std::endl;
    }

    std::cout << "Total entries written: " << entriesWritten << std::endl;

    return 0;
}