#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#ifndef PROSPERO_IMAGE_SIZE
#define PROSPERO_IMAGE_SIZE 1024
#endif

#ifndef PROSPERO_TARGET_TILE_SIZE
#define PROSPERO_TARGET_TILE_SIZE 16
#endif

constexpr int kImageSize = PROSPERO_IMAGE_SIZE;
constexpr int kTargetTileSize = PROSPERO_TARGET_TILE_SIZE;
constexpr int kBatchSize = kTargetTileSize * kTargetTileSize;

enum class OpCode { VarX, VarY, Const, Add, Sub, Mul, Max, Min, Neg, Square, Sqrt };

struct Instruction
{
    float constant = 0.0f;
    int input0 = -1;
    int input1 = -1;
    OpCode op = OpCode::Const;
};

struct ExportedTileProgram
{
    int px = 0;
    int py = 0;
    bool is_constant = false;
    std::uint8_t constant_value = 0;
    std::vector<Instruction> instructions;
};

std::vector<Instruction> parse_instructions(std::string_view text);
int prepare_thread_locals(std::size_t instruction_count);
void render_image(const std::vector<Instruction>& instructions, std::vector<std::uint8_t>& image);
std::vector<ExportedTileProgram> extract_tile_programs(const std::vector<Instruction>& instructions);
