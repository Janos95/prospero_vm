#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

constexpr int kImageSize = 1024;
constexpr int kTargetTileSize = 16;
constexpr int kBatchSize = kTargetTileSize * kTargetTileSize;

enum class OpCode { VarX, VarY, Const, Add, Sub, Mul, Max, Min, Neg, Square, Sqrt };

struct Instruction
{
    float constant = 0.0f;
    int input0 = -1;
    int input1 = -1;
    OpCode op = OpCode::Const;
};

std::vector<Instruction> parse_instructions(std::string_view text);
int prepare_thread_locals(std::size_t instruction_count);
void render_image(const std::vector<Instruction>& instructions, std::vector<float>& image);
