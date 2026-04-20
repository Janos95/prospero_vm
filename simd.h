#pragma once

#if defined(__wasm_simd128__)

#include <wasm_simd128.h>

namespace simd
{
struct mask4
{
    v128_t value;

    mask4() = default;
    explicit mask4(v128_t lanes) : value(lanes) {}
};

inline mask4 operator&(mask4 a, mask4 b)
{
    return mask4{wasm_v128_and(a.value, b.value)};
}

struct v4f
{
    static constexpr int width = 4;

    v128_t value;

    v4f() = default;
    explicit v4f(v128_t lanes) : value(lanes) {}

    static v4f load(const float* ptr)
    {
        return v4f{wasm_v128_load(ptr)};
    }

    void store(float* ptr) const
    {
        wasm_v128_store(ptr, value);
    }

    static v4f splat(float scalar)
    {
        return v4f{wasm_f32x4_splat(scalar)};
    }

    static v4f make(float x0, float x1, float x2, float x3)
    {
        return v4f{wasm_f32x4_make(x0, x1, x2, x3)};
    }
};

inline v4f operator+(v4f a, v4f b)
{
    return v4f{wasm_f32x4_add(a.value, b.value)};
}

inline v4f operator-(v4f a, v4f b)
{
    return v4f{wasm_f32x4_sub(a.value, b.value)};
}

inline v4f operator*(v4f a, v4f b)
{
    return v4f{wasm_f32x4_mul(a.value, b.value)};
}

inline v4f operator-(v4f value)
{
    return v4f{wasm_f32x4_neg(value.value)};
}

inline v4f min(v4f a, v4f b)
{
    return v4f{wasm_f32x4_min(a.value, b.value)};
}

inline v4f max(v4f a, v4f b)
{
    return v4f{wasm_f32x4_max(a.value, b.value)};
}

inline v4f sqrt(v4f value)
{
    return v4f{wasm_f32x4_sqrt(value.value)};
}

inline mask4 operator<(v4f a, v4f b)
{
    return mask4{wasm_f32x4_lt(a.value, b.value)};
}

inline mask4 operator<=(v4f a, v4f b)
{
    return mask4{wasm_f32x4_le(a.value, b.value)};
}

inline mask4 operator>=(v4f a, v4f b)
{
    return mask4{wasm_f32x4_ge(a.value, b.value)};
}

inline int mask_bits(mask4 mask)
{
    return wasm_i32x4_bitmask(mask.value);
}

inline v4f select(mask4 mask, v4f on_true, v4f on_false)
{
    return v4f{wasm_v128_bitselect(on_true.value, on_false.value, mask.value)};
}
} // namespace simd

#elif defined(__ARM_NEON)

#include <arm_neon.h>

#if defined(__aarch64__)
#define SIMD_NEON_MIN vminnmq_f32
#define SIMD_NEON_MAX vmaxnmq_f32
#else
#define SIMD_NEON_MIN vminq_f32
#define SIMD_NEON_MAX vmaxq_f32
#endif

namespace simd
{
struct mask4
{
    uint32x4_t value;

    mask4() = default;
    explicit mask4(uint32x4_t lanes) : value(lanes) {}
};

inline mask4 operator&(mask4 a, mask4 b)
{
    return mask4{vandq_u32(a.value, b.value)};
}

struct v4f
{
    static constexpr int width = 4;

    float32x4_t value;

    v4f() = default;
    explicit v4f(float32x4_t lanes) : value(lanes) {}

    static v4f load(const float* ptr)
    {
        return v4f{vld1q_f32(ptr)};
    }

    void store(float* ptr) const
    {
        vst1q_f32(ptr, value);
    }

    static v4f splat(float scalar)
    {
        return v4f{vdupq_n_f32(scalar)};
    }

    static v4f make(float x0, float x1, float x2, float x3)
    {
        const float lanes[4] = {x0, x1, x2, x3};
        return v4f{vld1q_f32(lanes)};
    }
};

inline v4f operator+(v4f a, v4f b)
{
    return v4f{vaddq_f32(a.value, b.value)};
}

inline v4f operator-(v4f a, v4f b)
{
    return v4f{vsubq_f32(a.value, b.value)};
}

inline v4f operator*(v4f a, v4f b)
{
    return v4f{vmulq_f32(a.value, b.value)};
}

inline v4f operator-(v4f value)
{
    return v4f{vnegq_f32(value.value)};
}

inline v4f min(v4f a, v4f b)
{
    return v4f{SIMD_NEON_MIN(a.value, b.value)};
}

inline v4f max(v4f a, v4f b)
{
    return v4f{SIMD_NEON_MAX(a.value, b.value)};
}

inline v4f sqrt(v4f value)
{
    return v4f{vsqrtq_f32(value.value)};
}

inline mask4 operator<(v4f a, v4f b)
{
    return mask4{vcltq_f32(a.value, b.value)};
}

inline mask4 operator<=(v4f a, v4f b)
{
    return mask4{vcleq_f32(a.value, b.value)};
}

inline mask4 operator>=(v4f a, v4f b)
{
    return mask4{vcgeq_f32(a.value, b.value)};
}

inline int mask_bits(mask4 mask)
{
    uint32_t lanes[4];
    vst1q_u32(lanes, mask.value);
    return (lanes[0] ? 1 : 0) | (lanes[1] ? 2 : 0) | (lanes[2] ? 4 : 0) | (lanes[3] ? 8 : 0);
}

inline v4f select(mask4 mask, v4f on_true, v4f on_false)
{
    return v4f{vbslq_f32(mask.value, on_true.value, on_false.value)};
}
} // namespace simd

#undef SIMD_NEON_MIN
#undef SIMD_NEON_MAX

#elif defined(__SSE__)

#include <xmmintrin.h>

namespace simd
{
struct mask4
{
    __m128 value;

    mask4() = default;
    explicit mask4(__m128 lanes) : value(lanes) {}
};

inline mask4 operator&(mask4 a, mask4 b)
{
    return mask4{_mm_and_ps(a.value, b.value)};
}

struct v4f
{
    static constexpr int width = 4;

    __m128 value;

    v4f() = default;
    explicit v4f(__m128 lanes) : value(lanes) {}

    static v4f load(const float* ptr)
    {
        return v4f{_mm_loadu_ps(ptr)};
    }

    void store(float* ptr) const
    {
        _mm_storeu_ps(ptr, value);
    }

    static v4f splat(float scalar)
    {
        return v4f{_mm_set1_ps(scalar)};
    }

    static v4f make(float x0, float x1, float x2, float x3)
    {
        return v4f{_mm_setr_ps(x0, x1, x2, x3)};
    }
};

inline v4f operator+(v4f a, v4f b)
{
    return v4f{_mm_add_ps(a.value, b.value)};
}

inline v4f operator-(v4f a, v4f b)
{
    return v4f{_mm_sub_ps(a.value, b.value)};
}

inline v4f operator*(v4f a, v4f b)
{
    return v4f{_mm_mul_ps(a.value, b.value)};
}

inline v4f operator-(v4f value)
{
    return v4f{_mm_sub_ps(_mm_setzero_ps(), value.value)};
}

inline v4f min(v4f a, v4f b)
{
    return v4f{_mm_min_ps(a.value, b.value)};
}

inline v4f max(v4f a, v4f b)
{
    return v4f{_mm_max_ps(a.value, b.value)};
}

inline v4f sqrt(v4f value)
{
    return v4f{_mm_sqrt_ps(value.value)};
}

inline mask4 operator<(v4f a, v4f b)
{
    return mask4{_mm_cmplt_ps(a.value, b.value)};
}

inline mask4 operator<=(v4f a, v4f b)
{
    return mask4{_mm_cmple_ps(a.value, b.value)};
}

inline mask4 operator>=(v4f a, v4f b)
{
    return mask4{_mm_cmpge_ps(a.value, b.value)};
}

inline int mask_bits(mask4 mask)
{
    return _mm_movemask_ps(mask.value);
}

inline v4f select(mask4 mask, v4f on_true, v4f on_false)
{
    return v4f{
        _mm_or_ps(_mm_and_ps(mask.value, on_true.value),
                  _mm_andnot_ps(mask.value, on_false.value))};
}
} // namespace simd

#else

#error "simd.h requires wasm SIMD, ARM NEON, or SSE support"

#endif
