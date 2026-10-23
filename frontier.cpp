// Term-based ("frontier") Prospero renderer. See frontier.h for an overview.
//
// Reconstructed from the shipped WebAssembly build; the arithmetic below follows
// that build operation for operation (including WebAssembly min/max semantics for
// NaN and signed zero, and fused multiply-adds in the row evaluator), so it
// reproduces the published image exactly. Build with -ffp-contract=off.

#include "frontier.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace frontier
{
namespace
{

enum Op : int
{
    kConst = 0,
    kVarX = 1,
    kVarY = 2,
    kAdd = 3,
    kSub = 4,
    kMul = 5,
    kNeg = 6,
    kSquare = 7,
    kSqrt = 8,
    kMin = 9,
    kMax = 10,
    kSubK = 11,  // in0 - k, produced by the sqrt rewrite
    kKSub = 12,  // k - in0, produced by the sqrt rewrite
};

// WebAssembly f32.min / f32.max: NaN if either input is NaN, and -0 < +0.
inline float wmin(float a, float b)
{
    if (a != a || b != b) return NAN;
    if (a == b) return std::signbit(a) ? a : b;
    return a < b ? a : b;
}

inline float wmax(float a, float b)
{
    if (a != a || b != b) return NAN;
    if (a == b) return std::signbit(a) ? b : a;
    return a > b ? a : b;
}

struct Inst
{
    int op;
    std::int16_t in0, in1;  // instruction indices, -1 if the argument is a literal
    float k0, k1;           // literal arguments
};

constexpr int kMaxInstructions = 10000;
constexpr int kMaxTermSize = 128;  // larger terms are not rendered (none occur in prospero.vm)
constexpr int kMaxPolygon = 264;

std::vector<Inst> g_prog;
int g_last_var_x = -1;
int g_last_var_y = -1;
float g_xcoord[kImageSize];
float g_ycoord[kImageSize];
std::uint8_t g_image[kImageSize * kImageSize];
int g_term_count = 0;
int g_skipped_terms = 0;

// Generation stamps let us mark instructions as visited without clearing an
// array the size of the whole program for every frame or every term.
struct Stamps
{
    std::vector<std::uint16_t> mark;
    std::uint16_t gen = 1;

    void resize(std::size_t n)
    {
        mark.assign(n, 0);
        gen = 1;
    }
    void next()
    {
        if (++gen == 0)
        {
            std::fill(mark.begin(), mark.end(), 0);
            gen = 1;
        }
    }
    bool visit(int i)  // true the first time i is seen in this generation
    {
        if (mark[static_cast<std::size_t>(i)] == gen) return false;
        mark[static_cast<std::size_t>(i)] = gen;
        return true;
    }
};

Stamps g_frame_stamps;
Stamps g_term_stamps;
std::vector<int> g_remap;
std::vector<int> g_terms;
std::vector<int> g_frame_stack;
std::vector<int> g_term_nodes;
std::vector<std::pair<int, int>> g_term_stack;

// Local, renumbered program for the term being rendered.
struct Term
{
    int n = 0;
    int root = 0;
    bool fill_only = true;  // no min and no node mixing x and y except max: the clipped box is solid
    Inst ins[kMaxTermSize];
    std::uint8_t kind[kMaxTermSize];   // 0 constant, 1 x only, 2 y only, 3 both
    std::uint8_t sflag[kMaxTermSize];  // bit 0: in0 is a per-row scalar, bit 1: in1 is
    std::vector<int> list[4];          // live nodes by kind, evaluated per row span / per row
};

Term T;
float val[kMaxTermSize];                         // scalar values (constants, y-only per row)
bool aff[kMaxTermSize];                          // node is affine: cx*x + cy*y + c0
float acx[kMaxTermSize], acy[kMaxTermSize], ac0[kMaxTermSize];
float ilo[kMaxTermSize], ihi[kMaxTermSize];      // forward intervals
float tlo[kMaxTermSize], thi[kMaxTermSize];      // backward target intervals
bool tset[kMaxTermSize];
float rowbuf[kMaxTermSize][kImageSize + 8];      // per-node values along a row span

// Interval evaluation of the term program over [x0, x1] x [y0, y1].
void interval_eval(float x0, float x1, float y0, float y1, float& lo, float& hi)
{
    for (int c = 0; c < T.n; c++)
    {
        if (aff[c])
        {
            const float cx = acx[c], cy = acy[c], c0 = ac0[c];
            const float h = x1 * cx, i = x0 * cx, k = y1 * cy, l = y0 * cy;
            ihi[c] = (wmax(i, h) + c0) + wmax(l, k);
            ilo[c] = (c0 + wmin(h, i)) + wmin(k, l);
            continue;
        }
        const Inst& f = T.ins[c];
        const int a = f.in0, b = f.in1;
        const float k = f.k0;
        switch (f.op)
        {
            case kAdd: ihi[c] = ihi[b] + ihi[a]; ilo[c] = ilo[b] + ilo[a]; break;
            case kSub: ihi[c] = ihi[a] - ilo[b]; ilo[c] = ilo[a] - ihi[b]; break;
            case kMul:
            {
                const float j = ilo[b] * ilo[a], i = ihi[b] * ilo[a], l = ilo[b] * ihi[a], h = ihi[b] * ihi[a];
                ihi[c] = wmax(wmax(j, i), wmax(l, h));
                ilo[c] = wmin(wmin(i, j), wmin(h, l));
                break;
            }
            case kNeg: ihi[c] = -ilo[a]; ilo[c] = -ihi[a]; break;
            case kSquare:
            {
                const float h = ilo[a], i = ihi[a];
                const float j = h * h, kk = i * i;
                ihi[c] = wmax(j, kk);
                const float m = wmin(j, kk);
                ilo[c] = (h <= 0.0f) ? (i >= 0.0f ? 0.0f : m) : m;
                break;
            }
            case kSqrt:
                if (ihi[a] < 0.0f)
                {
                    ilo[c] = 1.0f;  // empty interval
                    ihi[c] = 0.0f;
                }
                else
                {
                    ilo[c] = std::sqrt(wmax(ilo[a], 0.0f));
                    ihi[c] = std::sqrt(wmax(ihi[a], 0.0f));
                }
                break;
            case kMin: ihi[c] = wmin(ihi[a], ihi[b]); ilo[c] = wmin(ilo[a], ilo[b]); break;
            case kMax: ihi[c] = wmax(ihi[a], ihi[b]); ilo[c] = wmax(ilo[a], ilo[b]); break;
            case kSubK: ihi[c] = ihi[a] - k; ilo[c] = ilo[a] - k; break;
            case kKSub: ihi[c] = k - ilo[a]; ilo[c] = k - ihi[a]; break;
            default: break;
        }
    }
    lo = ilo[T.root];
    hi = ihi[T.root];
}

// Evaluates one node over a row span of l pixels starting at column x0, at height yv.
void eval_row(int node, int x0, float yv, int l)
{
    float* out = rowbuf[node];
    if (aff[node])
    {
        const float d = std::fma(acy[node], yv, ac0[node]);
        const float cx = acx[node];
        for (int i = 0; i < l; i++) out[i] = std::fma(cx, g_xcoord[x0 + i], d);
        return;
    }
    const Inst& f = T.ins[node];
    const int a = f.in0, b = f.in1;
    const float* g = a >= 0 ? rowbuf[a] : nullptr;
    const float* h = b >= 0 ? rowbuf[b] : nullptr;
    const float k = f.k0;
    switch (f.op)
    {
        case kSubK: for (int i = 0; i < l; i++) out[i] = g[i] - k; return;
        case kKSub: for (int i = 0; i < l; i++) out[i] = k - g[i]; return;
        default: break;
    }
    const float n = val[a];
    const float d = b < 0 ? 0.0f : val[b];
    const bool s0 = T.sflag[node] & 1, s1 = T.sflag[node] & 2;
    for (int i = 0; i < l; i++)
    {
        const float A = s0 ? n : g[i];
        const float B = s1 ? d : (h ? h[i] : 0.0f);
        float r;
        switch (f.op)
        {
            case kAdd: r = B + A; break;
            case kSub: r = A - B; break;
            case kMul: r = B * A; break;
            case kMin: r = wmin(A, B); break;
            case kMax: r = wmax(A, B); break;
            case kNeg: r = -g[i]; break;
            case kSquare: r = g[i] * g[i]; break;
            case kSqrt: r = std::sqrt(g[i]); break;
            default: return;
        }
        out[i] = r;
    }
}

float eval_scalar(int node, float yv)
{
    if (aff[node]) return std::fma(acy[node], yv, ac0[node]);
    const Inst& f = T.ins[node];
    const float c = val[f.in0], k = f.k0;
    switch (f.op)
    {
        case kSubK: return c - k;
        case kKSub: return k - c;
        default: break;
    }
    const float d = f.in1 < 0 ? 0.0f : val[f.in1];
    switch (f.op)
    {
        case kAdd: return d + c;
        case kSub: return c - d;
        case kMul: return d * c;
        case kMin: return wmin(c, d);
        case kMax: return wmax(c, d);
        case kNeg: return -c;
        case kSquare: return c * c;
        case kSqrt: return std::sqrt(c);
        default: return c;
    }
}

// Brute-force evaluation of an inclusive pixel rectangle. x-only nodes are
// evaluated once for the span, y-only nodes once per row as scalars.
void leaf_eval(int x0, int x1, int y0, int y1)
{
    if (x1 < x0 || y1 < y0) return;
    const int l = x1 - x0 + 1;
    for (int nd : T.list[1]) eval_row(nd, x0, 0.0f, l);
    for (int y = y0; y <= y1; y++)
    {
        const float e = g_ycoord[y];
        for (int nd : T.list[2]) val[nd] = eval_scalar(nd, e);
        for (int nd : T.list[3]) eval_row(nd, x0, e, l);
        std::uint8_t* row = g_image + y * kImageSize + x0;
        const int r = T.root;
        const std::uint8_t kr = T.kind[r];
        if (kr == 0 || kr == 2)
        {
            if (!(val[r] >= 0.0f)) std::memset(row, 255, static_cast<std::size_t>(l));
        }
        else
        {
            for (int i = 0; i < l; i++)
                if (!(rowbuf[r][i] >= 0.0f)) row[i] = 255;
        }
    }
}

void fill_rect(int x0, int x1, int y0, int y1)
{
    for (int y = y0; y <= y1; y++)
        std::memset(g_image + y * kImageSize + x0, 255, static_cast<std::size_t>(x1 - x0 + 1));
}

inline void tighten(int idx, float lo, float hi)
{
    if (lo > hi) return;
    if (tset[idx])
    {
        tlo[idx] = wmax(tlo[idx], lo);
        thi[idx] = wmin(thi[idx], hi);
    }
    else
    {
        tlo[idx] = lo;
        thi[idx] = hi;
        tset[idx] = true;
    }
}

// Collects the term's sub-expression in post-order (in0 subtree first) and
// renumbers it into T. Returns false if the term is too large.
bool extract_term(int term_root)
{
    g_term_stamps.next();
    g_term_nodes.clear();
    g_term_stack.clear();
    g_term_stack.push_back({term_root, 0});
    while (!g_term_stack.empty())
    {
        auto& top = g_term_stack.back();
        const int o = top.first;
        if (top.second == 0)
        {
            if (!g_term_stamps.visit(o))
            {
                g_term_stack.pop_back();
                continue;
            }
            top.second = 1;
            const Inst& I = g_prog[static_cast<std::size_t>(o)];
            if (I.in1 >= 0) g_term_stack.push_back({I.in1, 0});
            if (I.in0 >= 0) g_term_stack.push_back({I.in0, 0});
            continue;
        }
        if (static_cast<int>(g_term_nodes.size()) >= kMaxTermSize) return false;
        g_term_nodes.push_back(o);
        g_term_stack.pop_back();
    }
    const int n = static_cast<int>(g_term_nodes.size());
    for (int i = 0; i < n; i++) g_remap[static_cast<std::size_t>(g_term_nodes[static_cast<std::size_t>(i)])] = i;
    return true;
}

void render_term(int term_root)
{
    if (!extract_term(term_root))
    {
        g_skipped_terms++;
        return;
    }
    const int g = static_cast<int>(g_term_nodes.size());
    T.n = g;
    T.root = 0;
    T.fill_only = true;
    for (auto& list : T.list) list.clear();
    std::uint8_t nonneg[kMaxTermSize];
    std::memset(nonneg, 0, sizeof nonneg);

    // Forward pass: copy and renumber, classify by variable dependence, fold
    // constants, track affine forms and which nodes are provably non-negative.
    for (int d = 0; d < g; d++)
    {
        const int global = g_term_nodes[static_cast<std::size_t>(d)];
        Inst I = g_prog[static_cast<std::size_t>(global)];
        if (I.in0 >= 0) I.in0 = static_cast<std::int16_t>(g_remap[static_cast<std::size_t>(I.in0)]);
        if (I.in1 >= 0) I.in1 = static_cast<std::int16_t>(g_remap[static_cast<std::size_t>(I.in1)]);
        T.ins[d] = I;
        const int j = I.op, a = I.in0, f = I.in1;

        int kd;
        bool fold = false;
        if (j == kConst)
        {
            kd = 0;
            val[d] = I.k0;
        }
        else if (j == kVarX) kd = 1;
        else if (j == kVarY) kd = 2;
        else if (j >= kNeg && j <= kSqrt)
        {
            kd = (a < 0) ? 0 : T.kind[a];
            fold = (kd == 0);
        }
        else
        {
            const int c = T.kind[a], b = T.kind[f];
            if (c == 3 || b == 3) kd = 3;
            else if (c == 0) kd = b;
            else if (b == 0) kd = c;
            else kd = (c == b) ? c : 3;
            fold = (kd == 0);
        }
        T.kind[d] = static_cast<std::uint8_t>(kd);
        if (fold)
        {
            const float A = val[a];
            const float B = f < 0 ? 0.0f : val[f];
            float y = A;
            switch (j)
            {
                case kAdd: y = B + A; break;
                case kSub: y = A - B; break;
                case kMul: y = B * A; break;
                case kMin: y = wmin(A, B); break;
                case kMax: y = wmax(A, B); break;
                case kNeg: y = -A; break;
                case kSquare: y = A * A; break;
                case kSqrt: y = std::sqrt(A); break;
                default: break;
            }
            val[d] = y;
        }

        aff[d] = false;
        {
            float cx = 0, cy = 0, c0 = I.k0;
            bool ok = false;
            if (j == kConst) { cx = 0; cy = 0; c0 = I.k0; ok = true; }
            else if (j == kVarX) { cx = 1; cy = 0; c0 = 0; ok = true; }
            else if (j == kVarY) { cx = 0; cy = 1; c0 = 0; ok = true; }
            else if (j == kNeg)
            {
                if (aff[a]) { c0 = -ac0[a]; cy = -acy[a]; cx = -acx[a]; ok = true; }
            }
            else if (j == kAdd || j == kSub)
            {
                if (aff[a] && aff[f])
                {
                    const float s = (j == kAdd) ? 1.0f : -1.0f;
                    c0 = ac0[f] * s + ac0[a];
                    cy = acy[f] * s + acy[a];
                    cx = acx[f] * s + acx[a];
                    ok = true;
                }
            }
            else if (j == kMul)
            {
                if (T.kind[a] == 0 && aff[f])
                {
                    const float k = val[a];
                    c0 = ac0[f] * k; cy = acy[f] * k; cx = acx[f] * k; ok = true;
                }
                else if (T.kind[f] == 0 && aff[a])
                {
                    const float k = val[f];
                    c0 = ac0[a] * k; cy = acy[a] * k; cx = acx[a] * k; ok = true;
                }
            }
            if (ok)
            {
                aff[d] = true;
                acx[d] = cx;
                acy[d] = cy;
                ac0[d] = c0;
            }
        }

        switch (j)
        {
            case kConst: nonneg[d] = I.k0 >= 0.0f; break;
            case kAdd: case kMul: case kMin: nonneg[d] = nonneg[a] && nonneg[f]; break;
            case kMax: nonneg[d] = nonneg[a] || nonneg[f]; break;
            case kSquare: case kSqrt: nonneg[d] = 1; break;
            default: break;
        }
        if (j == kMin || (kd == 3 && j != kMax)) T.fill_only = false;
        if (global == term_root) T.root = d;
    }

    // Rewrite sqrt(e) - k as e - k^2 (and k - sqrt(e) as k^2 - e) where that
    // keeps the sign, on nodes that reach the root through min, max and neg only.
    std::uint8_t live[kMaxTermSize];
    std::memset(live, 0, sizeof live);
    live[T.root] = 1;
    for (int d = g - 1; d >= 0; d--)
    {
        if (!live[d]) continue;
        const Inst& I = T.ins[d];
        if (I.op == kMin || I.op == kMax)
        {
            if (I.in0 >= 0) live[I.in0] = 1;
            if (I.in1 >= 0) live[I.in1] = 1;
        }
        else if (I.op == kNeg)
        {
            if (I.in0 >= 0) live[I.in0] = 1;
        }
    }
    for (int d = 0; d < g; d++)
    {
        if (!live[d]) continue;
        Inst& I = T.ins[d];
        if (I.op != kSub || I.in0 < 0 || I.in1 < 0) continue;
        const int c = I.in0, b = I.in1;
        int k_idx, sq, new_op;
        if (T.ins[c].op == kSqrt && T.kind[b] == 0)
        {
            k_idx = b; sq = c; new_op = kSubK;
        }
        else
        {
            if (T.kind[c] != 0) continue;
            if (T.ins[b].op != kSqrt) continue;
            k_idx = c; sq = b; new_op = kKSub;
        }
        const int arg = T.ins[sq].in0;
        if (arg < 0) continue;
        const float y = val[k_idx];
        if (y < 0.0f) continue;
        if (!nonneg[arg]) continue;
        I.op = new_op;
        I.in0 = static_cast<std::int16_t>(arg);
        I.in1 = -1;
        I.k0 = y * y;
        I.k1 = 0.0f;
    }

    // Liveness for the row evaluator: stop at affine and constant nodes, which are
    // evaluated directly. Bucket live nodes by what they depend on.
    std::memset(live, 0, sizeof live);
    live[T.root] = 1;
    for (int d = g - 1; d >= 0; d--)
    {
        if (!live[d] || aff[d] || T.kind[d] == 0) continue;
        const Inst& I = T.ins[d];
        if (I.in0 >= 0 && T.kind[I.in0]) live[I.in0] = 1;
        if (I.in1 >= 0 && T.kind[I.in1]) live[I.in1] = 1;
    }
    for (int d = 0; d < g; d++)
    {
        const int f = T.kind[d];
        const Inst& I = T.ins[d];
        int h = 0, b = 0;
        if (I.in0 >= 0)
        {
            const int kc = T.kind[I.in0];
            h = (kc == (kc ? (f == 3 ? 2 : 0) : 0));
        }
        if (I.in1 >= 0)
        {
            const int c = T.kind[I.in1];
            b = c ? 0 : 2;
            if (c && f == 3) b = (c == 2) ? 2 : 0;
        }
        T.sflag[d] = static_cast<std::uint8_t>(b | h);
        if (live[d] && f >= 1) T.list[f].push_back(d);
    }

    // Skip the term if it is positive everywhere on the square.
    float lo, hi;
    interval_eval(-1, 1, -1, 1, lo, hi);
    if (lo > hi || !(lo < 0.0f))
    {
        g_skipped_terms++;
        return;
    }

    // Push "root <= 0" backwards to get target intervals for each node.
    std::memset(tset, 0, sizeof tset);
    tset[T.root] = true;
    tlo[T.root] = -1e30f;
    thi[T.root] = 0.0f;
    const float kNegBig = -1e30f;
    for (int d = g - 1; d >= 0; d--)
    {
        if (!tset[d] || aff[d]) continue;
        const Inst& I = T.ins[d];
        const int a = I.in0, b = I.in1;
        const bool na = a < 0, nb = b < 0;
        float y, aa, z, ca;  // in0 in [y, aa], in1 in [z, ca]
        if (na) y = aa = I.k0;
        else { y = ilo[a]; aa = ihi[a]; }
        if (nb) z = ca = I.k1;
        else { z = ilo[b]; ca = ihi[b]; }
        const float fa = tlo[d], da = thi[d];
        switch (I.op)
        {
            case kAdd:
                if (!na) tighten(a, wmax(fa - ca, y), wmin(da - z, aa));
                if (!nb) tighten(b, wmax(fa - aa, z), wmin(da - y, ca));
                break;
            case kSub:
                if (!na) tighten(a, wmax(z + fa, y), wmin(ca + da, aa));
                if (!nb) tighten(b, wmax(y - da, z), wmin(aa - fa, ca));
                break;
            case kMul:
                if (!na && !(z <= 0.0f && ca >= 0.0f))
                {
                    const float q1 = fa / z, q2 = fa / ca, q3 = da / z, q4 = da / ca;
                    tighten(a, wmax(wmin(wmin(q1, q2), wmin(q3, q4)), y),
                            wmin(wmax(wmax(q1, q2), wmax(q4, q3)), aa));
                }
                if (!(y <= 0.0f && aa >= 0.0f) && !nb)
                {
                    const float q1 = fa / y, q2 = fa / aa, q3 = da / y, q4 = da / aa;
                    tighten(b, wmax(z, wmin(wmin(q1, q2), wmin(q3, q4))),
                            wmin(wmax(wmax(q2, q1), wmax(q4, q3)), ca));
                }
                break;
            case kNeg:
                if (!na) tighten(a, wmax(-da, y), wmin(-fa, aa));
                break;
            case kSquare:
                if (!na)
                {
                    const float s = std::sqrt(wmax(da, 0.0f));
                    tighten(a, wmax(y, -s), wmin(s, aa));
                }
                break;
            case kSqrt:
                if (!na && !(da < 0.0f))
                {
                    const float l0 = wmax(fa, 0.0f), h0 = wmax(da, 0.0f);
                    tighten(a, wmax(y, l0 * l0), wmin(h0 * h0, aa));
                }
                break;
            case kMin:
                if (!na) tighten(a, wmax(y, fa), aa);
                if (!nb)
                {
                    tighten(b, wmax(z, fa), ca);
                    if (!(y <= da)) tighten(b, wmax(z, kNegBig), wmin(ca, da));
                }
                if (!na && !(z <= da)) tighten(a, wmax(y, kNegBig), wmin(aa, da));
                break;
            case kMax:
                if (!na) tighten(a, wmax(y, kNegBig), wmin(aa, da));
                if (!nb) tighten(b, wmax(z, kNegBig), wmin(ca, da));
                break;
            case kSubK:
                if (!na) tighten(a, wmax(fa + I.k0, y), wmin(da + I.k0, aa));
                break;
            case kKSub:
                if (!na) tighten(a, wmax(I.k0 - da, y), wmin(I.k0 - fa, aa));
                break;
            default: break;
        }
    }

    // Clip the square by the half-planes lo <= cx*x + cy*y + c0 <= hi of every
    // constrained affine node (Sutherland-Hodgman).
    static std::vector<float> PX, PY, NX, NY;
    PX.assign({-1, 1, 1, -1});
    PY.assign({-1, -1, 1, 1});
    auto clip = [&](auto value) -> bool {  // keep value(x, y) <= 1e-6
        const int m = static_cast<int>(PX.size());
        NX.clear();
        NY.clear();
        float px = PX[static_cast<std::size_t>(m - 1)], py = PY[static_cast<std::size_t>(m - 1)];
        float pv = value(px, py);
        bool pin = pv <= 1e-6f;
        for (int c = 0; c < m; c++)
        {
            const float cx = PX[static_cast<std::size_t>(c)], cy = PY[static_cast<std::size_t>(c)];
            const float cv = value(cx, cy);
            const bool cin = cv <= 1e-6f;
            if (cin != pin)
            {
                if (NX.size() >= kMaxPolygon) return false;
                const float t = pv / (pv - cv);
                NY.push_back(t * (cy - py) + py);
                NX.push_back(t * (cx - px) + px);
            }
            if (cin)
            {
                if (NX.size() >= kMaxPolygon) return false;
                NY.push_back(cy);
                NX.push_back(cx);
            }
            px = cx; py = cy; pv = cv; pin = cin;
        }
        std::swap(PX, NX);
        std::swap(PY, NY);
        return !PX.empty();
    };
    for (int d = 0; d < g; d++)
    {
        if (!tset[d] || !aff[d]) continue;
        const float lo_ = tlo[d], hi_ = thi[d];
        const float c0 = ac0[d], cx = acx[d], cy = acy[d];
        if (cx == 0.0f && cy == 0.0f)
        {
            if (lo_ > c0 || hi_ < c0)
            {
                g_skipped_terms++;
                return;
            }
            continue;
        }
        const bool hi_inf = hi_ >= 1e20f;
        if (hi_inf && !(lo_ > -1e20f)) continue;
        if (!hi_inf)
        {
            const float ia = hi_ - c0;
            if (!clip([&](float X, float Y) { return (X * cx - ia) + Y * cy; }))
            {
                g_skipped_terms++;
                return;
            }
        }
        if (!(lo_ <= -1e20f))
        {
            const float ia = lo_ - c0;
            if (!clip([&](float X, float Y) { return ia - (X * cx + cy * Y); }))
            {
                g_skipped_terms++;
                return;
            }
        }
    }

    // Pixel bounding box of the clipped polygon.
    float xmin = PX[0], xmax = PX[0], ymin = PY[0], ymax = PY[0];
    for (std::size_t i = 1; i < PX.size(); i++)
    {
        ymax = wmax(ymax, PY[i]);
        ymin = wmin(ymin, PY[i]);
        xmax = wmax(xmax, PX[i]);
        xmin = wmin(xmin, PX[i]);
    }
    if (wmax(xmax - xmin, 0.0f) * wmax(ymax - ymin, 0.0f) <= 0.0f)
    {
        g_skipped_terms++;
        return;
    }
    auto to_hi = [](float v) {
        const int r = static_cast<int>(std::floor(v * 511.5f + 511.5f));
        return r < -1 ? -1 : (r > 1023 ? 1023 : r);
    };
    auto to_lo = [](float v) {
        const int r = static_cast<int>(std::ceil(v * 511.5f + 511.5f));
        return r < 0 ? 0 : (r > 1024 ? 1024 : r);
    };
    const int iy1 = to_hi(ymax), ix1 = to_hi(xmax), iy0 = to_lo(ymin), ix0 = to_lo(xmin);
    if (iy1 < iy0 || ix1 < ix0)
    {
        g_skipped_terms++;
        return;
    }
    if (T.fill_only)
    {
        fill_rect(ix0, ix1, iy0, iy1);
        return;
    }

    // Subdivide the box: drop rectangles that are outside, fill ones that are
    // inside, brute-force small ones.
    struct Rect
    {
        int x0, x1, y0, y1;
    };
    Rect stack[128];
    int count = 1;
    stack[0] = {ix0, ix1, iy0, iy1};
    while (count)
    {
        const int top = count - 1;
        const Rect r = stack[top];
        const int h = r.y1 - r.y0, w = r.x1 - r.x0;
        if ((h + 1) * (w + 1) <= 4096)
        {
            leaf_eval(r.x0, r.x1, r.y0, r.y1);
            count = top;
            continue;
        }
        float lo2, hi2;
        interval_eval(g_xcoord[r.x0], g_xcoord[r.x1], g_ycoord[r.y0], g_ycoord[r.y1], lo2, hi2);
        if (lo2 > hi2 || lo2 >= 0.0f)
        {
            count = top;
            continue;
        }
        if (!(hi2 >= 0.0f))
        {
            fill_rect(r.x0, r.x1, r.y0, r.y1);
            count = top;
            continue;
        }
        if (count >= 127)
        {
            leaf_eval(r.x0, r.x1, r.y0, r.y1);
            count = top;
            continue;
        }
        if (w >= h)
        {
            const int mid = (r.x0 + r.x1) >> 1;
            stack[top] = {mid + 1, r.x1, r.y0, r.y1};
            stack[count] = {r.x0, mid, r.y0, r.y1};
        }
        else
        {
            const int mid = (r.y0 + r.y1) >> 1;
            stack[top] = {r.x0, r.x1, mid + 1, r.y1};
            stack[count] = {r.x0, r.x1, r.y0, mid};
        }
        count++;
    }
}

}  // namespace

int load(const char* text)
{
    for (int i = 0; i < kImageSize; i++)
    {
        const float t = static_cast<float>(i) * (2.0f / 1023.0f);
        g_xcoord[i] = t + -1.0f;
        g_ycoord[i] = t + -1.0f;
    }
    g_last_var_x = g_last_var_y = -1;
    g_prog.clear();
    std::unordered_map<std::string, int> names;
    names.reserve(kMaxInstructions);
    const char* p = text;
    while (*p)
    {
        int n = 0;
        while (p[n] && p[n] != '\n' && n < 255) n++;
        char line[256];
        std::memcpy(line, p, static_cast<std::size_t>(n));
        line[n] = 0;
        const char* q = p + n;
        while (*q && *q != '\n') q++;  // characters beyond 255 are ignored
        p = q + (*q == '\n');

        char name[32] = {0}, op[32] = {0}, a1[64] = {0}, a2[64] = {0};
        const int cnt = std::sscanf(line, "%31s %31s %63s %63s", name, op, a1, a2);
        if (cnt < 2 || line[0] == '#') continue;

        int code;
        if (!std::strcmp(op, "const")) code = kConst;
        else if (!std::strcmp(op, "var-x")) code = kVarX;
        else if (!std::strcmp(op, "var-y")) code = kVarY;
        else if (!std::strcmp(op, "add")) code = kAdd;
        else if (!std::strcmp(op, "sub")) code = kSub;
        else if (!std::strcmp(op, "mul")) code = kMul;
        else if (!std::strcmp(op, "neg")) code = kNeg;
        else if (!std::strcmp(op, "square")) code = kSquare;
        else if (!std::strcmp(op, "sqrt")) code = kSqrt;
        else if (!std::strcmp(op, "min")) code = kMin;
        else if (!std::strcmp(op, "max")) code = kMax;
        else return -1;
        if (static_cast<int>(g_prog.size()) >= kMaxInstructions || static_cast<int>(names.size()) >= kMaxInstructions)
            return -2;

        names.emplace(name, static_cast<int>(g_prog.size()));  // first definition wins
        Inst in{code, -1, -1, 0.0f, 0.0f};
        auto arg = [&](const char* s, std::int16_t& idx, float& k) {
            if (s[0] == '_')
            {
                const auto it = names.find(s);
                if (it != names.end()) idx = static_cast<std::int16_t>(it->second);
            }
            else
            {
                k = std::strtof(s, nullptr);
            }
        };
        if (cnt >= 3) arg(a1, in.in0, in.k0);
        if (cnt >= 4) arg(a2, in.in1, in.k1);
        if (code == kVarX) g_last_var_x = static_cast<int>(g_prog.size());
        if (code == kVarY) g_last_var_y = static_cast<int>(g_prog.size());
        g_prog.push_back(in);
    }

    g_frame_stamps.resize(g_prog.size());
    g_term_stamps.resize(g_prog.size());
    g_remap.assign(g_prog.size(), 0);
    return static_cast<int>(g_prog.size());
}

void render()
{
    g_terms.clear();
    g_term_count = 0;
    g_skipped_terms = 0;
    std::memset(g_image, 0, sizeof g_image);
    if (g_prog.empty()) return;

    // Split the top-level min-tree into terms.
    g_frame_stamps.next();
    g_frame_stack.clear();
    g_frame_stack.push_back(static_cast<int>(g_prog.size()) - 1);
    while (!g_frame_stack.empty())
    {
        const int t = g_frame_stack.back();
        g_frame_stack.pop_back();
        if (!g_frame_stamps.visit(t)) continue;
        const Inst& I = g_prog[static_cast<std::size_t>(t)];
        if (I.op == kMin)
        {
            if (I.in0 >= 0) g_frame_stack.push_back(I.in0);
            if (I.in1 >= 0) g_frame_stack.push_back(I.in1);
        }
        else
        {
            g_terms.push_back(t);
        }
    }
    g_term_count = static_cast<int>(g_terms.size());
    for (int t : g_terms) render_term(t);
}

const std::uint8_t* image() { return g_image; }
int instruction_count() { return static_cast<int>(g_prog.size()); }
int term_count() { return g_term_count; }
int skipped_terms() { return g_skipped_terms; }

}  // namespace frontier
