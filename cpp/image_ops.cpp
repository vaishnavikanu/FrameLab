// FrameLab native core (C++17 + pybind11).
//   1. adjust_brightness_contrast : 256-entry LUT, bit-exact with the NumPy reference
//   2. motion_estimate            : block matching (SAD) with full search or diamond search
//   3. gradient_energy            : mean absolute spatial gradient (spatial complexity)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using u8array = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

// ---------------------------------------------------------------- brightness / contrast
py::array_t<std::uint8_t> adjust_brightness_contrast(u8array image, double alpha, double beta) {
    auto info = image.request();
    if (info.ndim != 2 && info.ndim != 3) {
        throw std::runtime_error("Expected a 2D or 3D uint8 image array");
    }
    // Build the LUT in float32 arithmetic so it matches NumPy's float32 reference exactly.
    // `volatile` stops the compiler from fusing multiply+add (FMA), which would change rounding.
    std::uint8_t lut[256];
    const float a = static_cast<float>(alpha);
    const float b = static_cast<float>(beta);
    for (int i = 0; i < 256; ++i) {
        volatile float prod = static_cast<float>(i) * a;
        float v = std::nearbyintf(prod + b);
        if (!(v > 0.0f)) v = 0.0f;        // also maps NaN to 0
        if (v > 255.0f) v = 255.0f;
        lut[i] = static_cast<std::uint8_t>(v);
    }
    py::array_t<std::uint8_t> result(info.shape);
    const auto* src = static_cast<const std::uint8_t*>(info.ptr);
    auto* dst = result.mutable_data();
    const std::size_t n = static_cast<std::size_t>(info.size);
    {
        py::gil_scoped_release release;
        const std::uint8_t* __restrict s = src;
        std::uint8_t* __restrict d = dst;
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {              // manual unroll: independent loads hide latency
            d[i] = lut[s[i]];         d[i + 1] = lut[s[i + 1]]; d[i + 2] = lut[s[i + 2]]; d[i + 3] = lut[s[i + 3]];
            d[i + 4] = lut[s[i + 4]]; d[i + 5] = lut[s[i + 5]]; d[i + 6] = lut[s[i + 6]]; d[i + 7] = lut[s[i + 7]];
        }
        for (; i < n; ++i) d[i] = lut[s[i]];
    }
    return result;
}

// ---------------------------------------------------------------- motion estimation
namespace {

struct Frame {
    const std::uint8_t* data;
    int w, h;
};

inline int block_sad(const Frame& cur, const Frame& ref, int bx, int by, int dx, int dy, int B) {
    int sad = 0;
    for (int y = 0; y < B; ++y) {
        const std::uint8_t* c = cur.data + static_cast<std::size_t>(by + y) * cur.w + bx;
        const std::uint8_t* r = ref.data + static_cast<std::size_t>(by + y + dy) * ref.w + bx + dx;
        for (int x = 0; x < B; ++x) sad += std::abs(static_cast<int>(c[x]) - static_cast<int>(r[x]));
    }
    return sad;
}

inline bool candidate_ok(const Frame& f, int bx, int by, int dx, int dy, int B, int S) {
    if (std::abs(dx) > S || std::abs(dy) > S) return false;
    const int x = bx + dx, y = by + dy;
    return x >= 0 && y >= 0 && x + B <= f.w && y + B <= f.h;
}

const int LDSP[8][2] = {{0, -2}, {-1, -1}, {1, -1}, {-2, 0}, {2, 0}, {-1, 1}, {1, 1}, {0, 2}};
const int SDSP[4][2] = {{0, -1}, {-1, 0}, {1, 0}, {0, 1}};

// Tie-breaking everywhere is "strictly lower SAD wins", starting from the zero vector,
// so the Python reference implementation produces identical vectors.
void search_full(const Frame& cur, const Frame& ref, int bx, int by, int B, int S,
                 int& best_dx, int& best_dy, int& best_sad) {
    best_dx = best_dy = 0;
    best_sad = block_sad(cur, ref, bx, by, 0, 0, B);
    for (int dy = -S; dy <= S; ++dy) {
        for (int dx = -S; dx <= S; ++dx) {
            if (dx == 0 && dy == 0) continue;
            if (!candidate_ok(ref, bx, by, dx, dy, B, S)) continue;
            const int s = block_sad(cur, ref, bx, by, dx, dy, B);
            if (s < best_sad) { best_sad = s; best_dx = dx; best_dy = dy; }
        }
    }
}

void search_diamond(const Frame& cur, const Frame& ref, int bx, int by, int B, int S,
                    int& best_dx, int& best_dy, int& best_sad) {
    int cx = 0, cy = 0;
    best_sad = block_sad(cur, ref, bx, by, 0, 0, B);
    while (true) {                                   // large diamond until the centre wins
        int ndx = cx, ndy = cy;
        for (const auto& p : LDSP) {
            const int dx = cx + p[0], dy = cy + p[1];
            if (!candidate_ok(ref, bx, by, dx, dy, B, S)) continue;
            const int s = block_sad(cur, ref, bx, by, dx, dy, B);
            if (s < best_sad) { best_sad = s; ndx = dx; ndy = dy; }
        }
        if (ndx == cx && ndy == cy) break;
        cx = ndx; cy = ndy;
    }
    int fdx = cx, fdy = cy;                          // one small-diamond refinement
    for (const auto& p : SDSP) {
        const int dx = cx + p[0], dy = cy + p[1];
        if (!candidate_ok(ref, bx, by, dx, dy, B, S)) continue;
        const int s = block_sad(cur, ref, bx, by, dx, dy, B);
        if (s < best_sad) { best_sad = s; fdx = dx; fdy = dy; }
    }
    best_dx = fdx; best_dy = fdy;
}

}  // namespace

// For every BxB block of `cur`, find the displacement (dx, dy) into `prev` that minimises SAD.
// Returns (mvs[nby, nbx, 2] = (dx, dy), sads[nby, nbx]).
py::tuple motion_estimate(u8array prev, u8array cur, int block, int search, const std::string& method) {
    auto pi = prev.request();
    auto ci = cur.request();
    if (pi.ndim != 2 || ci.ndim != 2) throw std::runtime_error("prev and cur must be 2D grayscale arrays");
    if (pi.shape[0] != ci.shape[0] || pi.shape[1] != ci.shape[1])
        throw std::runtime_error("prev and cur must have the same shape");
    if (block < 2 || search < 1) throw std::runtime_error("block must be >= 2 and search >= 1");
    const int H = static_cast<int>(ci.shape[0]);
    const int W = static_cast<int>(ci.shape[1]);
    if (H < block || W < block) throw std::runtime_error("frame smaller than one block");
    bool diamond;
    if (method == "diamond") diamond = true;
    else if (method == "full") diamond = false;
    else throw std::runtime_error("method must be 'diamond' or 'full'");

    const int nby = H / block, nbx = W / block;
    std::vector<py::ssize_t> mv_shape{nby, nbx, 2};
    std::vector<py::ssize_t> sad_shape{nby, nbx};
    py::array_t<std::int32_t> mvs(mv_shape);
    py::array_t<std::int32_t> sads(sad_shape);
    auto* mv_ptr = mvs.mutable_data();
    auto* sad_ptr = sads.mutable_data();
    const Frame ref{static_cast<const std::uint8_t*>(pi.ptr), W, H};
    const Frame cr{static_cast<const std::uint8_t*>(ci.ptr), W, H};
    {
        py::gil_scoped_release release;
        for (int by = 0; by < nby; ++by) {
            for (int bx = 0; bx < nbx; ++bx) {
                int dx, dy, s;
                if (diamond) search_diamond(cr, ref, bx * block, by * block, block, search, dx, dy, s);
                else search_full(cr, ref, bx * block, by * block, block, search, dx, dy, s);
                const std::size_t k = static_cast<std::size_t>(by) * nbx + bx;
                mv_ptr[2 * k] = dx;
                mv_ptr[2 * k + 1] = dy;
                sad_ptr[k] = s;
            }
        }
    }
    return py::make_tuple(mvs, sads);
}

// ---------------------------------------------------------------- spatial complexity
double gradient_energy(u8array gray) {
    auto info = gray.request();
    if (info.ndim != 2) throw std::runtime_error("gradient_energy expects a 2D grayscale array");
    const long H = static_cast<long>(info.shape[0]);
    const long W = static_cast<long>(info.shape[1]);
    if (H < 2 || W < 2) throw std::runtime_error("image must be at least 2x2");
    const auto* p = static_cast<const std::uint8_t*>(info.ptr);
    long long total = 0;
    {
        py::gil_scoped_release release;
        for (long y = 0; y < H - 1; ++y) {
            const std::uint8_t* row = p + y * W;
            const std::uint8_t* next = p + (y + 1) * W;
            for (long x = 0; x < W - 1; ++x) {
                total += std::abs(static_cast<int>(row[x + 1]) - static_cast<int>(row[x]));
                total += std::abs(static_cast<int>(next[x]) - static_cast<int>(row[x]));
            }
        }
    }
    return static_cast<double>(total) / static_cast<double>((H - 1) * (W - 1));
}

PYBIND11_MODULE(_cpp, m) {
    m.doc() = "FrameLab native kernels";
    m.attr("__version__") = "2.0.0";
    m.def("adjust_brightness_contrast", &adjust_brightness_contrast,
          py::arg("image"), py::arg("alpha") = 1.0, py::arg("beta") = 0.0,
          "saturate(round(image*alpha + beta)) via a 256-entry lookup table.");
    m.def("motion_estimate", &motion_estimate,
          py::arg("prev"), py::arg("cur"), py::arg("block") = 16, py::arg("search") = 16,
          py::arg("method") = "diamond",
          "Block-matching motion estimation. Returns (mvs[nby,nbx,2], sads[nby,nbx]).");
    m.def("gradient_energy", &gradient_energy, py::arg("gray"),
          "Mean |dx|+|dy| of a grayscale image.");
}
