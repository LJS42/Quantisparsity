#pragma once

// Single-row INT4 sparse quantization: min ||(w-z)X||_2^2, nnz(z) <= k.
// w and z are row vectors; X is Weights x samples. k is this row's budget.
// z is unscaled signed INT4 (-8,...,7).
// Lift with binary y: -8*y <= z <= 7*y, sum(y) <= k.
// All z and y are integer. The node LMO optimizes over the convex hull
// of the node's discrete feasible points, using endpoint choices and top-k.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <functional>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <gpuboscia/node.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <gpuboscia/lmo.cuh>
#include <gpuboscia/gpuboscia.cuh>
#include <gpuboscia/logger.h>

namespace quantization_sparse {
#ifndef QUANTIZATION_SPARSE_WEIGHTS
#define QUANTIZATION_SPARSE_WEIGHTS 576
#endif
constexpr int Weights = QUANTIZATION_SPARSE_WEIGHTS;
static_assert(Weights > 0, "Weights must be positive");
constexpr int Vars = 2 * Weights;

// H is symmetric. Accumulate d*H with contiguous, vectorizable row reads.
inline void host_matvec(const double* H, const double* d, double* product,
                        double scale = 1) {
    std::fill(product,product+Weights,0.0);
    for (int k = 0; k < Weights; ++k) {
        const double value = scale*d[k];
        for (int j = 0; j < Weights; ++j) product[j] += value*H[k*Weights+j];
    }
}

inline double host_dot(int n, const double* a, const double* b) {
    return std::inner_product(a,a+n,b,0.0);
}

struct SparseInt4LMO {
    static constexpr bool supports_dicg = true;
    static constexpr int SortSize = [] {
        int n = 1;
        while (n < Weights) n *= 2;
        return n;
    }();
    int k;

    struct Domain {
        double lo, hi;
        bool off, on;
    };
    __host__ __device__ static Domain domain(
        int i, int m, const double* lb, const double* ub) {
        return {ceil(fmax(-8.0, lb[i])), floor(fmin(7.0, ub[i])),
            lb[i] <= 0 && ub[i] >= 0 && lb[m+i] <= 0 && ub[m+i] >= 0,
            ceil(fmax(-8.0, lb[i])) <= floor(fmin(7.0, ub[i]))
                && lb[m+i] <= 1 && ub[m+i] >= 1};
    }
    __host__ __device__ static bool tight(
        double a, double b, double atol, double rtol) {
        return fabs(a-b) <= atol + rtol*fmax(fabs(a), fabs(b));
    }
    struct Choice {
        double value, cost;
        bool off, on;
    };
    __host__ __device__ static Choice choice(
        int i, int m, const double* g, const double* lb, const double* ub,
        const double* x, double atol, double rtol) {
        const Domain d = domain(i, m, lb, ub);
        double lo = d.lo, hi = d.hi;
        bool off = d.off, on = d.on;
        if (x) {
            const double y = x[m+i];
            if (tight(y, 0, atol, rtol)) on = false;
            if (tight(y, 1, atol, rtol)) off = false;
            // Keep all active inequalities of the node hull, including
            // its sloping faces. Fixing coordinate bounds alone is wrong.
            if (on && tight(x[i], d.lo*y, atol, rtol)) hi = d.lo;
            if (on && tight(x[i], d.hi*y, atol, rtol)) lo = d.hi;
            on = on && lo <= hi;
        }
        const double value = on ? (g[i] > 0 ? lo : hi) : 0;
        return {value, g[i]*value + g[m+i], off, on};
    }

    bool check_feasible(const Node& node, const std::vector<int>&) const {
        const int m = static_cast<int>(node.lb.size())/2;
        int used = 0;
        for (int i = 0; i < m; ++i) {
            const Domain d = domain(i, m, node.lb.data(), node.ub.data());
            if (!d.off && !d.on) return false;
            used += !d.off;
        }
        return used <= k;
    }

    void oracle_cpu(const std::vector<double>& g,
                    const std::vector<double>& lb, const std::vector<double>& ub,
                    std::vector<double>& v, const double* x = nullptr,
                    double atol = 1e-9, double rtol = 1e-9) const {
        const int m = static_cast<int>(g.size())/2;
        v.assign(g.size(), 0);
        std::vector<Choice> choices(m);
        std::vector<int> eligible;
        int used = 0;
        const bool full = x && tight(std::accumulate(x+m, x+2*m, 0.0),
                                     k, atol, rtol);
        for (int i = 0; i < m; ++i) {
            const Choice c = choice(i, m, g.data(), lb.data(), ub.data(), x, atol, rtol);
            if (!c.off && !c.on) throw std::runtime_error("empty LMO face");
            choices[i] = c;
            if (!c.off) { v[i] = c.value; v[m+i] = 1; ++used; }
            else if (c.on && (full || c.cost < 0)) eligible.push_back(i);
        }
        if (used > k || (full && used + static_cast<int>(eligible.size()) < k))
            throw std::runtime_error("empty cardinality face");
        std::sort(eligible.begin(), eligible.end(), [&](int a, int b) {
            return choices[a].cost < choices[b].cost
                || (choices[a].cost == choices[b].cost && a < b);
        });
        for (int i : eligible) {
            if (used == k) break;
            v[i] = choices[i].value; v[m+i] = 1; ++used;
        }
    }
    void compute_v_cpu(const std::vector<double>& g,
                       const std::vector<double>& lb, const std::vector<double>& ub,
                       std::vector<double>& v) const {
        oracle_cpu(g, lb, ub, v);
    }

    __device__ double optional_cost(int i, int m, const double* g,
        const double* lb, const double* ub, const double* x,
        bool full, double atol, double rtol) const {
        if (i >= m) return CUDART_INF;
        const Choice c = choice(i,m,g,lb,ub,x,atol,rtol);
        return c.off && c.on && (full || c.cost < 0) ? c.cost : CUDART_INF;
    }

    // Large nodes sort a permutation in the output workspace. A second
    // numeric sort restores coordinate order after tagging the selected IDs.
    // SortSize <= 2*m, so the padded permutation fits in the n=2*m outputs.
    __device__ void oracle_large(int tid, int m, const double* g,
        const double* lb, const double* ub, double* v, const double* x,
        bool full, int remaining, double atol, double rtol) const {
        for (int pos = tid; pos < SortSize; pos += blockDim.x) v[pos] = pos;
        __syncthreads();
        for (int pass = 0; pass < 2; ++pass) {
            for (int width = 2; width <= SortSize; width *= 2) {
                for (int stride = width/2; stride > 0; stride /= 2) {
                    for (int left = tid; left < SortSize; left += blockDim.x) {
                        const int right = left ^ stride;
                        if (right <= left) continue;
                        const double li = v[left], ri = v[right];
                        const double lk = pass ? li : optional_cost(
                            static_cast<int>(li),m,g,lb,ub,x,full,atol,rtol);
                        const double rk = pass ? ri : optional_cost(
                            static_cast<int>(ri),m,g,lb,ub,x,full,atol,rtol);
                        const bool after = rk < lk || (rk == lk && ri < li);
                        if (after == ((left & width) == 0)) {
                            v[left] = ri; v[right] = li;
                        }
                    }
                    __syncthreads();
                }
            }
            if (pass == 0) {
                for (int rank = tid; rank < SortSize; rank += blockDim.x) {
                    const int i = static_cast<int>(v[rank]);
                    const bool selected = rank < remaining && optional_cost(
                        i,m,g,lb,ub,x,full,atol,rtol) < CUDART_INF;
                    v[rank] = 2.0*i + selected;
                }
                __syncthreads();
            }
        }
        for (int i = tid; i < m; i += blockDim.x) {
            const bool selected = static_cast<int>(v[i]) & 1;
            const Choice c = choice(i,m,g,lb,ub,x,atol,rtol);
            v[i] = selected || !c.off ? c.value : 0;
            v[m+i] = selected || !c.off ? 1 : 0;
        }
        __syncthreads();
    }

    // Shared storage is bounded: <=24 KiB for small sorts, <=32 KiB for
    // medium sorts, O(1) for large sorts. The solver output is global scratch.
    __device__ void oracle_device(int tid, int n, const double* g,
        const double* lb, const double* ub, double* v,
        const double* x = nullptr, double atol = 1e-9, double rtol = 1e-9) const {
        const int m = n/2;
        double local_mass = 0, mandatory = 0;
        for (int i = tid; i < m; i += blockDim.x) {
            if (x) local_mass += x[m+i];
            mandatory += !choice(i,m,g,lb,ub,x,atol,rtol).off;
        }
        const double mass = lmo_block_sum(local_mass,tid);
        const bool full = x && tight(mass,k,atol,rtol);
        const int remaining = k-static_cast<int>(lmo_block_sum(mandatory,tid));
        if constexpr (SortSize > 8192) {
            oracle_large(tid,m,g,lb,ub,v,x,full,remaining,atol,rtol);
        } else {
            __shared__ int indices[SortSize];
            __shared__ double small_keys[SortSize <= 2048 ? SortSize : 1];
            double* keys = SortSize <= 2048 ? small_keys : v;
            for (int i = tid; i < SortSize; i += blockDim.x) {
                keys[i] = optional_cost(i,m,g,lb,ub,x,full,atol,rtol);
                indices[i] = i;
            }
            __syncthreads();
            for (int width = 2; width <= SortSize; width *= 2) {
                for (int stride = width/2; stride > 0; stride /= 2) {
                    for (int left = tid; left < SortSize; left += blockDim.x) {
                        const int right = left ^ stride;
                        if (right <= left) continue;
                        const double lk = keys[left], rk = keys[right];
                        const int li = indices[left], ri = indices[right];
                        const bool after = rk < lk || (rk == lk && ri < li);
                        if (after == ((left & width) == 0)) {
                            keys[left] = rk; keys[right] = lk;
                            indices[left] = ri; indices[right] = li;
                        }
                    }
                    __syncthreads();
                }
            }
            // All sorted keys have been consumed; v can become the output.
            for (int i = tid; i < m; i += blockDim.x) {
                const Choice c = choice(i,m,g,lb,ub,x,atol,rtol);
                v[i] = c.off ? 0 : c.value;
                v[m+i] = c.off ? 0 : 1;
            }
            __syncthreads();
            for (int rank = tid; rank < remaining && rank < SortSize; rank += blockDim.x) {
                const int i = indices[rank];
                if (optional_cost(i,m,g,lb,ub,x,full,atol,rtol) < CUDART_INF) {
                    const Choice c = choice(i,m,g,lb,ub,x,atol,rtol);
                    v[i] = c.value; v[m+i] = 1;
                }
            }
            __syncthreads();
        }
    }
    __device__ void compute_v(int tid, int n, const double* g,
        const double* lb, const double* ub, double* v) const {
        oracle_device(tid, n, g, lb, ub, v);
    }
    __device__ void compute_inface_v(int tid, int n, const double* g, const double* x,
        const double* lb, const double* ub, double* a,
        double atol = 1e-9, double rtol = 1e-9) const {
        oracle_device(tid, n, g, lb, ub, a, x, atol, rtol);
    }

    // For x-gamma*d, a constraint with slack s decreasing at rate t
    // limits gamma to s/t. Do not omit the z/y coupling inequalities.
    __device__ static double step_limit(double slack, double rate) {
        return rate > 0 ? fmin(1.0, fmax(0.0, slack)/rate) : 1.0;
    }
    __device__ static double coordinate_step(int i, int m,
        const double* direction, const double* x, const double* lb, const double* ub) {
        const Domain d = domain(i, m, lb, ub);
        const double z = x[i], y = x[m+i], dz = direction[i], dy = direction[m+i];
        const double lo = d.on ? d.lo : 0, hi = d.on ? d.hi : 0;
        double step = step_limit(y - (d.off ? 0 : 1), dy);
        step = fmin(step, step_limit((d.on ? 1 : 0) - y, -dy));
        step = fmin(step, step_limit(z - lo*y, dz - lo*dy));
        return fmin(step, step_limit(hi*y - z, hi*dy - dz));
    }
    __device__ double dicg_maximum_step_local(int tid, int n, const double* direction,
        const double* x, const double* lb, const double* ub, double = 1e-6) const {
        double step = 1;
        for (int i = tid; i < n/2; i += blockDim.x)
            step = fmin(step, coordinate_step(i,n/2,direction,x,lb,ub));
        return step;
    }
    __device__ double dicg_sum_maximum_step(int tid, int n,
        const double* direction, const double* x) const {
        double mass = 0, slope = 0;
        for (int i = tid; i < n/2; i += blockDim.x) {
            mass += x[n/2+i]; slope += direction[n/2+i];
        }
        const double total_mass = lmo_block_sum(mass,tid);
        const double total_slope = lmo_block_sum(slope,tid);
        return step_limit(k-total_mass,-total_slope);
    }

    bool candidate_feasible(const std::vector<double>& x,
                            const std::vector<double>& lb,
                            const std::vector<double>& ub, double tol) const {
        if (x.size() != lb.size() || x.size() != ub.size()) return false;
        const int m = static_cast<int>(x.size())/2;
        int count = 0;
        for (int i = 0; i < 2*m; ++i)
            if (!std::isfinite(x[i]) || x[i] != std::round(x[i])
                || x[i] < lb[i]-tol || x[i] > ub[i]+tol) return false;
        for (int i = 0; i < m; ++i) {
            if (x[m+i] != 0 && x[m+i] != 1) return false;
            if (x[i] < -8*x[m+i] || x[i] > 7*x[m+i]) return false;
            count += static_cast<int>(x[m+i]);
        }
        return count <= k;
    }
    std::vector<double> integer_candidate(const std::vector<double>& x,
        const std::vector<double>& lb, const std::vector<double>& ub) const {
        if (x.size() != lb.size() || x.size() != ub.size()
                || x.size()%2 != 0 || k < 0) return {};
        const int m = static_cast<int>(x.size())/2;
        std::vector<double> rounded(x.size(), 0.0);
        std::vector<int> optional;
        int selected = 0;
        for (int i = 0; i < m; ++i) {
            if (!std::isfinite(x[i])) return {};
            const Domain d = domain(i, m, lb.data(), ub.data());
            if (!d.off && !d.on) return {}; // Empty node domain.
            if (!d.on) continue;           // Forced off by node bounds.
            const double z = std::max(d.lo, std::min(d.hi, std::round(x[i])));
            // Forced-on selectors consume capacity even if their weight is zero.
            if (!d.off || z != 0.0) {
                rounded[i] = z;
                rounded[m+i] = 1.0;
                ++selected;
                if (d.off) optional.push_back(i);
            }
        }
        if (selected > k) {
            // Drop the smallest rounded magnitudes; index breaks ties stably.
            std::sort(optional.begin(), optional.end(), [&](int a, int b) {
                const double aa = std::abs(rounded[a]), bb = std::abs(rounded[b]);
                return aa < bb || (aa == bb && a < b);
            });
            for (int i : optional) {
                if (selected <= k) break;
                rounded[i] = rounded[m+i] = 0.0;
                --selected;
            }
        }
        if (!candidate_feasible(rounded, lb, ub, 0.0)) return {};
        return rounded;
    }
};

// H=XX', d=z-w: f=d H d', grad_z=2dH, grad_y=0.
// Store [H,w] contiguously for the solver's persisting-L2 access window.
// The selection variables never enter a matrix product.
struct QuantizationObjective {
    static constexpr bool supports_quadratic_exact = true;
    static constexpr int n_vars = Vars;
    static constexpr int block_threads = Weights < 256 ? ((Weights+31)/32)*32 : 256;
    static constexpr int DeltaTile = 128;
    // This contract sizes per-node GLOBAL workspace, not shared memory.
    static constexpr int shared_bytes = 2*Weights*sizeof(double) + 16;
    const double* d_coefficients;
    const double* h_coefficients;

    const void* l2_window_base() const { return d_coefficients; }
    size_t l2_window_bytes() const { return size_t(Weights*Weights+Weights)*sizeof(double); }

    double f(const std::vector<double>& x) const {
        std::array<double,Weights> delta, product;
        const double* w = h_coefficients+Weights*Weights;
        std::transform(x.begin(),x.begin()+Weights,w,delta.begin(),std::minus<double>());
        host_matvec(h_coefficients,delta.data(),product.data());
        return host_dot(Weights,delta.data(),product.data());
    }
    void gradient_host(const std::vector<double>& x, std::vector<double>& g) const {
        std::array<double,Weights> delta;
        const double* w = h_coefficients+Weights*Weights;
        std::transform(x.begin(),x.begin()+Weights,w,delta.begin(),std::minus<double>());
        g.assign(Vars,0);
        host_matvec(h_coefficients,delta.data(),g.data(),2);
    }
    __device__ static double* buffer(unsigned char* workspace) {
        const uintptr_t p = (reinterpret_cast<uintptr_t>(workspace)+15u) & ~uintptr_t(15u);
        return reinterpret_cast<double*>(p);
    }
    __device__ void prepare_block(int, unsigned char*) const {}

    // Each thread computes an output coordinate. Adjacent threads read
    // adjacent H entries; delta is broadcast from a fixed 1 KiB shared tile.
    // Block-stride batches handle longer vectors without growing shared memory.
    __device__ void multiply(int tid, const double* x, unsigned char* workspace, bool direction = false) const {
        __shared__ double tile[DeltaTile];
        const double* H = d_coefficients;
        const double* w = H+Weights*Weights;
        double* delta = buffer(workspace);
        double* product = delta+Weights;
        for (int i = tid; i < Weights; i += blockDim.x) delta[i] = direction ? x[i] : x[i]-w[i];
        __syncthreads();
        for (int base = 0; base < Weights; base += blockDim.x) {
            const int j = base+tid;
            double sum = 0;
            for (int start = 0; start < Weights; start += DeltaTile) {
                const int count = Weights-start < DeltaTile ? Weights-start : DeltaTile;
                for (int k = tid; k < count; k += blockDim.x) tile[k] = delta[start+k];
                __syncthreads();
                if (j < Weights)
                    for (int k = 0; k < count; ++k)
                        sum = fma(tile[k],H[(start+k)*Weights+j],sum);
                __syncthreads();
            }
            if (j < Weights) product[j] = sum;
        }
        __syncthreads();
    }
    __device__ void grad(int tid, const double* x, double* g, unsigned char* workspace) const {
        multiply(tid,x,workspace);
        const double* product = buffer(workspace)+Weights;
        for (int i = tid; i < Vars; i += blockDim.x)
            g[i] = i < Weights ? 2*product[i] : 0;
        __syncthreads();
    }
    __device__ double value_from_factor_local(int tid, unsigned char* workspace) const {
        const double* delta = buffer(workspace);
        const double* product = delta+Weights;
        double value = 0;
        for (int i = tid; i < Weights; i += blockDim.x)
            value = fma(delta[i],product[i],value);
        return value;
    }
    __device__ double quadratic_curvature_local(
        int tid, const double* dir, unsigned char* workspace) const {
        multiply(tid, dir, workspace, true);
        return 2.0 * value_from_factor_local(tid, workspace);
    }
    __device__ double value_local(int tid, const double* x, unsigned char* workspace) const {
        multiply(tid,x,workspace);
        return value_from_factor_local(tid,workspace);
    }
};

struct DeviceBuffer {
    double* p = nullptr;
    explicit DeviceBuffer(size_t count) {
        const auto e = cudaMalloc(&p, count*sizeof(double));
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    }
    explicit DeviceBuffer(const std::vector<double>& x) : DeviceBuffer(x.size()) {
        auto e = cudaMemcpy(p, x.data(), x.size()*sizeof(double), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (p) cudaFree(p); }
};

struct BlasHandle {
    cublasHandle_t handle = nullptr;
    BlasHandle() {
        check(cublasCreate(&handle));
    }
    static void check(cublasStatus_t status) {
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS failed, status=" + std::to_string(status));
    }
    BlasHandle(const BlasHandle&) = delete;
    BlasHandle& operator=(const BlasHandle&) = delete;
    ~BlasHandle() { cublasDestroy(handle); }
};

struct Instance {
    int samples;
    std::vector<double> w, X, coefficients;
    Instance(unsigned seed, int n, const std::string& input = "") : samples(n), w(Weights), X(static_cast<size_t>(Weights)*n),
                                    coefficients(Weights*Weights+Weights) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> weights(-8,7);
        std::normal_distribution<double> inputs(0,1);
        if (input.empty()) {
            for (double& a : w) a = weights(rng);
            for (double& a : X) a = inputs(rng);
        } else {
            // Text format: m samples, then w/scale, then row-major scale*X.
            // This preserves the original physical reconstruction loss.
            std::ifstream stream(input);
            int m = 0, count = 0;
            if (!(stream >> m >> count) || m != Weights || count != samples)
                throw std::runtime_error("input dimensions must match compiled Weights and --samples");
            for (auto* values : {&w, &X})
                for (double& value : *values)
                    if (!(stream >> value) || !std::isfinite(value))
                        throw std::runtime_error("invalid or truncated input");
            std::string extra;
            if (stream >> extra) throw std::runtime_error("unexpected trailing input");
        }
        std::puts("Building H=XX^T on GPU (cuBLAS, FP64)...");
        std::fflush(stdout);
        const auto started = std::chrono::steady_clock::now();
        {
            DeviceBuffer dX(X), dH(static_cast<size_t>(Weights)*Weights);
            BlasHandle blas;
            const double alpha = 1.0, beta = 0.0;
            // Row-major X (Weights x n) is column-major A (n x Weights).
            // Compute the upper triangle of A^T A; it appears as the lower
            // triangle in the host row-major buffer. Mirror for exact symmetry.
            BlasHandle::check(cublasDsyrk(blas.handle, CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_T, Weights, n, &alpha, dX.p, n, &beta, dH.p, Weights));
            const auto status = cudaMemcpy(coefficients.data(), dH.p,
                static_cast<size_t>(Weights)*Weights*sizeof(double), cudaMemcpyDeviceToHost);
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        }
        for (int i = 0; i < Weights; ++i)
            for (int j = 0; j < i; ++j)
                coefficients[j*Weights+i] = coefficients[i*Weights+j];
        std::printf("H ready in %.3f s (GPU setup, transfers and symmetry included).\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count());
        std::fflush(stdout);
        std::copy(w.begin(),w.end(),coefficients.begin()+Weights*Weights);
    }
    double residual(const std::vector<double>& z) const {
        std::vector<double> residuals(samples,0);
        for (int i = 0; i < Weights; ++i) {
            const double delta = z[i]-w[i];
            for (int s = 0; s < samples; ++s) residuals[s] += delta*X[i*samples+s];
        }
        return host_dot(samples,residuals.data(),residuals.data());
    }
    double enumerate(int k) const {
        // Explicit verification is only practical for small support budgets.
        long double combinations = 1, levels = 1, count = 1;
        for (int j = 1; j <= k; ++j) {
            combinations *= static_cast<long double>(Weights-j+1)/j;
            levels *= 15;
            count += combinations*levels;
        }
        if (count > 1e7L)
            throw std::runtime_error("--verify exceeds 10 million candidates; reduce k or dimensions");
        std::vector<double> z(Vars,0);
        double best = std::numeric_limits<double>::infinity();
        std::function<void(int,int)> visit = [&](int i, int used) {
            if (i == Weights) { best = std::min(best,residual(z)); return; }
            for (int q = -8; q <= 7; ++q) {
                if (used + (q != 0) > k) continue;
                z[i] = q;
                visit(i+1,used+(q != 0));
            }
        };
        visit(0,0);
        return best;
    }
};

inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}


} // namespace quantization_sparse

