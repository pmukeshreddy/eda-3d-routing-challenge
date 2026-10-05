// Block 1: single-net source-rooted SPT and Held/Perner Algorithm 1 adaptation.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
using Delay = int64_t;
using Key = uint64_t;
using Edge = std::pair<int, int>;
using Real = long double;

static Key edge_key(int a, int b) {
    if (a > b) std::swap(a, b);
    return (Key(uint32_t(a)) << 32) | uint32_t(b);
}

struct Merge {
    int u, v, representative;
    Delay weight_u, weight_v;
    Real distance;
};
struct Candidate {
    std::string method, status = "unreachable";
    std::vector<Edge> edges;
    std::vector<Merge> merges;
    Delay delay = 0;
    Real price = 0, objective = 0;
    uint64_t expansions = 0;
};
struct Result {
    Candidate selected;
    std::vector<Candidate> candidates;
    uint64_t expansions = 0;
    bool exhausted = false;
};
struct Search {
    std::string status;
    int hit = -1;
    Real distance = 0;
};

class NativeEngine {
    int w, h, layers, wh, n;
    std::vector<Delay> layer_delay;
    Delay via_delay;
    // These buffers and binary-heap storage are reused across every search/call.
    std::vector<Delay> integer_dist;
    std::vector<Real> real_dist;
    std::vector<int> predecessor;
    std::vector<uint32_t> stamp;
    uint32_t epoch = 0;
    std::vector<std::pair<Delay, int>> integer_heap;
    std::vector<std::pair<Real, int>> real_heap;
    std::vector<unsigned char> forbidden;
    std::unordered_set<Key> blocked_edges;
    std::unordered_map<Key, Real> prices;
    bool priced = false;
    std::mutex mutex;

    template<class F> void neighbors(int u, F visit) const {
        int z = u / wh, r = u % wh, y = r / w, x = r % w;
        if (x + 1 < w) visit(u + 1, layer_delay[z]);
        if (x > 0) visit(u - 1, layer_delay[z]);
        if (y + 1 < h) visit(u + w, layer_delay[z]);
        if (y > 0) visit(u - w, layer_delay[z]);
        if (z + 1 < layers) visit(u + wh, via_delay);
        if (z > 0) visit(u - wh, via_delay);
    }
    Real price(Key key) const {
        auto it = prices.find(key);
        return it == prices.end() ? 0 : it->second;
    }
    Delay delay(int a, int b) const {
        return a / wh == b / wh ? layer_delay[a / wh] : via_delay;
    }
    void begin_search() {
        if (++epoch == 0) {
            std::fill(stamp.begin(), stamp.end(), 0);
            epoch = 1;
        }
    }
    template<class Cost>
    Search search(int source, const std::unordered_set<int>& targets, bool first,
                  Delay weight, uint64_t& work, uint64_t limit,
                  const std::unordered_set<Key>* allowed = nullptr) {
        begin_search();
        auto& dist = [&]() -> auto& {
            if constexpr (std::is_same_v<Cost, Delay>) return integer_dist;
            else return real_dist;
        }();
        auto& heap = [&]() -> auto& {
            if constexpr (std::is_same_v<Cost, Delay>) return integer_heap;
            else return real_heap;
        }();
        heap.clear();
        auto compare = std::greater<std::pair<Cost, int>>();
        dist[source] = 0;
        stamp[source] = epoch;
        predecessor[source] = -1;
        heap.emplace_back(0, source);
        size_t remaining = targets.size();
        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(), compare);
            Cost distance = heap.back().first;
            int u = heap.back().second;
            heap.pop_back();
            if (distance != dist[u]) continue;
            if (work == limit) return {"budget_exhausted"};
            ++work;
            if (targets.count(u) && (first || --remaining == 0))
                return {"success", u, Real(distance)};
            neighbors(u, [&](int v, Delay d) {
                Key key = edge_key(u, v);
                if (forbidden[v] || blocked_edges.count(key) || (allowed && !allowed->count(key))) return;
                Cost cost;
                if constexpr (std::is_same_v<Cost, Delay>) {
                    if (d > (std::numeric_limits<Delay>::max() - distance) / weight)
                        throw std::overflow_error("search delay overflow");
                    cost = distance + weight * d;
                } else {
                    cost = distance + Real(weight) * d + price(key);
                    if (!std::isfinite(cost)) throw std::overflow_error("search price overflow");
                }
                if (stamp[v] != epoch || cost < dist[v]) {
                    stamp[v] = epoch;
                    dist[v] = cost;
                    predecessor[v] = u;
                    heap.emplace_back(cost, v);
                    std::push_heap(heap.begin(), heap.end(), compare);
                }
            });
        }
        return {"unreachable"};
    }
    Search weighted_search(int source, const std::unordered_set<int>& targets,
                           bool first, Delay weight, uint64_t& work, uint64_t limit) {
        if (!priced) return search<Delay>(source, targets, first, weight, work, limit);
        return search<Real>(source, targets, first, weight, work, limit);
    }
    std::vector<Edge> path(int target) const {
        std::vector<Edge> edges;
        for (int v = target; predecessor[v] != -1; v = predecessor[v])
            edges.emplace_back(std::min(v, predecessor[v]), std::max(v, predecessor[v]));
        return edges;
    }
    // Recover only destination paths. One predecessor per vertex ensures no cycles;
    // stopping at an already recovered branch also prunes every unused leaf.
    std::vector<Edge> tree(const std::vector<int>& sinks) const {
        std::set<Edge> edges;
        for (int sink : sinks) {
            for (int v = sink; predecessor[v] != -1; v = predecessor[v]) {
                Edge edge = {std::min(v, predecessor[v]), std::max(v, predecessor[v])};
                if (!edges.insert(edge).second) break;
            }
        }
        return {edges.begin(), edges.end()};
    }
    void measure(Candidate& candidate, int source, const std::vector<int>& sinks) {
        std::unordered_map<int, std::vector<std::pair<int, Delay>>> adjacency;
        candidate.price = 0;
        for (auto [a, b] : candidate.edges) {
            Delay d = delay(a, b);
            adjacency[a].emplace_back(b, d);
            adjacency[b].emplace_back(a, d);
            candidate.price += price(edge_key(a, b));
        }
        std::unordered_map<int, Delay> distance = {{source, 0}};
        std::vector<int> queue = {source};
        for (size_t i = 0; i < queue.size(); ++i) {
            int u = queue[i];
            for (auto [v, d] : adjacency[u]) {
                if (distance.count(v)) continue;
                if (distance[u] > std::numeric_limits<Delay>::max() - d)
                    throw std::overflow_error("tree path delay overflow");
                distance[v] = distance[u] + d;
                queue.push_back(v);
            }
        }
        if (distance.size() != adjacency.size() || candidate.edges.size() + 1 != distance.size())
            throw std::logic_error("internal error: physical output is not a connected tree");
        candidate.delay = 0;
        for (int sink : sinks) {
            if (!distance.count(sink)) throw std::logic_error("internal error: missing sink");
            if (candidate.delay > std::numeric_limits<Delay>::max() - distance[sink])
                throw std::overflow_error("total tree delay overflow");
            candidate.delay += distance[sink];
        }
        candidate.objective = Real(candidate.delay) + candidate.price;
        if (candidate.objective > std::numeric_limits<double>::max())
            throw std::overflow_error("objective exceeds the exposed floating-point range");
        candidate.status = "success";
    }
    Candidate shortest(int source, const std::vector<int>& sinks, uint64_t& work, uint64_t limit) {
        Candidate result;
        result.method = "shortest_path";
        auto search_result = weighted_search(source, {sinks.begin(), sinks.end()}, false, 1, work, limit);
        result.status = search_result.status;
        if (result.status == "success") {
            result.edges = tree(sinks);
            measure(result, source, sinks);
        }
        return result;
    }
    // Exact Algorithm 1 merge selection, with beta=0 and weights initially one.
    // min over directed searches using c+w(u)*d equals Eq. (5): the direction
    // from the smaller-weight endpoint realizes c+min(w(u),w(v))*d.
    Candidate cost_distance(int source, const std::vector<int>& sinks, uint64_t seed,
                            uint64_t& work, uint64_t limit) {
        Candidate result;
        result.method = "cost_distance";
        std::map<int, Delay> active;
        for (int sink : sinks) active[sink] = 1;
        std::unordered_set<Key> candidate_union;
        std::mt19937_64 random(seed);
        while (!active.empty()) {
            std::unordered_set<int> targets = {source};
            for (auto [u, weight] : active) { (void)weight; targets.insert(u); }
            Real best = std::numeric_limits<Real>::infinity();
            int best_u = -1, best_v = -1;
            std::vector<Edge> best_path;
            for (auto [u, weight] : active) {
                targets.erase(u);
                auto found = weighted_search(u, targets, true, weight, work, limit);
                targets.insert(u);
                if (found.status != "success") {
                    result.status = found.status;
                    return result; // Never expose an unfinished union as a tree.
                }
                if (found.distance < best) {
                    best = found.distance;
                    best_u = u;
                    best_v = found.hit;
                    best_path = path(found.hit);
                }
            }
            Delay wu = active.at(best_u), wv = best_v == source ? 0 : active.at(best_v);
            int representative = source;
            if (best_v != source) {
                // Unbiased bounded sampling, independent of stdlib distribution details.
                uint64_t bound = uint64_t(wu + wv), threshold = -bound % bound, draw;
                do { draw = random(); } while (draw < threshold);
                representative = draw % bound < uint64_t(wu) ? best_u : best_v;
            }
            result.merges.push_back({best_u, best_v, representative, wu, wv, best});
            for (auto [a, b] : best_path) candidate_union.insert(edge_key(a, b));
            active.erase(best_u);
            if (best_v != source) {
                active.erase(best_v);
                active[representative] = wu + wv;
            }
        }
        // Physical embeddings can overlap/cycle. Extract an exact integer-delay
        // SPT in the union, then prune to the required terminals and rescore.
        auto found = search<Delay>(source, {sinks.begin(), sinks.end()}, false,
                                   1, work, limit, &candidate_union);
        result.status = found.status;
        if (result.status == "success") {
            result.edges = tree(sinks);
            measure(result, source, sinks);
        }
        return result;
    }

public:
    NativeEngine(int width, int height, int count, std::vector<Delay> delays, Delay via)
        : w(width), h(height), layers(count), layer_delay(std::move(delays)), via_delay(via) {
        if (w < 1 || h < 1 || layers < 1 || int64_t(w) * h > INT32_MAX ||
            int64_t(w) * h * layers > INT32_MAX || int(layer_delay.size()) != layers || via < 1 ||
            std::any_of(layer_delay.begin(), layer_delay.end(), [](Delay d) { return d < 1; }))
            throw std::invalid_argument("invalid grid or delays");
        wh = w * h; n = wh * layers;
        integer_dist.resize(n); real_dist.resize(n); predecessor.resize(n);
        stamp.resize(n, 0); forbidden.resize(n, 0);
    }
    Result route(int source, const std::vector<int>& sinks, const std::vector<int>& blocked,
                 const std::vector<Edge>& blocked_input,
                 const std::vector<std::tuple<int, int, double>>& price_input,
                 const std::string& method, uint64_t seed, uint64_t budget) {
        std::lock_guard<std::mutex> lock(mutex);
        auto valid_vertex = [&](int v) {
            if (v < 0 || v >= n) throw std::invalid_argument("vertex out of bounds");
        };
        auto valid_edge = [&](int a, int b) {
            valid_vertex(a); valid_vertex(b);
            int distance = std::abs(a % w - b % w) + std::abs((a % wh) / w - (b % wh) / w)
                           + std::abs(a / wh - b / wh);
            if (distance != 1) throw std::invalid_argument("nonadjacent edge");
        };
        valid_vertex(source);
        if (sinks.empty()) throw std::invalid_argument("missing sinks");
        std::unordered_set<int> unique = {source};
        for (int sink : sinks) {
            valid_vertex(sink);
            if (!unique.insert(sink).second) throw std::invalid_argument("repeated terminal");
        }
        if (method != "shortest_path" && method != "cost_distance" && method != "best")
            throw std::invalid_argument("unknown method");
        std::fill(forbidden.begin(), forbidden.end(), 0);
        for (int v : blocked) { valid_vertex(v); forbidden[v] = 1; }
        blocked_edges.clear(); prices.clear(); priced = false;
        for (auto [a, b] : blocked_input) { valid_edge(a, b); blocked_edges.insert(edge_key(a, b)); }
        for (auto [a, b, cost] : price_input) {
            valid_edge(a, b);
            if (!std::isfinite(cost) || cost < 0) throw std::invalid_argument("invalid edge price");
            if (!prices.emplace(edge_key(a, b), Real(cost)).second)
                throw std::invalid_argument("duplicate edge price");
            priced = priced || cost > 0;
        }
        Result result;
        result.selected.method = method;
        for (int terminal : unique) if (forbidden[terminal]) return result;
        auto attempt = [&](const std::string& name, uint64_t limit) {
            uint64_t before = result.expansions;
            Candidate candidate = name == "shortest_path"
                ? shortest(source, sinks, result.expansions, limit)
                : cost_distance(source, sinks, seed, result.expansions, limit);
            candidate.expansions = result.expansions - before;
            result.exhausted = result.exhausted || candidate.status == "budget_exhausted";
            if (candidate.status == "success" && (result.selected.status != "success" ||
                                                   candidate.objective < result.selected.objective))
                result.selected = candidate;
            result.candidates.push_back(std::move(candidate));
        };
        if (method == "best" && priced) {
            attempt("shortest_path", budget / 2);
            attempt("cost_distance", budget);
        } else {
            attempt(method == "best" ? "shortest_path" : method, budget);
        }
        if (result.selected.status != "success") {
            bool unreachable = std::any_of(result.candidates.begin(), result.candidates.end(),
                [](const Candidate& c) { return c.status == "unreachable"; });
            result.selected.status = unreachable ? "unreachable" : "budget_exhausted";
        }
        return result;
    }
};

static py::dict as_dict(const Result& result) {
    const Candidate& selected = result.selected;
    bool success = selected.status == "success";
    py::dict output;
    output["status"] = selected.status;
    output["method"] = selected.method;
    output["edges"] = selected.edges;
    output["total_delay"] = success ? py::cast(selected.delay) : py::none();
    output["resource_cost"] = success ? py::cast(double(selected.price)) : py::none();
    output["objective"] = success ? py::cast(double(selected.objective)) : py::none();
    output["expansions"] = result.expansions;
    output["budget_exhausted"] = result.exhausted;
    py::list summaries, merges;
    for (const auto& candidate : result.candidates) {
        py::dict summary;
        summary["method"] = candidate.method; summary["status"] = candidate.status;
        summary["expansions"] = candidate.expansions;
        summary["objective"] = candidate.status == "success" ? py::cast(double(candidate.objective)) : py::none();
        summaries.append(summary);
    }
    for (const auto& merge : selected.merges) {
        py::dict entry;
        entry["u"] = merge.u; entry["v"] = merge.v;
        entry["weight_u"] = merge.weight_u; entry["weight_v"] = merge.weight_v;
        entry["representative"] = merge.representative; entry["distance"] = double(merge.distance);
        merges.append(entry);
    }
    output["candidates"] = summaries; output["merges"] = merges;
    return output;
}

PYBIND11_MODULE(_wire_native, module) {
    py::class_<NativeEngine>(module, "NativeEngine")
        .def(py::init<int, int, int, std::vector<Delay>, Delay>())
        .def("route", [](NativeEngine& engine, int source, const std::vector<int>& sinks,
                         const std::vector<int>& blocked, const std::vector<Edge>& edges,
                         const std::vector<std::tuple<int, int, double>>& prices,
                         const std::string& method, uint64_t seed, uint64_t budget) {
            Result result;
            {
                py::gil_scoped_release release;
                result = engine.route(source, sinks, blocked, edges, prices, method, seed, budget);
            }
            return as_dict(result);
        });
}
