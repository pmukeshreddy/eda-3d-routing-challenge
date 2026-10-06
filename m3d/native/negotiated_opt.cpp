// C++17, standard library only. Private text input / JSON output protocol.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
using namespace std;
using Clock = chrono::steady_clock;
using Cost = long long;
struct Timeout {};
struct WorkLimit {};
struct Route { vector<pair<int,int>> edges; vector<int> vertices; Cost delay=0; };
struct Net { int id; vector<int> pins; Cost lower=0; };
struct Stats {
    long long attempted_groups=0, reroute_orders=0, completed_orders=0, timed_out_orders=0;
    long long bounded_orders=0, accepted_improvements=0, accepted_equal=0, accepted_worse=0;
    long long legal_candidates=0, single_net_improvements=0, single_net_gain=0, work_steps=0;
    long long expansions=0;
};
template<class Timer=Clock>
struct SearchWithClock {
    using Clock=Timer;
    int w,h,l,wh,size,via; vector<int> layer; vector<Net> nets;
    vector<Route> current,best; Cost current_delay=0,best_delay=0,initial_delay=0;
    vector<int> pin_owner, owners;
    vector<vector<pair<int,int>>> adj;
    mt19937 rng; Stats stats; typename Clock::time_point deadline;
    double budget, setup_seconds=0, search_seconds=0; long long steps_limit;
    long long order_left=-1; string stop_reason="time_budget";
    vector<double> dist; vector<int> previous, stamps; int stamp=0;
    vector<Route> relaxed; vector<char> relaxed_ready;

    void deadline_check() { if(Clock::now()>=deadline) throw Timeout{}; }
    void tick() {
        ++stats.expansions;
        if(order_left>=0 && --order_left<0) throw WorkLimit{};
        if((stats.expansions&255)==0) deadline_check();
    }
    static uint32_t hash(int v,uint32_t seed) {
        uint32_t x=uint32_t(v)^seed; x^=x>>16; x*=0x7feb352dU;
        x^=x>>15; x*=0x846ca68bU; return x^(x>>16);
    }
    int weight(int a,int b) const {return abs(a-b)==wh && a%wh==b%wh ? via : layer[a/wh];}
    void read(istream& in) {
        uint32_t seed; int n;
        if(!(in>>w>>h>>l>>via>>n>>seed>>budget>>steps_limit) || w<=0||h<=0||l<=0||n<0)
            throw runtime_error("invalid worker header");
        wh=w*h; size=wh*l; rng.seed(seed); layer.resize(l);
        for(auto &x:layer) in>>x;
        nets.resize(n); current.resize(n); pin_owner.assign(size,-1);
        for(int i=0;i<n;++i) {
            int k; in>>nets[i].id>>k; nets[i].pins.resize(k);
            for(auto &p:nets[i].pins) { in>>p; if(p<0||p>=size) throw runtime_error("bad pin"); pin_owner[p]=i; }
            int ne; in>>ne; current[i].edges.resize(ne);
            for(auto &e:current[i].edges) in>>e.first>>e.second;
        }
        if(!in) throw runtime_error("truncated input");
        adj.resize(size);
        for(int v=0;v<size;++v) {
            int x=v%w,y=(v/w)%h,z=v/wh;
            if(x+1<w) adj[v].push_back({v+1,layer[z]});
            if(x>0) adj[v].push_back({v-1,layer[z]});
            if(y+1<h) adj[v].push_back({v+w,layer[z]});
            if(y>0) adj[v].push_back({v-w,layer[z]});
            if(z+1<l) adj[v].push_back({v+wh,via});
            if(z>0) adj[v].push_back({v-wh,via});
        }
        for(int i=0;i<n;++i) {
            if(!validate_route(i,current[i])) throw runtime_error("invalid initial tree");
            current_delay+=current[i].delay;
            int a=nets[i].pins[0], ax=a%w, ay=(a/w)%h, az=a/wh;
            for(size_t k=1;k<nets[i].pins.size();++k) {
                int b=nets[i].pins[k], xy=abs(ax-b%w)+abs(ay-(b/w)%h);
                Cost bound=numeric_limits<Cost>::max();
                for(int z=0;z<l;++z) bound=min(bound,Cost(xy)*layer[z]+Cost(via)*(abs(az-z)+abs(b/wh-z)));
                nets[i].lower+=bound;
            }
        }
        if(!validate(current)) throw runtime_error("invalid initial capacity");
        best=current; best_delay=initial_delay=current_delay; refresh();
        dist.resize(size); previous.resize(size); stamps.assign(size,0);
        relaxed.resize(n); relaxed_ready.assign(n,0);
    }
    // Independent of construction: exact tree, adjacency, foreign-pin and pin checks.
    bool validate_route(int nid, Route &r) const {
        vector<int> index(size,-1); vector<int> vertices=nets[nid].pins;
        for(auto [a,b]:r.edges) {
            if(a<0||b<0||a>=size||b>=size) return false;
            bool adjacent=false;
            for(auto [v,c]:adj[a]) if(v==b) { adjacent=true; break; }
            if(!adjacent) return false;
            vertices.push_back(a); vertices.push_back(b);
        }
        sort(vertices.begin(),vertices.end()); vertices.erase(unique(vertices.begin(),vertices.end()),vertices.end());
        if(r.edges.size()+1!=vertices.size()) return false;
        vector<vector<pair<int,int>>> graph(vertices.size());
        for(size_t i=0;i<vertices.size();++i) {
            int v=vertices[i]; if(pin_owner[v]>=0&&pin_owner[v]!=nid) return false;
            index[v]=int(i);
        }
        for(auto [a,b]:r.edges) {
            graph[index[a]].push_back({index[b],weight(a,b)});
            graph[index[b]].push_back({index[a],weight(a,b)});
        }
        vector<Cost> distance(vertices.size(),-1); int root=index[nets[nid].pins[0]];
        vector<int> queue{root}; distance[root]=0;
        for(size_t i=0;i<queue.size();++i) {
            int u=queue[i]; for(auto [v,c]:graph[u]) if(distance[v]<0) {distance[v]=distance[u]+c; queue.push_back(v);}
        }
        if(queue.size()!=vertices.size()) return false;
        r.delay=0; for(size_t i=1;i<nets[nid].pins.size();++i) r.delay+=distance[index[nets[nid].pins[i]]];
        r.vertices=std::move(vertices); return true;
    }
    bool validate(vector<Route>& routes) const {
        if(routes.size()!=nets.size()) return false;
        vector<int> occupied(size,-1);
        for(size_t i=0;i<routes.size();++i) {
            if(!validate_route(int(i),routes[i])) return false;
            for(int v:routes[i].vertices) { if(occupied[v]>=0) return false; occupied[v]=int(i); }
        }
        return true;
    }
    void refresh() { owners.assign(size,-1); for(size_t n=0;n<current.size();++n) for(int v:current[n].vertices) owners[v]=int(n); }
    struct Item {double d; uint32_t tie; int v; bool operator>(const Item& o) const {if(d!=o.d)return d>o.d; if(tie!=o.tie)return tie>o.tie;return v>o.v;} };
    // Negative root_weight selects exact single-source cleanup. Otherwise grow
    // branches seeded by physical driver distance, charging congestion only on
    // new extensions. Weight 1 prices delay; smaller weights favor sharing.
    bool route(int nid, const vector<char>& blocked, const vector<int>& occ,
               const vector<double>& history, double pressure, uint32_t salt,
               double root_weight, Route& output) {
        deadline_check();
        bool branches=root_weight>=0;
        vector<Cost> root_distance(size,0);
        auto &pins=nets[nid].pins; vector<char> tree(size,0), pending(size,0);
        int remaining=0; for(size_t i=1;i<pins.size();++i) if(pins[i]!=pins[0]&&!pending[pins[i]]) { pending[pins[i]]=1; ++remaining; }
        tree[pins[0]]=1; output=Route{};
        if(++stamp==numeric_limits<int>::max()) {fill(stamps.begin(),stamps.end(),0);stamp=1;}
        priority_queue<Item,vector<Item>,greater<Item>> queue;
        int root=pins[0]; dist[root]=0; previous[root]=-1; stamps[root]=stamp;
        queue.push({0,hash(root,salt),root});
        auto attach=[&](int sink) {
            int v=sink; vector<int> branch;
            while(!tree[v]) {
                branch.push_back(v); v=previous[v];
                if(v<0) throw runtime_error("broken predecessor");
            }
            for(auto it=branch.rbegin();it!=branch.rend();++it) {
                int u=*it; root_distance[u]=root_distance[v]+weight(v,u);
                output.edges.push_back(minmax(v,u)); tree[u]=1;
                if(branches) {
                    dist[u]=root_weight*root_distance[u]; previous[u]=-1;
                    queue.push({dist[u],hash(u,salt),u});
                }
                v=u;
            }
        };
        vector<int> found;
        while(!queue.empty() && remaining>0) {
            auto item=queue.top(); queue.pop(); tick(); int u=item.v;
            if(item.d!=dist[u]) continue;
            if(pending[u]) {
                pending[u]=0; --remaining;
                if(branches) {attach(u); continue;}
                found.push_back(u);
                if(!remaining) break;
            }
            for(auto [v,c]:adj[u]) {
                if(blocked[v] || tree[v] || (pin_owner[v]>=0&&pin_owner[v]!=nid)) continue;
                double cost=item.d+c+history[v]+pressure*occ[v];
                if(stamps[v]!=stamp||cost<dist[v]) {
                    dist[v]=cost; stamps[v]=stamp; previous[v]=u;
                    queue.push({cost,hash(v,salt),v});
                }
            }
        }
        if(remaining) return false;
        for(int sink:found) attach(sink);
        sort(output.edges.begin(),output.edges.end());
        if(!validate_route(nid,output)) throw runtime_error("search produced invalid tree");
        return true;
    }
    bool consider(vector<Route> candidate, double temperature, bool single=false, bool counted=false) {
        if(!validate(candidate)) throw runtime_error("candidate violated legality");
        if(!counted) ++stats.legal_candidates;
        Cost total=0; for(auto &r:candidate) total+=r.delay;
        Cost delta=total-current_delay;
        if(delta>0) {
            if(temperature<=0||delta>max(2.0,0.01*current_delay)) return false;
            double uniform=(double(rng())+0.5)/4294967296.0;
            if(uniform>=exp(-double(delta)/temperature)) return false;
        }
        if(delta<0) {++stats.accepted_improvements; if(single){++stats.single_net_improvements;stats.single_net_gain-=delta;}}
        else if(delta>0) ++stats.accepted_worse; else ++stats.accepted_equal;
        current=std::move(candidate); current_delay=total; refresh();
        if(total<best_delay) {best=current;best_delay=total;}
        return true;
    }
    void cleanup(int nid,uint32_t salt) {
        vector<char> blocked(size,0); for(int v=0;v<size;++v) if(owners[v]>=0&&owners[v]!=nid) blocked[v]=1;
        vector<int> occ(size,0); vector<double> history(size,0); Route r;
        if(route(nid,blocked,occ,history,0,salt,-1,r)) {
            auto candidate=current; candidate[nid]=std::move(r); consider(std::move(candidate),0,true);
        }
    }
    vector<int> group(int target, uint32_t salt) {
        vector<char> included(nets.size(),0); included[target]=1; vector<int> result{target},frontier{target};
        vector<char> blocked(size,0); vector<int> occ(size,0); vector<double> history(size,0);
        for(int depth=0;depth<2;++depth) {
            vector<int> next;
            for(int nid:frontier) {
                // Periodically change equal-cost relaxed trees to expose different blockers.
                if(!relaxed_ready[nid] || stats.attempted_groups%7==0) {
                    if(!route(nid,blocked,occ,history,0,salt,-1,relaxed[nid])) throw runtime_error("no relaxed route");
                    relaxed_ready[nid]=1;
                }
                vector<int> hits(nets.size(),0);
                for(int v:relaxed[nid].vertices) if(owners[v]>=0&&!included[owners[v]]) ++hits[owners[v]];
                vector<int> blockers; for(size_t i=0;i<hits.size();++i) if(hits[i]) blockers.push_back(int(i));
                sort(blockers.begin(),blockers.end(),[&](int a,int b){return hits[a]!=hits[b]?hits[a]>hits[b]:a<b;});
                for(int other:blockers) if(!included[other]) { included[other]=1;result.push_back(other);next.push_back(other); }
            }
            frontier=std::move(next);
        }
        if(result.size()>16) {
            // Large closures become a regional neighborhood: target plus EVERY
            // route intersecting a cube centered on an obstructed relaxed path.
            // Shrink the cube, never truncate its set of intersecting nets.
            vector<int> intersections;
            for(int v:relaxed[target].vertices) if(owners[v]>=0&&owners[v]!=target) intersections.push_back(v);
            if(intersections.empty()) return vector<int>{target};
            int center=intersections[salt%intersections.size()];
            int cx=center%w,cy=(center/w)%h,cz=center/wh;
            for(int radius: {6,3,1,0}) {
                fill(included.begin(),included.end(),0); included[target]=1; result={target};
                for(int z=max(0,cz-radius);z<=min(l-1,cz+radius);++z)
                    for(int y=max(0,cy-radius);y<=min(h-1,cy+radius);++y)
                        for(int x=max(0,cx-radius);x<=min(w-1,cx+radius);++x) {
                            int owner=owners[(z*h+y)*w+x];
                            if(owner>=0&&!included[owner]) {included[owner]=1;result.push_back(owner);}
                        }
                if(result.size()<=16) break;
            }
        }
        return result;
    }
    // Each replacement is exact and atomic: even an interrupted pass leaves
    // the last fully legal candidate available for objective comparison.
    void polish(vector<Route>& trial,const vector<int>& members,uint32_t salt) {
        vector<char> blocked(size,0); vector<int> occ(size,0);
        vector<double> history(size,0);
        for(auto &r:trial) for(int v:r.vertices) blocked[v]=1;
        auto order=members;
        sort(order.begin(),order.end(),[&](int a,int b) {
            Cost da=trial[a].delay-nets[a].lower,db=trial[b].delay-nets[b].lower;
            return da!=db?da>db:a<b;
        });
        for(int nid:order) {
            if(trial[nid].delay==nets[nid].lower) continue;
            for(int v:trial[nid].vertices) blocked[v]=0;
            Route r;
            if(!route(nid,blocked,occ,history,0,salt+uint32_t(nid),-1,r))
                throw runtime_error("legal polish became disconnected");
            if(r.delay>trial[nid].delay) throw runtime_error("exact polish worsened delay");
            trial[nid]=std::move(r);
            for(int v:trial[nid].vertices) blocked[v]=1;
        }
    }
    bool reroute(const vector<int>& members,const vector<int>& order,int target,
                 uint32_t salt,int variant,vector<Route>& candidate) {
        vector<char> inside(nets.size(),0),blocked(size,0);
        for(int n:members) inside[n]=1;
        for(int v=0;v<size;++v) if(owners[v]>=0&&!inside[owners[v]]) blocked[v]=1;
        vector<int> occ(size,0); vector<double> history(size,0);
        vector<Route> trial=current;
        // Warm repair is another order variant; fresh orders rip up the entire group.
        for(int n:members) { if(variant>=3) for(int v:trial[n].vertices) ++occ[v]; else trial[n]=Route{}; }
        double pressure=8;
        for(int iteration=0;iteration<14;++iteration) {
            for(int nid:order) {
                bool conflict=trial[nid].vertices.empty() || (iteration==0 && nid==target);
                if(!conflict) for(int v:trial[nid].vertices) if(occ[v]>1) {conflict=true;break;}
                if(!conflict && (iteration>0 || variant==4)) continue;
                for(int v:trial[nid].vertices) --occ[v];
                Route r;
                double root_weight=1.0;
                if(!route(nid,blocked,occ,history,(variant==4 && iteration==0 && nid==target)?0:pressure,salt+uint32_t(iteration*7919+nid),root_weight,r)) return false;
                trial[nid]=std::move(r); for(int v:trial[nid].vertices) ++occ[v];
            }
            bool legal=true;
            for(int v=0;v<size;++v) if(occ[v]>1) {legal=false;history[v]+=8*(occ[v]-1);}
            if(legal) {
                if(!validate(trial)) throw runtime_error("group construction violated legality");
                candidate=std::move(trial); return true;
            }
            pressure*=1.7; deadline_check();
        }
        return false;
    }
    void run() {
        auto start=Clock::now(); deadline=start+chrono::duration_cast<typename Clock::duration>(chrono::duration<double>(budget));
        auto full_deadline=deadline;
        if(steps_limit<0 && budget>=0.05)
            deadline-=chrono::duration_cast<typename Clock::duration>(chrono::duration<double>(min(2.0,0.05*budget)));
        vector<int> ranked(nets.size()); iota(ranked.begin(),ranked.end(),0);
        auto rank=[&](){sort(ranked.begin(),ranked.end(),[&](int a,int b){Cost da=current[a].delay-nets[a].lower,db=current[b].delay-nets[b].lower;return da!=db?da>db:a<b;});};
        rank(); vector<int> cleanup_order=ranked; size_t cursor=0;
        try {
            while(steps_limit<0||stats.work_steps<steps_limit) {
                deadline_check(); if(nets.empty()){stop_reason="delay_lower_bound_reached";break;}
                uint32_t salt=rng();
                if(stats.work_steps<(long long)nets.size()) cleanup(cleanup_order[stats.work_steps],salt);
                else {
                    if(cursor==0) rank();
                    int target=ranked[cursor]; cursor=(cursor+1)%ranked.size();
                    cleanup(target,salt);
                    ++stats.attempted_groups; auto members=group(target,salt);
                    if(members.size()>1) {
                        vector<Route> best_trial; Cost trial_delay=numeric_limits<Cost>::max();
                        for(int variant=0;variant<(stats.attempted_groups%16==0?5:4);++variant) {
                            auto neighborhood=members;
                            if(variant==4) {
                                neighborhood={target};
                                for(size_t n=0;n<nets.size();++n) if(int(n)!=target) neighborhood.push_back(int(n));
                                sort(neighborhood.begin()+1,neighborhood.end(),[&](int a,int b) {
                                    Cost da=current[a].delay-nets[a].lower,db=current[b].delay-nets[b].lower;
                                    return da!=db?da>db:a<b;
                                });
                            }
                            auto order=neighborhood;
                            if(variant==1) reverse(order.begin(),order.end());
                            if(variant==2) shuffle(order.begin(),order.end(),rng);
                            ++stats.reroute_orders;
                            order_left=min(variant==4?8000000LL:2000000LL,max(25000LL,1LL*size*max(8,int(neighborhood.size())*2)));
                            vector<Route> trial; bool expired=false;
                            try {
                                if(reroute(neighborhood,order,target,salt,variant,trial)) {
                                    ++stats.legal_candidates;
                                    polish(trial,neighborhood,salt);
                                }
                                ++stats.completed_orders;
                            } catch(const WorkLimit&) {++stats.bounded_orders;}
                              catch(const Timeout&) {++stats.timed_out_orders;expired=true;}
                            order_left=-1;
                            // A raw or partially polished legal trial survives a
                            // later timeout, as does the best earlier order.
                            if(!trial.empty()) {
                                if(!validate(trial)) throw runtime_error("invalid polished candidate");
                                Cost delay=0; for(auto &r:trial) delay+=r.delay;
                                if(delay<trial_delay) {trial_delay=delay;best_trial=std::move(trial);}
                            }
                            if(expired) break;
                        }
                        order_left=-1;
                        if(!best_trial.empty()) consider(std::move(best_trial),max(1.0,0.002*initial_delay)*pow(0.995,double(stats.attempted_groups)),false,true);
                    }
                }
                ++stats.work_steps;
            }
            if(steps_limit>=0&&stats.work_steps>=steps_limit) stop_reason="work_limit";
        } catch(const Timeout&) {stop_reason="time_budget";}
        // Reserve a bounded tail for the retained best, since group moves may
        // have opened shorter hard-obstacle paths for outside nets.
        if(steps_limit<0 && budget>=0.05) {
            deadline=full_deadline; order_left=-1;
            auto trial=best;
            try {polish(trial,ranked,rng());} catch(const Timeout&) {}
            consider(std::move(trial),0);
        }
        search_seconds=chrono::duration<double>(Clock::now()-start).count();
        if(!validate(best)) throw runtime_error("invalid best snapshot");
    }
    void write(ostream& out) const {
        out<<"{\"delay\":"<<best_delay<<",\"stats\":{";
#define FIELD(x) out<<"\"" #x "\":"<<stats.x<<",";
        FIELD(attempted_groups) FIELD(reroute_orders) FIELD(completed_orders) FIELD(timed_out_orders)
        FIELD(bounded_orders) FIELD(accepted_improvements) FIELD(accepted_equal) FIELD(accepted_worse)
        FIELD(legal_candidates) FIELD(single_net_improvements) FIELD(single_net_gain) FIELD(work_steps) FIELD(expansions)
#undef FIELD
        out<<"\"native_setup_seconds\":"<<setup_seconds<<",\"native_search_seconds\":"<<search_seconds<<",\"stop_reason\":\""<<stop_reason<<"\"},\"routes\":[";
        for(size_t n=0;n<best.size();++n) {
            if(n) out<<",";out<<"{\"net\":"<<nets[n].id<<",\"edges\":[";
            for(size_t e=0;e<best[n].edges.size();++e) {if(e)out<<",";out<<"["<<best[n].edges[e].first<<","<<best[n].edges[e].second<<"]";}
            out<<"]}";
        }
        out<<"]}\n";
    }
};
using Search=SearchWithClock<>;
#ifndef NEGOTIATED_OPT_NO_MAIN
int main() {
    try {ios::sync_with_stdio(false);cin.tie(nullptr);auto start=Clock::now();Search s;s.read(cin);s.setup_seconds=chrono::duration<double>(Clock::now()-start).count();s.run();s.write(cout);}
    catch(const exception& e) {cerr<<e.what()<<"\n";return 2;}
}
#endif
