// Unit-level access to the same native search/acceptance methods as production.
#define NEGOTIATED_OPT_NO_MAIN
#include "../m3d/native/negotiated_opt.cpp"
// A deterministic monotonic clock expires only after the second order has
// performed real graph work. No wall-clock race or production timeout hook.
struct OrderClock {
    using duration=Clock::duration;
    using time_point=chrono::time_point<OrderClock>;
    static Stats* stats;
    static long long started_expansions;
    static bool expired;
    static time_point now() {
        if(stats && stats->reroute_orders==2) {
            if(started_expansions<0) started_expansions=stats->expansions;
            else if(stats->expansions>started_expansions) expired=true;
        }
        return time_point(expired?chrono::seconds(100):chrono::seconds(0));
    }
};
Stats* OrderClock::stats=nullptr;
long long OrderClock::started_expansions=-1;
bool OrderClock::expired=false;
struct PolishClock {
    using duration=Clock::duration;
    using time_point=chrono::time_point<PolishClock>;
    static int calls;
    static time_point now() {return time_point(chrono::seconds(calls++?100:0));}
};
int PolishClock::calls=0;
int main(int argc,char**argv) {
    try {
        if(argc!=2) throw runtime_error("expected test mode");
        if(string(argv[1])=="partial_order_timeout") {
            SearchWithClock<OrderClock> timed;
            timed.read(cin); timed.steps_limit=-1;
            OrderClock::stats=&timed.stats;
            timed.run();
            if(timed.stats.completed_orders!=1 || timed.stats.timed_out_orders!=1)
                throw runtime_error("timeout did not follow one completed order");
            if(!OrderClock::expired || timed.stats.expansions<=OrderClock::started_expansions)
                throw runtime_error("second order did not perform partial work");
            timed.write(cout); return 0;
        }
        if(string(argv[1])=="partial_polish_timeout") {
            SearchWithClock<PolishClock> timed;
            timed.read(cin);
            timed.deadline=PolishClock::time_point(chrono::seconds(10));
            auto trial=timed.current;
            bool expired=false;
            try {timed.polish(trial,{0,1},17);} catch(const Timeout&) {expired=true;}
            if(!expired || trial[0].delay>=timed.current[0].delay ||
               trial[1].edges!=timed.current[1].edges)
                throw runtime_error("partial polishing did not preserve legal progress");
            timed.consider(std::move(trial),0);
            timed.write(cout); return 0;
        }
        Search search; search.read(cin);
        search.deadline=Clock::now()+chrono::seconds(20);
        string mode=argv[1];
        if(mode=="global_order") {
            search.stats.attempted_groups=15;
            search.steps_limit=search.nets.size()+1;
            search.run();
            if(search.stats.reroute_orders!=5)
                throw runtime_error("periodic global repair order was not exercised");
        } else if(mode=="root_distance" || mode=="sunk_congestion") {
            vector<char> blocked(search.size,0);
            vector<int> occ(search.size,0);
            vector<double> history(search.size,0);
            if(mode=="root_distance") {
                blocked[2*search.w+1]=blocked[2*search.w+2]=1;
            } else {
                for(int x=1;x<=4;++x) blocked[search.w+x]=1;
                occ[1]=1;
                Route weighted;
                if(!search.route(0,blocked,occ,history,3,17,-1,weighted) || weighted.delay!=12)
                    throw runtime_error("weighted-root comparison no longer distinguishes sunk cost");
            }
            Route route;
            if(!search.route(0,blocked,occ,history,3,17,1.0,route))
                throw runtime_error("root-distance route failed");
            search.consider({route},0);
        } else if(mode=="polish") {
            auto trial=search.current;
            search.polish(trial,{0,1},17);
            search.consider(std::move(trial),0);
        } else if(mode=="warm_preserves") {
            vector<Route> trial;
            if(!search.reroute({0,1},{0,1},0,17,4,trial))
                throw runtime_error("warm route failed");
            if(trial[1].delay!=search.current[1].delay)
                throw runtime_error("warm repair rebuilt an unaffected route");
            search.consider(std::move(trial),0);
        } else if(mode=="outside") {
            vector<Route> trial;
            if(!search.reroute({0},{0},0,17,0,trial)) throw runtime_error("outside route failed");
            search.consider(std::move(trial),0);
        } else if(mode=="retention") {
            auto initial=search.current;
            Search improved; improved.read(cin);
            if(!search.consider(improved.current,0)) throw runtime_error("improvement rejected");
            if(!search.consider(initial,1e9)) throw runtime_error("uphill move rejected");
            if(search.current_delay<=search.best_delay) throw runtime_error("no uphill move happened");
            // A subsequent interrupted trial must also leave the best intact.
            search.deadline=Clock::now()-chrono::seconds(1);
            try {vector<Route> ignored;search.reroute({0,1},{0,1},0,17,0,ignored);throw runtime_error("missed timeout");}
            catch(const Timeout&) {}
        } else throw runtime_error("unknown mode");
        search.write(cout);
    } catch(const exception& e) {cerr<<e.what()<<"\n";return 2;}
}
