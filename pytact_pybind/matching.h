#include <stdio.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <time.h>
#include <thread>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

typedef std::vector<double> vd;
typedef std::vector<std::vector<double>> vvd;

#define MAXN 30
#define MAXM 30
#define MAXNM 900 
#define PI 3.14159265


struct Point_t
{
    double x, y;
    int id;

    Point_t(){}

    Point_t(double x, double y, double id=0){
        this->x = x;
        this->y = y;
        this->id = id;
    }
    bool operator<(Point_t other) const
    {
        return (x < other.x || (x == other.x && y < other.y));
    }

    Point_t operator-(Point_t other) const
    {
        Point_t ret(x-other.x, y-other.y);
        return ret;
    }

    Point_t operator+(Point_t other) const
    {
        Point_t ret(x+other.x, y+other.y);
        return ret;
    }

    Point_t operator/(double other) const
    {
        Point_t ret(x/other, y/other);
        return ret;
    }
};

class Matching{
private:
    double x_0, y_0, dx, dy;

    int Row[MAXNM], Col[MAXNM];
    int Dist[MAXNM][MAXNM], done[MAXN], occupied[MAXN][MAXM], first[MAXN];
    int fps;
    double degree[MAXNM][MAXNM];
    double dmin, dmax, theta;
    double moving_max;
    double cost_threshold, flow_difference_threshold;
    clock_t time_st;

public:
    int n;
    int N, M, NM;
    int flag_record = 1;

    int MinRow[MAXNM], MinCol[MAXNM], MinOccupied[MAXN][MAXM];
    double minf = -1;

    Point_t O[MAXN][MAXM], D[MAXN][MAXM], C[MAXNM], MinD[MAXN][MAXM];
    double K1 = 0.1, K2 = 1;

    Matching(int N_=8, int M_=8, int fps_=30, double x_0 = 80., double y_0 = 15., double dx = 21.0, double dy = 21.0);
    void init(std::vector<std::vector<double>> input);
    int precessor(int i, int j);
    double calc_cost(int i);
    void dfs(int i, double cost, int missing, int spare);
    void run();
    std::tuple<vvd, vvd, vvd, vvd, vvd> get_flow();
    std::tuple<double, double> test();
    double infer();
};