#include "sys/types.h"
#include "sys/sysinfo.h"
#include <iostream>
#include <vector>
#include "minipool.hpp"

#include "stdlib.h"
#include "stdio.h"
#include "string.h"

int parseLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}

int getValue(){ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmSize:", 7) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}

struct Empty1024 {
    float ary[256];
};
struct KiloByte {
    Empty1024 array[1024];
};
struct MegaByte {
    KiloByte array[1024];
};

int main(int argc, char const *argv[])
{
    //allocate 128MB
    minipool<KiloByte> pool (1024);

    std::vector<KiloByte*> refs;

    for(unsigned int i = 0; i < 1024*4; ++i){
        auto pointer = pool.alloc();
        // if(i > 1024*7)
        //     pool.free(pointer);
        refs.push_back(pointer);
    }

    struct sysinfo memInfo;

    sysinfo (&memInfo);
    long long totalVirtualMem = memInfo.totalram;
    //Add other values in next statement to avoid int overflow on right hand side...
    //totalVirtualMem += memInfo.totalswap;
    totalVirtualMem *= memInfo.mem_unit;

    std::cout<< totalVirtualMem << "  "<< getValue() << std::endl;

    getchar();
    
    //refs.resize(1025);
    //std::vector<KiloByte*> refs2(refs.end() - 1025, refs.end());
    for(auto pointer : refs)
    {
        pool.free(pointer);
    }

    std::cout<< totalVirtualMem << "  "<< getValue() << std::endl;

    getchar();

    pool.garbage_collect();

    std::cout<< totalVirtualMem << "  "<< getValue() << std::endl;

    getchar();
    pool.alloc();
    getchar();
    return 0;
}
