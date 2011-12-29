#define TEXCUT_BLOCK_SIZE 4
#define QUANTIZATION_LEVEL SHRT_MAX//16384.0f
#define QUANTIZATION_LEVEL_HARF QUANTIZATION_LEVEL/2
#define GRADIENT_HETEROGENUITY 1000

#include "graph.h"
typedef Graph<int,int,int> TexCutGraph;
typedef unsigned char BGPixType;
