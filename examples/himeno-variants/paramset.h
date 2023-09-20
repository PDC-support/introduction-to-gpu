// This header is in common with each implementation (except original one) to provide
// compile-time parametric matrix size.

#if defined(SSMALL)
#define ROWS 32
#define COLS 32
#define DEPS 64
#elif defined(SMALL)
#define ROWS 64
#define COLS 64
#define DEPS 128
#elif defined(MIDDLE)
#define ROWS 128
#define COLS 128
#define DEPS 256
#elif defined(LARGE)
#define ROWS 256
#define COLS 256
#define DEPS 512
#elif defined(ELARGE)
#define ROWS 512
#define COLS 512
#define DEPS 1024
#else
#error Missing size, expecting "SSMALL", "SMALL", "MIDDLE", "LARGE" or "ELARGE".
#endif

// Common to all sizes
#define OMEGA 0.8
