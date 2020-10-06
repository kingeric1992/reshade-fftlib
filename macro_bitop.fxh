#ifndef _MACRO_BITOP_
#define _MACRO_BITOP_

#define CONST_LOG2(v) ( \
    (((v) & 0xAAAAAAAA) != 0) | \
    ((((v) & 0xFFFF0000) != 0) << 4) | \
    ((((v) & 0xFF00FF00) != 0) << 3) | \
    ((((v) & 0xF0F0F0F0) != 0) << 2) | \
    ((((v) & 0xCCCCCCCC) != 0) << 1))

#define BIT2_LOG2(v)  ( (v) | ( (v) >> 1) )
#define BIT4_LOG2(v)  ( BIT2_LOG2(v) | ( BIT2_LOG2(v) >> 2) )
#define BIT8_LOG2(v)  ( BIT4_LOG2(v) | ( BIT4_LOG2(v) >> 4) )
#define BIT16_LOG2(v) ( BIT8_LOG2(v) | ( BIT8_LOG2(v) >> 8) )
#define BIT32_LOG2(v) ( BIT16_LOG2(v) | ( BIT16_LOG2(v) >> 16) )

#endif // _MACRO_BITOP_