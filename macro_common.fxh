#ifndef _MACRO_COMMON_
#define _MACRO_COMMON_

#define _STR(a) # a
#define STR(a) _STR(a)

#define _CAT(a,b) a##b
#define CAT(a,b) _CAT(a,b)

#define ADDRESS(a) AddressU = a; AddressV = a; AddressW = a
#define FILTER(a)  MagFilter = a; MinFilter = a; MipFilter = a

#endif // _MACRO_COMMON