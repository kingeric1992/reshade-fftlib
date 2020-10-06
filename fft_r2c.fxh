//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//  radix-2 single channel pixelshader r2c fft header for ReShade.
//
//                                              Oct.3.2020 by kingeric1992
//  todo: add option to specify transform dir.
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ifndef _FFT_R2C_FXH_
#define _FFT_R2C_FXH_

#ifndef FFT_SRC_SIZEX
#   error "undeined FFT_SRC_SIZEX"
#endif
#ifndef FFT_SRC_SIZEY
#   error "undeined FFT_SRC_SIZEY"
#endif
#ifndef FFT_SRC_GET
#   error "undeined FFT_SRC_GET(x,y)"
#endif

#include "macro_common.fxh"
#include "macro_bitop.fxh"

namespace fft_r2c {

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  setup
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#define FFT_SRC_POTX CONST_LOG2(FFT_SRC_SIZEX)
#define FFT_SRC_POTY CONST_LOG2(FFT_SRC_SIZEY)

texture2D texBufHA  { Width = FFT_SRC_SIZEX / 2; Height = FFT_SRC_SIZEY / 2; Format = RGBA32F; };
texture2D texBufHB  { Width = FFT_SRC_SIZEX / 2; Height = FFT_SRC_SIZEY / 2; Format = RGBA32F; };

// we can ditch these if we use verticle fft instead of transpose the input
texture2D texBufVA  { Width = FFT_SRC_SIZEY / 2; Height = FFT_SRC_SIZEX / 2; Format = RGBA32F; };
texture2D texBufVB  { Width = FFT_SRC_SIZEY / 2; Height = FFT_SRC_SIZEX / 2; Format = RGBA32F; };

// mixed name for easier macro setup (address mode doesn't affact tex.Load())
sampler2D sampBufHA { Texture = texBufHB; FILTER(POINT); };
sampler2D sampBufHB { Texture = texBufHA; FILTER(POINT); };
sampler2D sampBufVA { Texture = texBufVB; FILTER(POINT); };
sampler2D sampBufVB { Texture = texBufVA; FILTER(POINT); };

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Util
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/*
    raw input -> bit reverse -> butterfly -> output

    raw input layout:
        0 | 1 | 2 | 3 | 4 | ...

    r0 output layout:
        0 ║ 2 ║ 4 ║ 6 ║ 8 ║ ...
        1 ║ 3 ║ 5 ║ 7 ║ 9 ║ ...

    r1 output layout:
        0 | 1 ║ 4 | 5 ║  8 |  9 ║ ...
        2 | 3 ║ 6 | 7 ║ 10 | 11 ║ ...

    r2 output layout
        0 | 1 | 2 | 3 ║  8 |  9 | 10 | 11 ║ ...
        4 | 5 | 6 | 7 ║ 12 | 13 | 14 | 15 ║ ...

    rN output layout (last)
        0   | 1       | 2       | ... | 2^N - 1     ║
        2^N | 2^N + 1 | 2^N + 2 | ... | 2^(N+1) - 1 ║
*/
// fetch sample by xPos
// output pos -> sampleID -> input pos
float2x2 tex2DfetchID( sampler2D texIn, float4 vpos, float r) {
    vpos.x += trunc(vpos.x/r)*r;
    int2 id = float2(vpos.x,vpos.x+r); // working id

    // access working id in src buffer
    vpos.xw = trunc(id/r);
    r      *= .5;
    vpos.xw = vpos.xw*r + id%r;
    id      = trunc(id/r) % 2;

    return float2x2(
        float2x2(tex2Dfetch(texIn, vpos.xyzz))[id.x],
        float2x2(tex2Dfetch(texIn, vpos.wyzz))[id.y]
    );
}

// handles 16bits(65535) bit tops
uint rBit( uint v, uint l ) {
    #define R2(n)    n,     n + 2*64,     n + 1*64,     n + 3*64
    #define R4(n) R2(n), R2(n + 2*16), R2(n + 1*16), R2(n + 3*16)
    #define R6(n) R4(n), R4(n + 2*4 ), R4(n + 1*4 ), R4(n + 3*4 )

    static const uint lut[256] = { R6(0), R6(2), R6(1), R6(3) };
    return (( lut[v % 256 ] << 8) + (lut[ v >> 8] )) >> (16 - l);

    #undef R2
    #undef R4
    #undef R6
}

// twiddle factor. (DIT, negative k for DIF)
float2x2 twiddle( float k, float r ) {
    return sincos(-3.1415926*k/r,k,r), float2x2(r,k,-k,r); // cos, sin, -sin, cos
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  shaders
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

float4 vs_fft( uint vid : SV_VERTEXID ) : SV_POSITION {
    return float4((vid.xx == uint2(2,1)? float2(-3,3):float2(1,-1)), 0,1);
}
// main fft pass.
float4 ps_fft( sampler2D texIn, int4 vpos, float r) {
    float2x2 s = tex2DfetchID(texIn, vpos, r = exp2(r));
    s[1] = mul(s[1],twiddle(vpos.x%r,r)); // implicit cmul
    return float4(s[0]+s[1], s[0]-s[1]);
}

// r2c, two rows at once, ->  N/2 row, N column, horizontal N point complex FFT
float4 ps_hori(float4 vpos ) {
    vpos.xy = trunc(vpos.xy)*2;             // 0,1,2,...WH/2 -1 -> 0,2,4, ... WH -2
    vpos.w  = rBit(vpos.x+1, FFT_SRC_POTX);     // bitReverwe( p0 )
    vpos.x  = rBit(vpos.x,   FFT_SRC_POTX);     // bitReverwe( p1 )
    vpos.z  = vpos.y + 1;                   // Second input node

    float4 c;
    c.x = (SRC_GET(vpos.x,vpos.y)), c.z = (SRC_GET(vpos.w,vpos.y)); // Even row on Real
    c.y = (SRC_GET(vpos.x,vpos.z)), c.w = (SRC_GET(vpos.w,vpos.z)); // Odd row on img
    return float4( c.xy+c.zw, c.xy-c.zw);
}

//  (src buffer content)
//           vpos.y                   w - vpos.y
//   0 on .xy   ↓         w/2 on .zw      ↓
//      ┌─────┬──┬──────┐       ┌───────┬───┬───┐
//      ├─────┼──┼──────┤       ├───────┼───┼───┤
//      │     │p0│      │       │       │p0*│   │ <- rBit(vpos.x, h)/2
//      ├─────┼──┼──────┤       ├───────┼───┼───┤    (unpack to even/odd row with p0 & p0*)
//      │               │       │               │
//      │               │       │               │
//      │               │       │               │
//      ├─────┼──┼──────┤       ├───────┼───┼───┤
//      │     │p1│      │       │       │p1*│   │ <— rBit(vpos.x+1, h)/2
//      ├─────┼──┼──────┤       ├───────┼───┼───┤    (unpack to even/odd row with p1 & p1*)
//      └─────┴──┴──────┘       └───────┴───┴───┘
//
float4 ps_vert( sampler2D texIn, float4 vpos) {

    vpos.y  = trunc(vpos.y);
    vpos.x  = trunc(vpos.x)*2;          // [0, h/2-1] * 2
    vpos.w  = rBit(vpos.x+1, FFT_SRC_POTY); // src index for p1
    vpos.x  = rBit(vpos.x,   FFT_SRC_POTY); // src index for p0
    int sel = vpos.x % 2;               // is accessed index on odd or even row.
    vpos.xw = vpos.xw / 2;              // (either both odd or even with bit reversed)

    // vpos.y = [0, w/2-1] < src buffer size X
    float4 X, cX;
    X.xy = tex2Dfetch(texIn, vpos.yxzz).xy; // p0
    X.zw = tex2Dfetch(texIn, vpos.ywzz).xy; // p1

    // complex conjugate
    vpos.y = (FFT_SRC_SIZEX - vpos.y) % (FFT_SRC_SIZEX*.5); // vpos.y = 0 -> 0
    cX.xy  = tex2Dfetch(texIn, vpos.yxzz).zw; // p0*
    cX.zw  = tex2Dfetch(texIn, vpos.ywzz).zw; // p1*
    cX.yw  = -cX.yw;

    // can't use float4x2
    // even: (X+cX)*.5;  odd: i(cX-X) * .5
    float4 p = int(vpos.y)?                     // if vpos.y = 0, pack id(0) & id(w/2)
        (sel? float4(X.yw-cX.yw,cX.xz-X.xz) : (X.xzyw+cX.xzyw)) *.5 :
        (sel? float4(X.yw,cX.yw) : float4(X.xz,cX.xz)); // wrap row w/2 into row 0

    return float4(p.xz+p.yw,p.xz-p.yw);
}

//
//  A[r]    = (HA[r]    + HA[h-r]*) *.5
//  C[r]    = (HA[h-r]* - HA[r]) *.5 *i
//
//  Src Texture Layout:
//
//       posY  0.xy  ...  h/2-1  0.zw  ...  h/2-1
//            ├────────────────┤├────────────────┤
//        ID   0                 h/2
//     ┬ posX ┌────────────────┐┌────────────────┐
//     │  0   │    (HA[r]    + HA[h-r]*) *.5     │ HA
//     │      ├────────────────┤├────────────────┤
//     │      │                ││                │ <┐
//     │      │       HB       ││       HB       │  │ Hermitian
//     │      │                ││                │  │ symmatry on HB
//     ┼      ├────────────────┤├────────────────┤  │ HB [k,r] = HB *[k,w-r]
//     │ w/2  ¦    (HA[h-r]* - HA[r]) *.5 *i     ¦  │
//     │      ¦················¦¦················¦  │
//     │      ¦                ¦¦                ¦  │
//     │      ¦      HB*       ¦¦       HB*      ¦  │
//     │      ¦                ¦¦                ¦ <┘
//     ┴      ····································
//                                         HB[h-1, w-1] = HB*[h-1,1]
//
#if (FFT_SRC_POTY & 1)
#   define sampBufV sampBufVB
#else
#   define sampBufV sampBufVA
#endif
// get fft (real, img) from buffer directly without additional pass
// unpacked M x N ( instead of M x (N/2+1))
float2 get( int2 pos ) {
    int2 hSize = int2(FFT_SRC_SIZEX, FFT_SRC_SIZEY);

    if ( any(pos < 0 || pos >= hSize))
        return 0;

    float4 p = float4(pos,hSize-pos);
    hSize /= 2;

    int3 sel = p.xyw / hSize.xyy; // x-> conj sel, y -> layer sel, z -> conj layer sel
    p.yw %= hSize.y;              // wrap coord

    [flatten]
    if( pos.x == 0 || pos.x == hSize.x) { // on first or w/2 column
        float4 H;
        p.z  = 0;
        H.xy = float2x2(tex2Dfetch(sampBufV, p.yxzz))[sel.y];
        H.zw = float2x2(tex2Dfetch(sampBufV, p.wxzz))[sel.z], H.w = -H.w; // conj
        return (pos.x? float2(H.z+H.y,H.w-H.x):(H.xy+H.zw))*.5;
    }
    p.w = 0;
    float2 res = float2x2(tex2Dfetch(sampBufV, sel.x? p.yzww:p.yxww))[sel.y];
    return float2(res.x, sel.x? -res.y:res.y); // conj
}
#undef  sampBufV

float amp( int2 pos )   { return length(get(pos)); }
float phase( int2 pos ) { return atan2(get(pos).yx); }

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Pass Setup
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ps ctor; < t: output buffer > < r: stage number (0 index) >
#define PS_FFT( t, r ) \
float4 ps_fft_##t##r( float4 vpos : SV_POSITION ) : SV_TARGET { return ps_fft( samp##t, vpos, r ); }

// horizontal pixel shaders
float4 ps_fft_BufHA0 (float4 vpos : SV_POSITION) : SV_TARGET {return ps_hori(vpos);}
PS_FFT(BufHB,  1)
PS_FFT(BufHA,  2)
PS_FFT(BufHB,  3)
PS_FFT(BufHA,  4)
PS_FFT(BufHB,  5)
PS_FFT(BufHA,  6)
PS_FFT(BufHB,  7)
PS_FFT(BufHA,  8)
PS_FFT(BufHB,  9)
PS_FFT(BufHA, 10)
PS_FFT(BufHB, 11)
PS_FFT(BufHA, 12)

// vertical pixel shaders
#if (FFT_SRC_POTX & 1)
#   define sampBufH sampBufHB
#else
#   define sampBufH sampBufHA
#endif
float4 ps_fft_BufVA0 (float4 vpos : SV_POSITION) : SV_TARGET {return ps_vert(sampBufH,vpos);}
#undef sampBufH

PS_FFT(BufVB,  1)
PS_FFT(BufVA,  2)
PS_FFT(BufVB,  3)
PS_FFT(BufVA,  4)
PS_FFT(BufVB,  5)
PS_FFT(BufVA,  6)
PS_FFT(BufVB,  7)
PS_FFT(BufVA,  8)
PS_FFT(BufVB,  9)
PS_FFT(BufVA, 10)
PS_FFT(BufVB, 11)
PS_FFT(BufVA, 12)

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  technique
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#include "macro_cond.fxh"
#define COND_PASS( a,b,e,d,c ) COND( a##d { b (e,  d); }, c , d )
#undef  COND_EXP
#define COND_EXP(k)     pass k
#define PASS_FFT(t,r)   VertexShader=vs_fft; PixelShader=ps_fft_##t##r;  RenderTarget=tex##t

//compile-time resolve pass layout
technique fastNfurious {

#   define EVAL_EXP FFT_SRC_POTX
#   include "macro_eval.fxh"   // evaluate preprocessor expression
    COND_PASS( hori, PASS_FFT, BufHA,  0, EVAL_RES) // ps_fft_BufHA0
    COND_PASS( hori, PASS_FFT, BufHB,  1, EVAL_RES)
    COND_PASS( hori, PASS_FFT, BufHA,  2, EVAL_RES)
    COND_PASS( hori, PASS_FFT, BufHB,  3, EVAL_RES)
    COND_PASS( hori, PASS_FFT, BufHA,  4, EVAL_RES)
    COND_PASS( hori, PASS_FFT, BufHB,  5, EVAL_RES)
    COND_PASS( hori, PASS_FFT, BufHA,  6, EVAL_RES)
    COND_PASS( hori, PASS_FFT, BufHB,  7, EVAL_RES)
    COND_PASS( hori, PASS_FFT, BufHA,  8, EVAL_RES)
    COND_PASS( hori, PASS_FFT, BufHB,  9, EVAL_RES)
    COND_PASS( hori, PASS_FFT, BufHA, 10, EVAL_RES)
    COND_PASS( hori, PASS_FFT, BufHB, 11, EVAL_RES)
    COND_PASS( hori, PASS_FFT, BufHA, 12, EVAL_RES) // 8192
#   undef   EVAL_RES
#   undef   EVAL_EXP

#   define  EVAL_EXP FFT_SRC_POTY
#   include "macro_eval.fxh"
    COND_PASS( vert, PASS_FFT, BufVA,  0, EVAL_RES)
    COND_PASS( vert, PASS_FFT, BufVB,  1, EVAL_RES)
    COND_PASS( vert, PASS_FFT, BufVA,  2, EVAL_RES)
    COND_PASS( vert, PASS_FFT, BufVB,  3, EVAL_RES)
    COND_PASS( vert, PASS_FFT, BufVA,  4, EVAL_RES)
    COND_PASS( vert, PASS_FFT, BufVB,  5, EVAL_RES)
    COND_PASS( vert, PASS_FFT, BufVA,  6, EVAL_RES)
    COND_PASS( vert, PASS_FFT, BufVB,  7, EVAL_RES)
    COND_PASS( vert, PASS_FFT, BufVA,  8, EVAL_RES)
    COND_PASS( vert, PASS_FFT, BufVB,  9, EVAL_RES)
    COND_PASS( vert, PASS_FFT, BufVA, 10, EVAL_RES)
    COND_PASS( vert, PASS_FFT, BufVB, 11, EVAL_RES)
    COND_PASS( vert, PASS_FFT, BufVA, 12, EVAL_RES) // 8192
#   undef   EVAL_RES
#   undef   EVAL_EXP
}
} // fft_r2c
#endif // _FFT_R2C_FXH_
