//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//  radix-2 single channel pixelshader c2c fft header for ReShade.
//
//                                              Oct.3.2020 by kingeric1992
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ifndef _FFT_C2C_FXH_
#define _FFT_C2C_FXH_

#ifndef FFT_SRC_SIZEX
#   error "undefined FFT_SRC_SIZEX"
#endif
#ifndef FFT_SRC_SIZEY
#   error "undefined FFT_SRC_SIZEY"
#endif
#ifndef FFT_SRC_GET
#   error "undefined FFT_SRC_GET(x,y)"
#endif
#ifndef FFT_SRC_INV
#   define FFT_SRC_INV(a,b) FFT_SRC_GET(a,b)
#endif

#include "macro_common.fxh"
#include "macro_bitop.fxh"

namespace fft_c2c {

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  setup
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#define FFT_SRC_POTX CONST_LOG2(FFT_SRC_SIZEX)
#define FFT_SRC_POTY CONST_LOG2(FFT_SRC_SIZEY)

texture2D texBufHA  { Width = FFT_SRC_SIZEX / 2; Height = FFT_SRC_SIZEY; Format = RGBA32F; };
texture2D texBufHB  { Width = FFT_SRC_SIZEX / 2; Height = FFT_SRC_SIZEY; Format = RGBA32F; };

// we can ditch these if we use verticle fft instead of transpose the input
texture2D texBufVA  { Width = FFT_SRC_SIZEY / 2; Height = FFT_SRC_SIZEX; Format = RGBA32F; };
texture2D texBufVB  { Width = FFT_SRC_SIZEY / 2; Height = FFT_SRC_SIZEX; Format = RGBA32F; };

// mixed name for easier macro setup (address mode doesn't affact tex.Load())
sampler2D sampBufHA { Texture = texBufHB; FILTER(POINT); };
sampler2D sampBufHB { Texture = texBufHA; FILTER(POINT); };
sampler2D sampBufVA { Texture = texBufVB; FILTER(POINT); };
sampler2D sampBufVB { Texture = texBufVA; FILTER(POINT); };

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Util
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//
// raw input -> bit reverse -> butterfly -> output
//
//  forward flow
//   │
//   │  r0 output layout:         rN output layout (last)
//   │     0 ║ 2 ║ 4 ║ 6 ║ 8 ║ ...
//   │     1 ║ 3 ║ 5 ║ 7 ║ 9 ║ ...
//   ▼
//      r1 output layout:              rN-1 output layout
//         0 │ 1 ║ 4 │ 5 ║  8 │  9 ║ ...
//         2 │ 3 ║ 6 │ 7 ║ 10 │ 11 ║ ...
//
//      r2 output layout               rN-2 output layout
//         0 │ 1 │ 2 │ 3 ║  8 │  9 │ 10 │ 11 ║ ...
//         4 │ 5 │ 6 │ 7 ║ 12 │ 13 │ 14 │ 15 ║ ...
//                                                           ▲
//      rN output layout (last)           r0 output layout   │
//         0   │ 1       │ 2       │ ... │ 2^N - 1     ║     │
//         2^N │ 2^N + 1 │ 2^N + 2 │ ... │ 2^(N+1) - 1 ║     │
//                                                           │
//                                                invsersed flow

// fetch sample by xPos
// output pos -> sampleID -> input pos
// TODO: #1 varify inverse flow is properly configured
float4 tex2DfetchID( sampler2D texIn, float4 vpos, float r, float dir) {
    vpos.x += trunc(vpos.x/r)*r;
    int2 id = float2(vpos.x,vpos.x+r); // working id

    // access working id in src buffer
    r      *= dir; // src r. forward_dir = .5, inverse_dir = 2;
    vpos.xw = trunc(id/(r*2))*r + id%r;
    id      = trunc(id/r) % 2;

    return float4(
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
float atan2(float2 v) { return atan2(v.x,v.y); }

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  getters. get the result of either forward or inversed flow
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//  Src Texture Layout:
//           vpos.y
//   0 on .xy  ↓         h/2 on .zw
//      ┌─────┬───┬──────┐┌─────────────────┐
//      │     ¦   ¦                         │
//      ├─────┼───┼──────┤├─────────────────┤
//      │     │ p │                         │ <- vpos.x
//      ├─────┼───┼──────┤├─────────────────┤
//      │     ¦   ¦                         │
//      │     ¦   ¦                         │
//      │     ¦   ¦                         │
//      │     ¦   ¦                         │
//      │     ¦   ¦                         │
//      └─────┴───┴──────┘└─────────────────┘
//
#if (FFT_SRC_POTY & 1)
#   define sampBufV sampBufVB
#else
#   define sampBufV sampBufVA
#endif
// get fft (real, img) from buffer directly without additional pass
float2 get( int2 pos ) {
    float4 vpos = float4(pos,0,FFT_SRC_SIZEY*.5);
    vpos.y %= vpos.w, pos.y /= vpos.w;
    return float2x2(tex2Dfetch(sampBufV,vpos.yxzz))[pos.y];
}
#undef  sampBufV
float  amp( int2 pos)       { return length(get(pos)); }
float  phase( int2 pos)     { return atan2(get(pos).yx); }
// de-scramble sample
float2 getInv(int2 pos)     { return get(pos).yx; }
float  ampInv( int2 pos)    { return length(getInv(pos)); }
float  phaseInv( int2 pos)  { return atan2(getInv(pos).yx); }

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  shaders
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

float4 vs_fft( uint vid : SV_VERTEXID ) : SV_POSITION {
    return float4((vid.xx == uint2(2,1)? float2(-3,3):float2(1,-1)), 0,1);
}
// main fft pass.
float4 ps_fft( sampler2D texIn, int4 vpos, float r) {
    float4 c = tex2DfetchID(texIn, vpos, r=exp2(r),.5); // p0,p1
    c.zw = mul(c.zw,twiddle(vpos.x%r,r));               // twiddle
    return float4(c.xy+c.zw, c.xy-c.zw);                // butterfly
}

// c2c first pass
float4 ps_hori(float4 vpos ) {
    vpos.x  = trunc(vpos.x)*2;
    vpos.w  = rBit(vpos.x+1, FFT_SRC_POTX); // bitReverse( o0 ) = p0
    vpos.x  = rBit(vpos.x,   FFT_SRC_POTX); // bitReverse( o1 ) = p1
    float4 c;
    c.xy = (FFT_SRC_GET(vpos.x,vpos.y));    // p0
    c.zw = (FFT_SRC_GET(vpos.w,vpos.y));    // p1
    //c.zw = mul(c.zw,twiddle(vpos.x%1,1)); // twiddle (always 1)
    return float4( c.xy+c.zw, c.xy-c.zw);   // butterfly
}
float4 ps_hori_inv(float4 vpos ) {
    vpos.x  = trunc(vpos.x)*2;
    vpos.w  = rBit(vpos.x+1, FFT_SRC_POTX); // bitReverse( o0 ) = p0
    vpos.x  = rBit(vpos.x,   FFT_SRC_POTX); // bitReverse( o1 ) = p1
    float4 c;
    c.xy = (FFT_SRC_INV(vpos.x,vpos.y)).yx; // p0
    c.zw = (FFT_SRC_INV(vpos.w,vpos.y)).yx; // p1
    //c.zw = mul(c.zw,twiddle(vpos.x%1,1)); // twiddle (always 1)
    return float4( c.xy+c.zw, c.xy-c.zw)*.5; // butterfly
}

//  (src buffer content)
//           vpos.y
//   0 on .xy   ↓         w/2 on .zw
//      ┌─────┬──┬──────┐┌───────────────┐
//      ├─────┼──┼──────┤├───────────────┤
//      │     │p0│                       │ <- rBit(vpos.x, h)
//      ├─────┼──┼──────┤├───────────────┤
//      │     ¦  ¦                       │
//      │     ¦  ¦                       │
//      │     ¦  ¦                       │
//      ├─────┼──┼──────┤├───────────────┤
//      │     │p1│                       │ <— rBit(vpos.x+1, h)
//      ├─────┼──┼──────┤├───────────────┤
//      └─────┴──┴──────┘└───────────────┘
//
float4 ps_vert( sampler2D texIn, float4 vpos) {
    vpos.x  = trunc(vpos.x)*2;
    vpos.w  = rBit(vpos.x+1, FFT_SRC_POTY);             // src index for p1
    vpos.x  = rBit(vpos.x,   FFT_SRC_POTY);             // src index for p0

    int hh = (FFT_SRC_SIZEX*.5), sel = vpos.y / hh;
    vpos.y %= hh;

    float4 p;
    p.xy = float2x2(tex2Dfetch(texIn, vpos.yxzz))[sel]; // p0
    p.zw = float2x2(tex2Dfetch(texIn, vpos.ywzz))[sel]; // p1
    return float4(p.xy+p.zw,p.xy-p.zw);
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Pass Setup
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ps ctor; < t: output buffer > < r: stage number (0 index) >
#define PS_FFT( t, r, z ) \
float4 ps_fft_##t##r  ( float4 vpos : SV_POSITION) : SV_TARGET { return ps_fft( samp##t, vpos, r); } \
float4 ps_ifft_##t##r ( float4 vpos : SV_POSITION) : SV_TARGET { return ps_fft( samp##t, vpos, r)*.5; }

// horizontal init pass
float4 ps_fft_BufHA0  ( float4 vpos : SV_POSITION) : SV_TARGET { return ps_hori(vpos); }
float4 ps_ifft_BufHA0 ( float4 vpos : SV_POSITION) : SV_TARGET { return ps_hori_inv(vpos); }
// vertical init pass
#if (FFT_SRC_POTX & 1)
#   define sampBufH sampBufHB
#else
#   define sampBufH sampBufHA
#endif
float4 ps_fft_BufVA0  (float4 vpos : SV_POSITION) : SV_TARGET {return ps_vert(sampBufH,vpos);}
float4 ps_ifft_BufVA0 (float4 vpos : SV_POSITION) : SV_TARGET {return ps_vert(sampBufH,vpos)*.5;}
#undef sampBufH

// intermediate passes
PS_FFT(BufHB,  1, FFT_SRC_POTX)   PS_FFT(BufVB,  1, FFT_SRC_POTY)
PS_FFT(BufHA,  2, FFT_SRC_POTX)   PS_FFT(BufVA,  2, FFT_SRC_POTY)
PS_FFT(BufHB,  3, FFT_SRC_POTX)   PS_FFT(BufVB,  3, FFT_SRC_POTY)
PS_FFT(BufHA,  4, FFT_SRC_POTX)   PS_FFT(BufVA,  4, FFT_SRC_POTY)
PS_FFT(BufHB,  5, FFT_SRC_POTX)   PS_FFT(BufVB,  5, FFT_SRC_POTY)
PS_FFT(BufHA,  6, FFT_SRC_POTX)   PS_FFT(BufVA,  6, FFT_SRC_POTY)
PS_FFT(BufHB,  7, FFT_SRC_POTX)   PS_FFT(BufVB,  7, FFT_SRC_POTY)
PS_FFT(BufHA,  8, FFT_SRC_POTX)   PS_FFT(BufVA,  8, FFT_SRC_POTY)
PS_FFT(BufHB,  9, FFT_SRC_POTX)   PS_FFT(BufVB,  9, FFT_SRC_POTY)
PS_FFT(BufHA, 10, FFT_SRC_POTX)   PS_FFT(BufVA, 10, FFT_SRC_POTY)
PS_FFT(BufHB, 11, FFT_SRC_POTX)   PS_FFT(BufVB, 11, FFT_SRC_POTY)
PS_FFT(BufHA, 12, FFT_SRC_POTX)   PS_FFT(BufVA, 12, FFT_SRC_POTY)
#undef PS_FFT
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  technique
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#include "macro_cond.fxh"
#define COND_PASS( a,b,e,d,c ) COND( a##d { b (e,  d); }, c , d )
#undef  COND_EXP
#define COND_EXP(k)     pass k
#define PASS_FFT(t,r)   VertexShader=vs_fft; PixelShader=ps_fft_##t##r;  RenderTarget=tex##t
#define PASS_IFFT(t,r)  VertexShader=vs_fft; PixelShader=ps_ifft_##t##r; RenderTarget=tex##t

#ifdef FFT_USE_FORWARD
//compile-time resolve pass layout
technique forward {
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
#undef FFT_USE_FORWARD
#endif
#ifdef FFT_USE_INVERSE
technique inverse {

#   define EVAL_EXP FFT_SRC_POTX
#   include "macro_eval.fxh"   // evaluate preprocessor expression
    COND_PASS( hori, PASS_IFFT, BufHA,  0, EVAL_RES) // ps_ifft_BufHA0
    COND_PASS( hori, PASS_IFFT, BufHB,  1, EVAL_RES)
    COND_PASS( hori, PASS_IFFT, BufHA,  2, EVAL_RES)
    COND_PASS( hori, PASS_IFFT, BufHB,  3, EVAL_RES)
    COND_PASS( hori, PASS_IFFT, BufHA,  4, EVAL_RES)
    COND_PASS( hori, PASS_IFFT, BufHB,  5, EVAL_RES)
    COND_PASS( hori, PASS_IFFT, BufHA,  6, EVAL_RES)
    COND_PASS( hori, PASS_IFFT, BufHB,  7, EVAL_RES)
    COND_PASS( hori, PASS_IFFT, BufHA,  8, EVAL_RES)
    COND_PASS( hori, PASS_IFFT, BufHB,  9, EVAL_RES)
    COND_PASS( hori, PASS_IFFT, BufHA, 10, EVAL_RES)
    COND_PASS( hori, PASS_IFFT, BufHB, 11, EVAL_RES)
    COND_PASS( hori, PASS_IFFT, BufHA, 12, EVAL_RES) // 8192
#   undef   EVAL_RES
#   undef   EVAL_EXP

#   define  EVAL_EXP FFT_SRC_POTY
#   include "macro_eval.fxh"
    COND_PASS( vert, PASS_IFFT, BufVA,  0, EVAL_RES)
    COND_PASS( vert, PASS_IFFT, BufVB,  1, EVAL_RES)
    COND_PASS( vert, PASS_IFFT, BufVA,  2, EVAL_RES)
    COND_PASS( vert, PASS_IFFT, BufVB,  3, EVAL_RES)
    COND_PASS( vert, PASS_IFFT, BufVA,  4, EVAL_RES)
    COND_PASS( vert, PASS_IFFT, BufVB,  5, EVAL_RES)
    COND_PASS( vert, PASS_IFFT, BufVA,  6, EVAL_RES)
    COND_PASS( vert, PASS_IFFT, BufVB,  7, EVAL_RES)
    COND_PASS( vert, PASS_IFFT, BufVA,  8, EVAL_RES)
    COND_PASS( vert, PASS_IFFT, BufVB,  9, EVAL_RES)
    COND_PASS( vert, PASS_IFFT, BufVA, 10, EVAL_RES)
    COND_PASS( vert, PASS_IFFT, BufVB, 11, EVAL_RES)
    COND_PASS( vert, PASS_IFFT, BufVA, 12, EVAL_RES) // 8192
#   undef   EVAL_RES
#   undef   EVAL_EXP
}
#undef FFT_USE_INVERSE
#endif
} // fft_c2c
// clean up
#undef COND_PASS
#undef PASS_FFT
#undef PASS_IFFT
#undef FFT_SRC_POTX
#undef FFT_SRC_POTY

#endif // _FFT_C2C_FXH_
