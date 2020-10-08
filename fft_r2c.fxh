//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//  radix-2 single channel pixelshader r2c fft header for ReShade.
//
//                                              Oct.3.2020 by kingeric1992
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
#ifndef FFT_SRC_INV
#   define FFT_SRC_INV(a,b) FFT_SRC_GET(a,b)
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
//     │  0   │  A[r] = (HA[r] + HA[h-r]*)*.5    │ HA
//     │      ├────────────────┤├────────────────┤
//     │      │                ││                │ <┐
//     │      │       HB       ││       HB       │  │ Hermitian
//     │      │                ││                │  │ symmatry on HB
//     ┼      ├────────────────┤├────────────────┤  │ HB [k,r] = HB *[k,w-r]
//     │ w/2  ¦  C[r] = (HA[h-r]* - HA[r])*.5i   ¦  │
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
float  amp( int2 pos)       { return length(get(pos)); }
float  phase( int2 pos)     { return atan2(get(pos).yx); }
// de-scramble sample
float2 getInv(int2 pos)     { return pos.y = rBit(pos.y, FFT_SRC_POTY), get(pos); }
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
    float4 c = tex2DfetchID(texIn, vpos, r=exp2(r), .5);    // p0,p1
    c.zw = mul(c.zw,twiddle(vpos.x%r,r));                   // twiddle
    return float4(c.xy+c.zw, c.xy-c.zw);                    // butterfly
}
float4 ps_fft_inv( sampler2D texIn, int4 vpos, float r) {
    float4 c = tex2DfetchID(texIn, vpos, r=exp2(r), 2);     // p0,p1
           c = float4(c.xy+c.zw, c.xy-c.zw);                // butterfly
    return c.zw = mul(c.zw,twiddle(-(vpos.x%r),r)), c*.5;   // twiddle
}


// r2c, two rows at once, ->  N/2 row, N column, horizontal N point complex FFT
float4 ps_hori(float4 vpos ) {
    vpos.xy = trunc(vpos.xy)*2;                         // 0,1,2,...WH/2 -1 -> 0,2,4, ... WH -2
    vpos.z  = rBit(vpos.x+1, FFT_SRC_POTX);             // bitReverse( p0 )
    vpos.x  = rBit(vpos.x,   FFT_SRC_POTX);             // bitReverse( p1 )
    vpos.w  = vpos.y + 1;                               // odd row
    float4 c = float4(                                  // Even row on Real, Odd row on Img
        FFT_SRC_GET(vpos.x,vpos.y), FFT_SRC_GET(vpos.x,vpos.w), //p0
        FFT_SRC_GET(vpos.z,vpos.y), FFT_SRC_GET(vpos.z,vpos.w)  //p1
    );
    //c.zw = mul(c.zw,twiddle(vpos.w%1,1));             // twiddle (always 1)
    return float4(c.xy+c.zw, c.xy-c.zw);                // butterfly
}
// inv r2c pack n & n+h/2 row together.
float4 ps_hori_inv(float4 vpos ) {
    float hw = FFT_SRC_SIZEX*.5;
    vpos.z   = vpos.x + hw;                             // p1 output index.
    vpos.y   = trunc(vpos.y);                           // n row
    vpos.w   = vpos.y + FFT_SRC_SIZEX*.5;               // n + h/2 row
    float4 c = float4(                                  // n row on Real, n + h/2 row row on Img
        FFT_SRC_GET(vpos.x,vpos.y), FFT_SRC_GET(vpos.x,vpos.w), //p0
        FFT_SRC_GET(vpos.z,vpos.y), FFT_SRC_GET(vpos.z,vpos.w)  //p1
    );
    c    = float4( c.xy+c.zw, c.xy-c.zw);               // butterfly
    c.zw = mul(c.zw,twiddle(-(vpos.x%hw),hw));          // twiddle
    return c*.5;
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

    vpos.xy = trunc(vpos.xy);
    vpos.x *= 2;                            // [0, h/2-1] * 2
    vpos.w  = rBit(vpos.x+1, FFT_SRC_POTY); // src index for p1
    vpos.x  = rBit(vpos.x,   FFT_SRC_POTY); // src index for p0
    int sel = vpos.x % 2;                   // is accessed index on odd or even row.
    vpos.xw = vpos.xw / 2;                  // (either both odd or even with bit reversed)

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
float4 ps_vert_inv( sampler2D texIn, float4 vpos) {

    int4 sel;
    float2 hs = float2(FFT_SRC_SIZEX,FFT_SRC_SIZEY)*.5;
    vpos.x    = trunc(vpos.x);
    sel.yz    = vpos.y;
    sel.w     = FFT_SRC_SIZEX-sel.z;

    // vpos.y = [0, w/2-1] < src buffer size X
    float4 c;
    sel.z = (vpos.y = rBit(sel.z,FFT_SRC_POTX))/hs.x, vpos.y %= hs.x;   // descramble input coord
    c.xy  = float2x2(tex2Dfetch(texIn, vpos.yxzz))[sel.z];              // p0,p1
    sel.w = (vpos.y = rBit(sel.w,FFT_SRC_POTX))/hs.x, vpos.y %= hs.x;   // complex conjugate
    c.zw  = float2x2(tex2Dfetch(texIn, vpos.yxzz))[sel.y? sel.w:1];     // sel zw when vpos.y == 0
    c.w   = -c.w;  // p0*,p1*.
    // if vpos.y = 0, p0[0]=c.x, p1[0]=c.y; p0[w/2]=c.z, p1[w/2]=c.w;
    c     = sel.y? float4(c.xy+c.zw,c.yz-c.wx)*.5: c.xzyw;              // p0: (X+cX)*.5;  p1: i(cX-X)*.5
    c     = float4(c.xy+c.zw,c.xy-c.zw);                                // butterfly.
    c.zw  = mul(c.zw,twiddle(-(vpos.x%hs.y),hs.y));                     // twiddle
    return c*.5;
}


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Pass Setup
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ps ctor; < t: output buffer > < r: stage number (0 index) >
#define PS_FFT( t, r, z ) \
float4 ps_fft_##t##r( float4 vpos : SV_POSITION ) : SV_TARGET { \
    return ps_fft( samp##t, vpos, r); } \
float4 ps_ifft_##t##r( float4 vpos : SV_POSITION ) : SV_TARGET { \
    return ps_fft_inv( samp##t, vpos, (z)-(r)-1); }

// horizontal init pass
float4 ps_fft_BufHA0 (float4 vpos : SV_POSITION) : SV_TARGET {return ps_hori(vpos);}
float4 ps_ifft_BufHA0(float4 vpos : SV_POSITION) : SV_TARGET {return ps_hori_inv(vpos);}

// vertical init pass
#if (FFT_SRC_POTX & 1)
#   define sampBufH sampBufHB
#else
#   define sampBufH sampBufHA
#endif
float4 ps_fft_BufVA0 (float4 vpos : SV_POSITION) : SV_TARGET {return ps_vert(sampBufH,vpos);}
float4 ps_ifft_BufVA0(float4 vpos : SV_POSITION) : SV_TARGET {return ps_vert_inv(sampBufH,vpos);}
#undef sampBufH

PS_FFT(BufHB,  1, FFT_SRC_POTX)  PS_FFT(BufVB,  1, FFT_SRC_POTY)
PS_FFT(BufHA,  2, FFT_SRC_POTX)  PS_FFT(BufVA,  2, FFT_SRC_POTY)
PS_FFT(BufHB,  3, FFT_SRC_POTX)  PS_FFT(BufVB,  3, FFT_SRC_POTY)
PS_FFT(BufHA,  4, FFT_SRC_POTX)  PS_FFT(BufVA,  4, FFT_SRC_POTY)
PS_FFT(BufHB,  5, FFT_SRC_POTX)  PS_FFT(BufVB,  5, FFT_SRC_POTY)
PS_FFT(BufHA,  6, FFT_SRC_POTX)  PS_FFT(BufVA,  6, FFT_SRC_POTY)
PS_FFT(BufHB,  7, FFT_SRC_POTX)  PS_FFT(BufVB,  7, FFT_SRC_POTY)
PS_FFT(BufHA,  8, FFT_SRC_POTX)  PS_FFT(BufVA,  8, FFT_SRC_POTY)
PS_FFT(BufHB,  9, FFT_SRC_POTX)  PS_FFT(BufVB,  9, FFT_SRC_POTY)
PS_FFT(BufHA, 10, FFT_SRC_POTX)  PS_FFT(BufVA, 10, FFT_SRC_POTY)
PS_FFT(BufHB, 11, FFT_SRC_POTX)  PS_FFT(BufVB, 11, FFT_SRC_POTY)
PS_FFT(BufHA, 12, FFT_SRC_POTX)  PS_FFT(BufVA, 12, FFT_SRC_POTY)

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  technique
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#include "macro_cond.fxh"
#define COND_PASS( a,b,e,d,c ) COND( a##d { b (e,  d); }, c , d )
#undef  COND_EXP
#define COND_EXP(k)     pass k
#define PASS_FFT(t,r)   VertexShader=vs_fft; PixelShader=ps_fft_##t##r;  RenderTarget=tex##t
#define PASS_IFFT(t,r)  VertexShader=vs_fft; PixelShader=ps_ifft_##t##r; RenderTarget=tex##t

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
technique inverse {

#   define EVAL_EXP FFT_SRC_POTX
#   include "macro_eval.fxh"   // evaluate preprocessor expression
    COND_PASS( hori, PASS_IFFT, BufHA,  0, EVAL_RES) // ps_fft_BufHA0
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
} // fft_r2c
#endif // _FFT_R2C_FXH_
