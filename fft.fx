//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//  fftlib demo usage for ReShade.
//                                              Oct.10.2020 by kingeric1992
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#include "macro_common.fxh"

float2 shift(float2 vpos, float2 c) { return vpos += vpos<c? c:-c; }
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Setup
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// setup the getter & input dimension for input matrix
#define FFT_SRC_SIZEX       256
#define FFT_SRC_SIZEY       256
#define FFT_SRC_GET(_x, _y) (tex2Dfetch(sampSrc,int4(_x,_y,0,0)).x)

texture2D texSrc    < source  ="lena.png"; >
                    { Width = FFT_SRC_SIZEX; Height = FFT_SRC_SIZEY; Format = R8; };    // src  tex
texture2D texInv    { Width = FFT_SRC_SIZEX; Height = FFT_SRC_SIZEY; Format = RG32F; };    // dest tex
texture2D texDiff   { Width = FFT_SRC_SIZEX; Height = FFT_SRC_SIZEY; Format = RG32F; }; // intermediate tex
texture2D texC2C    { Width = FFT_SRC_SIZEX; Height = FFT_SRC_SIZEY; Format = RG32F; }; // intermediate tex
texture2D texR2C    { Width = FFT_SRC_SIZEX; Height = FFT_SRC_SIZEY; Format = RG32F; }; // intermediate tex
sampler2D sampSrc   { Texture = texSrc; FILTER(POINT); };

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Include FFT lib
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// it does support forward and inverse transform, but doesn't make too much sense
// doing inverse on real sequence?
#define  FFT_USE_FORWARD
#include "fft_r2c.fxh"

#undef  FFT_SRC_INV
#define FFT_SRC_INV(_x,_y)  (fft_r2c::get(float2(_x,_y))) // getter for inversed flow

#define FFT_USE_FORWARD
#define FFT_USE_INVERSE
#include "fft_c2c.fxh"

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  shaders
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#define NORM(a) (log(a+1)/log(1000))

float4 vs_demo( uint vid : SV_VERTEXID ) : SV_POSITION {
    return float4((vid.xx == uint2(2,1)? float2(-3,3):float2(1,-1)),0,1);
}
float2 ps_diff( float4 vpos : SV_POSITION ) : SV_TARGET {
    return abs(fft_r2c::get(vpos.xy) - fft_c2c::get(vpos.xy));
}
float2 ps_inv( float4 vpos : SV_POSITION ) : SV_TARGET { return fft_c2c::getInv(vpos.xy);}
float2 ps_r2c( float4 vpos : SV_POSITION ) : SV_TARGET { return fft_r2c::get(vpos.xy);}
float2 ps_c2c( float4 vpos : SV_POSITION ) : SV_TARGET { return fft_c2c::get(vpos.xy);}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  techniques
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

technique forward_view {
    pass r2c {
        VertexShader    = vs_demo;
        PixelShader     = ps_r2c;
        RenderTarget    = texR2C;
    }
    pass c2c {
        VertexShader    = vs_demo;
        PixelShader     = ps_c2c;
        RenderTarget    = texC2C;
    }
    pass diff {
        VertexShader    = vs_demo;
        PixelShader     = ps_diff;
        RenderTarget    = texDiff;
    }
}
technique inverse_view {
    pass p0 {
        VertexShader    = vs_demo;
        PixelShader     = ps_inv;
        RenderTarget    = texInv;
    }
}