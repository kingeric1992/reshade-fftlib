//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//  fft_r2c.fxh header demo usage for ReShade.
//                                              Oct.3.2020 by kingeric1992
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#if 0
#include "macro_common.fxh"

float2 shift(float2 vpos, float2 _half) { return vpos += vpos<c? c:-c; }
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Setup
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// setup the getter & input dimension for input matrix
#define FFT_SRC_SIZEX       256
#define FFT_SRC_SIZEY       256
#define FFT_SRC_GET(_x, _y) (tex2Dfetch(fft::sampSrc,int4(_x,_y,0,0)).x)

namespace r2cDemo {
    texture2D texSrc    < source  ="fft.jpg"; >
                        { Width = FFT_SRC_SIZEX; Height = FFT_SRC_SIZEY; Format = R8; };        // src  tex
    texture2D texDest   { Width = FFT_SRC_SIZEX; Height = FFT_SRC_SIZEY; Format = R32F; };      // dest tex
    sampler2D sampSrc   { Texture = texSrc; FILTER(POINT); };
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Include FFT lib
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// it does support forward and inverse transform, but doesn't make too much sense
// doing inverse on real sequence?
#include "fft_r2c.fxh"

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Setup
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

float norm( float u) { return log(u + 1)/log(100); }

float4 vs_post( uint vid : SV_VERTEXID ) : SV_POSITION {
    return float4((vid.xx == uint2(2,1)? float2(-3,3):float2(1,-1)),0,1);
}
float ps_post( float4 vpos : SV_POSITION ) : SV_TARGET {
    return norm(fft_r2c::amp(shift(vpos.xy, float2(FFT_SRC_SIZEX, FFT_SRC_SIZEY)*.5));
}

// view the amplitude.
technique fft_forward_view {
    pass p0 {
        VertexShader    = vs_post;
        PixelShader     = ps_post;
        RenderTarget    = fft::texDest;
    }
}
#endif