//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//  fft_c2c.fxh header demo usage for ReShade.
//                                              Oct.3.2020 by kingeric1992
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "macro_common.fxh"

float2 shift(float2 vpos, float2 c) { return vpos += vpos<c? c:-c; }
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Setup
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// setup the getter & input dimension for input matrix
#define FFT_SRC_SIZEX       256
#define FFT_SRC_SIZEY       256
#define FFT_SRC_GET(_x,_y) (tex2Dfetch(sampSrc,int4(_x,_y,0,0)).x)
#define FFT_SRC_INV(_x,_y) (get(float2(_x,_y))) // getter for inversed flow

texture2D texSrc    < source  ="lena.png"; >
                    { Width = FFT_SRC_SIZEX; Height = FFT_SRC_SIZEY; Format = R8; }; // src  tex
texture2D texDest   { Width = FFT_SRC_SIZEX; Height = FFT_SRC_SIZEY; Format = R8; }; // dest tex
texture2D texMid    { Width = FFT_SRC_SIZEX; Height = FFT_SRC_SIZEY; Format = RG32F; }; // dest tex
sampler2D sampSrc   { Texture = texSrc; FILTER(POINT); };

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Include FFT lib
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// c2c lib support both forward and inverse flow
#include "fft_c2c.fxh"

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Shaders
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#define NORM(a) (log(a+1)/log(1000))

float4 vs_demo( uint vid : SV_VERTEXID ) : SV_POSITION {
    return float4((vid.xx == uint2(2,1)? float2(-3,3):float2(1,-1)),0,1);
}
float2 ps_forward( float4 vpos : SV_POSITION ) : SV_TARGET {
    return fft_c2c::get(vpos.xy);
}
float ps_inverse( float4 vpos : SV_POSITION ) : SV_TARGET {
    return fft_c2c::get(vpos.xy).x;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Technique
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// view real component of c2c result.

technique c2c_forward_view {
    pass p0 {
        VertexShader    = vs_demo;
        PixelShader     = ps_forward;
        RenderTarget    = texMid;
    }
}
technique c2c_inverse_view {
    pass p1 {
        VertexShader    = vs_demo;
        PixelShader     = ps_inverse;
        RenderTarget    = texDest;
    }
}